use std::cmp::{Eq, PartialEq};
use std::iter;
use std::ops::ControlFlow;

use rustc_abi::{ExternAbi, VariantIdx};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::DiagMessage;
use rustc_hir::def::CtorKind;
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::{self as hir, AmbigArg};
use rustc_middle::bug;
use rustc_middle::ty::{
    self, Adt, AdtDef, AdtKind, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt,
};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use rustc_type_ir::{Binder, FnSig};
use tracing::debug;

use super::repr_nullable_ptr;
use crate::lints::{ImproperCTypes, ImproperCTypesLayer, UsesPowerAlignment};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

type Sig<'tcx> = Binder<TyCtxt<'tcx>, FnSig<TyCtxt<'tcx>>>;

/// for a given `extern "ABI"`, tell whether that ABI is *not* considered a FFI boundary
fn fn_abi_is_internal(abi: ExternAbi) -> bool {
    matches!(
        abi,
        ExternAbi::Rust | ExternAbi::RustCall | ExternAbi::RustCold | ExternAbi::RustIntrinsic
    )
}

// TODO: this only exists for debug purposes, remove me before it's upstreamed
macro_rules! printifenv {
    ($($elems:tt)*) => (
        if ::std::env::var("ENUM_ITEMS")
            .unwrap_or(String::from(""))
            .as_str() == "1"
        {
            println!($($elems)*)
        }
    )
}

// a shorthand for an often used lifetime-region normalisation step
#[inline]
fn normalize_if_possible<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty).unwrap_or(ty)
}

// getting the (normalized) type out of a field (for, e.g., an enum variant or a tuple)
#[inline]
fn get_type_from_field<'tcx>(
    cx: &LateContext<'tcx>,
    field: &ty::FieldDef,
    args: GenericArgsRef<'tcx>,
) -> Ty<'tcx> {
    let field_ty = field.ty(cx.tcx, args);
    normalize_if_possible(cx, field_ty)
}

/// Check a variant of a non-exhaustive enum for improper ctypes
/// returns two bools: "we have FFI-unsafety due to non-exhaustive enum" and
/// "we have FFI-unsafety due to a non-exhaustive enum variant"
///
/// We treat `#[non_exhaustive] enum` as "ensure that code will compile if new variants are added".
/// This includes linting, on a best-effort basis. There are valid additions that are unlikely.
///
/// Adding a data-carrying variant to an existing C-like enum that is passed to C is "unlikely",
/// so we don't need the lint to account for it.
/// e.g. going from enum Foo { A, B, C } to enum Foo { A, B, C, D(u32) }.
pub(crate) fn flag_non_exhaustive_variant(
    non_local_def: bool,
    variant: &ty::VariantDef,
) -> (bool, bool) {
    // non_exhaustive suggests it is possible that someone might break ABI
    // see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
    // so warn on complex enums being used outside their crate
    if non_local_def {
        // which is why we only warn about really_tagged_union reprs from https://rust.tf/rfc2195
        // with an enum like `#[repr(u8)] enum Enum { A(DataA), B(DataB), }`
        // but exempt enums with unit ctors like C's (e.g. from rust-bindgen)
        if variant_has_complex_ctor(variant) {
            return (true, false);
        }
    }

    let non_exhaustive_variant_fields = variant.is_field_list_non_exhaustive();
    if non_exhaustive_variant_fields && !variant.def_id.is_local() {
        return (false, true);
    }

    (false, false)
}

fn variant_has_complex_ctor(variant: &ty::VariantDef) -> bool {
    // CtorKind::Const means a "unit" ctor
    !matches!(variant.ctor_kind(), Some(CtorKind::Const))
}

// non_exhaustive suggests it is possible that someone might break ABI
// see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
// so warn on complex enums being used outside their crate
pub(crate) fn non_local_and_non_exhaustive(def: ty::AdtDef<'_>) -> bool {
    def.is_variant_list_non_exhaustive() && !def.did().is_local()
}

/// a way to keep track of what we want to lint for FFI-safety
/// in other words, the nature of the "original item" being checked, and its relation
/// to FFI boundaries
#[derive(Clone, Copy, Debug)]
enum CItemKind {
    /// Imported items in an `extern "C"` block (function declarations, static variables) -> IMPROPER_CTYPES
    ImportedExtern,
    /// `extern "C"` function definitions, to be used elsewhere -> IMPROPER_C_FN_DEFINITIONS,
    /// (TODO: can we detect static variables made to be exported?)
    ExportedFunction,
    /// `extern "C"` function pointers -> IMPROPER_C_CALLBACKS,
    Callback,
    /// `repr(C)` structs/enums/unions -> IMPROPER_CTYPE_DEFINITIONS
    AdtDef,
}

#[derive(Clone, Debug)]
struct FfiUnsafeReason<'tcx> {
    ty: Ty<'tcx>,
    note: DiagMessage,
    help: Option<DiagMessage>,
    inner: Option<Box<FfiUnsafeReason<'tcx>>>,
}

#[derive(Clone, Debug)]
struct FfiUnsafeDetails<'tcx> {
    /// the stack of
    reason: Box<FfiUnsafeReason<'tcx>>,
    /// optionally, override "what is at fault" (function, structure, FnPtr, etc)
    /// (this is in part used to mark that a specific lint will already be emitted.
    /// e.g. when multiple functions use the same, same-crate repr(C) FFI-unsafe struct,
    /// there should only be one lint emitted, pointing to the struct's span).
    override_itemkind: Option<CItemKind>,
    /// override "what is the cause of the fault" to point out the easiest way to fix the unsafety
    /// (e.g.: even if the lint goes into detail as to why a struct is unsafe,
    /// have the first line say that the fault lies in the use of said struct)
    override_cause_ty: Option<Ty<'tcx>>,

    /// also used to prevent duplicate lints, this time for identical types
    /// (a commonly used FnPtr will only have the full explanation linted once, the others will just say "see previous lint");
    #[allow(unused)]
    was_cached: bool,
}

#[derive(Clone, Debug)]
enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe(Vec<FfiUnsafeDetails<'tcx>>),
}

impl<'tcx> FfiResult<'tcx> {
    /// Simplified creation of the FfiUnsafe variant for a single unsafety reason
    fn new_with_reason(ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>) -> Self {
        Self::FfiUnsafe(vec![FfiUnsafeDetails {
            override_cause_ty: None,
            override_itemkind: None,
            was_cached: false,
            reason: Box::new(FfiUnsafeReason { ty, help, note, inner: None }),
        }])
    }
    // fn new_with_reason_and_override(ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>, override_itemkind: CItemKind) -> Self {
    //     Self::FfiUnsafe(vec![FfiUnsafeDetails{
    //         override_itemkind: Some(override_itemkind),
    //         reason: Box::new(FfiUnsafeReason { ty, help, note, inner: None }),
    //         was_cached: false,
    //     }])
    // }

    /// If the FfiUnsafe variant, 'wraps' all reasons,
    /// creating new `FfiUnsafeReason`s, putting the originals as their `inner` fields.
    /// Otherwise, keep unchanged
    fn wrap_all(self, ty: Ty<'tcx>, note: DiagMessage, help: Option<DiagMessage>) -> Self {
        match self {
            Self::FfiUnsafe(this) => {
                let unsafeties = this
                    .into_iter()
                    .map(
                        |FfiUnsafeDetails {
                             reason,
                             was_cached: _,
                             override_itemkind,
                             override_cause_ty,
                         }| {
                            let reason = Box::new(FfiUnsafeReason {
                                ty,
                                help: help.clone(),
                                note: note.clone(),
                                inner: Some(reason),
                            });
                            FfiUnsafeDetails {
                                reason,
                                override_itemkind,
                                override_cause_ty,
                                was_cached: false,
                            }
                        },
                    )
                    .collect::<Vec<_>>();
                Self::FfiUnsafe(unsafeties)
            }
            r @ _ => r,
        }
    }
    /// If the FfiPhantom variant, turns it into a FfiUnsafe version.
    /// Otherwise, keep unchanged.
    fn forbid_phantom(self) -> Self {
        match self {
            Self::FfiSafe | Self::FfiUnsafe(..) => self,
            Self::FfiPhantom(ty) => Self::FfiUnsafe(vec![FfiUnsafeDetails {
                override_itemkind: None,
                override_cause_ty: None,
                was_cached: false,
                reason: Box::new(FfiUnsafeReason {
                    ty,
                    note: fluent::lint_improper_ctypes_only_phantomdata,
                    help: None,
                    inner: None,
                }),
            }]),
        }
    }
}

impl<'tcx> std::ops::AddAssign<FfiResult<'tcx>> for FfiResult<'tcx> {
    fn add_assign(&mut self, other: Self) {
        // note: we shouldn't really encounter FfiPhantoms here, they should be dealt with beforehand
        // still, this function deals with them in a reasonable way, I think

        match (self, other) {
            (Self::FfiUnsafe(myself), Self::FfiUnsafe(mut other_reasons)) => {
                myself.append(&mut other_reasons);
            }
            (Self::FfiUnsafe(_), _) => {
                // nothing to do
            }
            (myself, other @ Self::FfiUnsafe(_)) => {
                *myself = other;
            }
            (Self::FfiPhantom(ref ty1), Self::FfiPhantom(ty2)) => {
                debug!("whoops, both FfiPhantom: self({:?}) += other({:?})", ty1, ty2);
            }
            (myself @ Self::FfiSafe, other @ Self::FfiPhantom(_)) => {
                *myself = other;
            }
            (_, Self::FfiSafe) => {
                // nothing to do
            }
        }
    }
}
impl<'tcx> std::ops::Add<FfiResult<'tcx>> for FfiResult<'tcx> {
    type Output = FfiResult<'tcx>;
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

/// Determine if a type is sized or not, and whether it affects references/pointers/boxes to it
#[derive(Clone, Copy)]
enum TypeSizedness {
    /// type of definite size (pointers are C-compatible)
    Definite,
    /// unsized type because it includes an opaque/foreign type (pointers are C-compatible)
    UnsizedWithExternType,
    /// unsized type for other reasons (slice, string, dyn Trait, closure, ...) (pointers are not C-compatible)
    UnsizedWithMetadata,
    /// not known, usually for placeholder types (Self in non-impl trait functions, type parameters, aliases, the like)
    NotYetKnown,
}

/// what type indirection points to a given type
#[derive(Clone, Copy)]
enum IndirectionType {
    /// box (valid non-null pointer, owns pointee)
    Box,
    /// ref (valid non-null pointer, borrows pointee)
    Ref,
    /// raw pointer (not necessarily non-null or valid. no info on ownership)
    RawPtr,
}

/// Is this type unsized because it contains (or is) a foreign type?
/// (Returns Err if the type happens to be sized after all)
fn get_type_sizedness<'tcx, 'a>(cx: &'a LateContext<'tcx>, ty: Ty<'tcx>) -> TypeSizedness {
    let tcx = cx.tcx;

    // note that sizedness is unrelated to inhabitedness
    if ty.is_sized(tcx, cx.typing_env()) {
        //let is_inh = ty.is_privately_uninhabited(tcx, cx.typing_env());
        TypeSizedness::Definite
    } else {
        // the overall type is !Sized or ?Sized
        match ty.kind() {
            ty::Slice(_) => TypeSizedness::UnsizedWithMetadata,
            ty::Str => TypeSizedness::UnsizedWithMetadata,
            ty::Dynamic(..) => TypeSizedness::UnsizedWithMetadata,
            ty::Foreign(..) => TypeSizedness::UnsizedWithExternType,
            ty::Adt(def, args) => {
                // for now assume: boxes and phantoms don't mess with this
                match def.adt_kind() {
                    AdtKind::Union | AdtKind::Enum => {
                        bug!("unions and enums are necessarily sized")
                    }
                    AdtKind::Struct => {
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                        {
                            return TypeSizedness::UnsizedWithMetadata;
                        }

                        // note: non-exhaustive structs from other crates are not assumed to be ?Sized
                        // for the purpose of sizedness, it seems we are allowed to look at its current contents.

                        if def.non_enum_variant().fields.is_empty() {
                            bug!("an empty struct is necessarily sized");
                        }

                        let variant = def.non_enum_variant();

                        // only the last field may be !Sized (or ?Sized in the case of type params)
                        let last_field = match (&variant.fields).iter().last(){
                            Some(last_field) => last_field,
                            // even nonexhaustive-empty structs from another crate are considered Sized
                            // (eventhough one could add a !Sized field to them)
                            None => bug!("Empty struct should be Sized, right?"), //
                        };
                        let field_ty = get_type_from_field(cx, last_field, args);
                        match get_type_sizedness(cx, field_ty) {
                            s @ (TypeSizedness::UnsizedWithMetadata
                            | TypeSizedness::UnsizedWithExternType
                            | TypeSizedness::NotYetKnown) => s,
                            TypeSizedness::Definite => {
                                bug!("failed to find the reason why struct `{:?}` is unsized", ty)
                            }
                        }
                    }
                }
            }
            ty::Tuple(tuple) => {
                // only the last field may be !Sized (or ?Sized in the case of type params)
                let item_ty: Ty<'tcx> = match tuple.last() {
                    Some(item_ty) => *item_ty,
                    None => bug!("Empty tuple (AKA unit type) should be Sized, right?"),
                };
                let item_ty = normalize_if_possible(cx, item_ty);
                match get_type_sizedness(cx, item_ty) {
                    s @ (TypeSizedness::UnsizedWithMetadata
                    | TypeSizedness::UnsizedWithExternType
                    | TypeSizedness::NotYetKnown) => s,
                    TypeSizedness::Definite => {
                        bug!("failed to find the reason why tuple `{:?}` is unsized", ty)
                    }
                }
            }

            ty_kind @ (ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Array(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnPtr(..)
            | ty::Never
            | ty::Pat(..) // these are (for now) numeric types with a range-based restriction
            ) => {
                // those types are all sized, right?
                bug!(
                    "This ty_kind (`{:?}`) should be sized, yet we are in a branch of code that deals with unsized types.",
                    ty_kind,
                )
            }

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach ty::Alias(ty::Opaque) here.
            ty::Param(..) | ty::Alias(ty::Opaque | ty::Projection | ty::Inherent, ..) => {
                return TypeSizedness::NotYetKnown;
            }

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Alias(ty::Weak, ..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Placeholder(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }
}

#[allow(non_snake_case)]
mod CTypesVisitorStateFlags {
    pub(super) const NO_FLAGS: u8 = 0b0000;
    /// for use in functions in general
    pub(super) const FUNC: u8 = 0b0010;
    /// for variables in function returns (implicitly: not for static variables)
    pub(super) const FN_RETURN: u8 = 0b0100;
    /// for variables in functions which are defined in rust (implicitly: not for static variables)
    pub(super) const FN_DEFINED: u8 = 0b1000;
    /// for things which are not quite contrete yet (struct/enum/union definitions, method declarations in traits)
    pub(super) const THEORETICAL: u8 = 0b0001;
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CTypesVisitorState {
    // uses bitflags from CTypesVisitorStateFlags
    StaticTy = CTypesVisitorStateFlags::NO_FLAGS,
    ArgumentTyInDefinition = CTypesVisitorStateFlags::FUNC | CTypesVisitorStateFlags::FN_DEFINED,
    ReturnTyInDefinition = CTypesVisitorStateFlags::FUNC
        | CTypesVisitorStateFlags::FN_DEFINED
        | CTypesVisitorStateFlags::FN_RETURN,
    // ArgumentTyInTraitDeclaration = CTypesVisitorStateFlags::FUNC | CTypesVisitorStateFlags::FN_DEFINED | CTypesVisitorStateFlags::THEORETICAL,
    // ReturnTyInTraitDeclaration = CTypesVisitorStateFlags::FUNC | CTypesVisitorStateFlags::FN_DEFINED | CTypesVisitorStateFlags::THEORETICAL | CTypesVisitorStateFlags::FN_RETURN,
    ArgumentTyInDeclaration = CTypesVisitorStateFlags::FUNC,
    ReturnTyInDeclaration = CTypesVisitorStateFlags::FUNC | CTypesVisitorStateFlags::FN_RETURN,
    AdtDef = CTypesVisitorStateFlags::THEORETICAL,
}

impl CTypesVisitorState {
    /// whether the type is used (directly or not) in a static variable
    fn is_in_function(self) -> bool {
        use CTypesVisitorStateFlags::*;
        ((self as u8) & FUNC) != 0
    }
    /// whether the type is used (directly or not) in a function, in return position
    fn is_in_function_return(self) -> bool {
        use CTypesVisitorStateFlags::*;
        let ret = ((self as u8) & FN_RETURN) != 0;
        #[cfg(debug_assertions)]
        if ret {
            assert!(self.is_in_function());
        }
        ret
    }
    /// whether the type is used (directly or not) in a defined function
    /// in other words, whether or not we allow non-FFI-safe types behind a C pointer,
    /// to be treated as an opaque type on the other side of the FFI boundary
    fn is_in_defined_function(self) -> bool {
        use CTypesVisitorStateFlags::*;
        let ret = ((self as u8) & FN_DEFINED) != 0;
        #[cfg(debug_assertions)]
        if ret {
            assert!(self.is_in_function());
        }
        ret
    }

    /// whether the type is currently being defined
    fn is_being_defined(self) -> bool {
        self == Self::AdtDef
    }

    /// whether the value for that type might come from the non-rust side of a FFI boundary
    fn value_may_be_unchecked(self) -> bool {
        // function declarations are assumed to be rust-caller, non-rust-callee
        // function definitions are assumed to be maybe-not-rust-caller, rust-callee
        // FnPtrs are... well, nothing's certain about anything. (FIXME need more flags in enum?)
        // Same with statics.

        if self.is_being_defined() {
            // some ADTs are only used to go through the FFI boundary in one direction,
            // so let's not make hasty judgement
            false
        } else if !self.is_in_function() {
            true
        } else if self.is_in_defined_function() {
            !self.is_in_function_return()
        } else {
            self.is_in_function_return()
        }
        // TODO: FnPtr
    }
}

/// visitor used to recursively traverse MIR types and evaluate FFI-safety
/// It uses ``check_*`` methods as entrypoints to be called elsewhere,
/// and ``visit_*`` methods to recurse
struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    /// used to store a cache of FfiResults that directly match an already-raised lint
    #[allow(unused)]
    cross_lint_state: &'a mut ImproperCTypesLint<'tcx>,
    /// type cache to prevent infinite recursion
    ty_cache: FxHashSet<Ty<'tcx>>,
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    fn new(lint: &'a mut ImproperCTypesLint<'tcx>, cx: &'a LateContext<'tcx>) -> Self {
        Self { cx, cross_lint_state: lint, ty_cache: FxHashSet::default() }
    }

    /// wrap around code that generates FfiResults "from a different cause".
    /// for instance, if we have a repr(C) struct in a function's argument, FFI insafeties inside the struct
    /// are to be blamed on the struct and not the function.
    /// This is where we use this wrapper, to tell "all FFI-unsafeties in there are caused by this `ty`"
    ///
    /// FfiResult::FfiUnsafe(FfiUnsafeDetails{..}) values returned by this function have ``override_itemkind`` set,
    /// as well as ``is cached`` on every run but the first on a given ``ty``, to help with lint deduplication.
    fn with_itemkind_override_and_cache(
        &mut self,
        override_cause_ty: Option<Ty<'tcx>>,
        citemkind_override: CItemKind,
        get_ffires: impl FnOnce(&mut Self) -> FfiResult<'tcx>,
    ) -> FfiResult<'tcx> {
        //use std::collections::hash_map::Entry;
        use FfiResult::*;

        // if let Some(res) = self.cross_lint_state.past_lints.get(&ty) {
        //     // if this type has already been visited, then no sense in visiting it twice.
        //     return res.clone();
        // }

        let mut ffires = get_ffires(self);

        if let FfiUnsafe(ref mut details) = ffires {
            details.iter_mut().for_each(|detail| {
                detail.override_itemkind = Some(citemkind_override);
                detail.override_cause_ty = override_cause_ty;
            });
        }
        // match self.cross_lint_state.past_lints.entry(ty) {
        //     Entry::Vacant(entry) => {
        //         let mut cached_res = ffires.clone();
        //         if let FfiUnsafe(ref mut details) = cached_res {
        //             details.iter_mut().for_each(|detail|{detail.was_cached = true});
        //         }
        //         entry.insert(cached_res);
        //     },
        //     // Occupied can only occur here if we have a recursive type of sorts. e.g.:
        //     // - recursive structs through a pointee
        //     // - FnPtr type with itself as an argument or return type (...is somebody making monads in rust?)
        //     Entry::Occupied(_) => {}
        // }
        ffires
    }

    /// Checks whether an `extern "ABI" fn` function pointer is indeed FFI-safe to call
    fn visit_fnptr(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
        sig: Sig<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;
        debug_assert!(!fn_abi_is_internal(sig.abi()));

        // if outer_ty.is_some() || !state.is_being_defined() then this enum is visited in the middle of another lint,
        // so we override the "cause type" of the lint
        // (for more detail, see comment in ``visit_struct_union`` before its call to ``with_itemkind_override_and_cache``)
        let override_cause_ty =
            if state.is_being_defined() { outer_ty.and(Some(ty)) } else { Some(ty) };
        self.with_itemkind_override_and_cache(override_cause_ty, CItemKind::Callback, |this| {
            let sig = this.cx.tcx.instantiate_bound_regions_with_erased(sig);

            let mut all_ffires = FfiSafe;

            for arg in sig.inputs() {
                let ffi_res =
                    this.visit_type(CTypesVisitorState::ArgumentTyInDeclaration, None, *arg);
                all_ffires += ffi_res.forbid_phantom().wrap_all(
                    ty,
                    fluent::lint_improper_ctypes_fnptr_indirect_reason,
                    None,
                );
            }

            let ret_ty = sig.output();

            let ffi_res = this.visit_type(CTypesVisitorState::ReturnTyInDeclaration, None, ret_ty);
            all_ffires += ffi_res.forbid_phantom().wrap_all(
                ty,
                fluent::lint_improper_ctypes_fnptr_indirect_reason,
                None,
            );
            all_ffires
        })
    }

    /// Checks if a simple numeric (int, float) type has an actual portable definition
    /// for the compile target
    fn visit_numeric(&mut self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        // FIXME: for now, this is very incomplete, and seems to assume a x86_64 target
        match ty.kind() {
            ty::Int(ty::IntTy::I128) | ty::Uint(ty::UintTy::U128) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_128bit, None)
            }
            ty::Int(..) | ty::Uint(..) | ty::Float(..) => FfiResult::FfiSafe,
            _ => bug!("visit_numeric is to be called with numeric (int, float) types"),
        }
    }

    /// Return the right help for Cstring and Cstr-linked unsafety
    fn visit_cstr(&mut self, outer_ty: Option<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        debug_assert!(matches!(ty.kind(), ty::Adt(def, _)
            if matches!(
                self.cx.tcx.get_diagnostic_name(def.did()),
                Some(sym::cstring_type | sym::cstr_type)
            )
        ));

        let help = if let Some(outer_ty) = outer_ty {
            match outer_ty.kind() {
                ty::Ref(..) | ty::RawPtr(..) => {
                    if outer_ty.is_mutable_ptr() {
                        fluent::lint_improper_ctypes_cstr_help_mut
                    } else {
                        fluent::lint_improper_ctypes_cstr_help_const
                    }
                }
                ty::Adt(..) if outer_ty.boxed_ty().is_some() => {
                    fluent::lint_improper_ctypes_cstr_help_owned
                }
                _ => fluent::lint_improper_ctypes_cstr_help_unknown,
            }
        } else {
            fluent::lint_improper_ctypes_cstr_help_owned
        };

        FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_cstr_reason, Some(help))
    }

    /// Checks if the given indirection (box,ref,pointer) is "ffi-safe"
    fn visit_indirection(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
        inner_ty: Ty<'tcx>,
        indirection_type: IndirectionType,
    ) -> FfiResult<'tcx> {
        let tcx = self.cx.tcx;

        if let ty::Adt(def, _) = inner_ty.kind() {
            if let Some(diag_name @ (sym::cstring_type | sym::cstr_type)) =
                tcx.get_diagnostic_name(def.did())
            {
                // we have better error messages when checking for C-strings directly
                let mut cstr_res = self.visit_cstr(Some(ty), inner_ty); // always unsafe with one depth-one reason.

                // Cstr pointer have metadata, CString is Sized
                if diag_name == sym::cstr_type {
                    // we need to override the "type" part of `cstr_res`'s only FfiResultReason
                    // so it says that it's the use of the indirection that is unsafe
                    match cstr_res {
                        FfiResult::FfiUnsafe(ref mut reasons) => {
                            reasons.first_mut().unwrap().reason.ty = ty;
                        }
                        _ => unreachable!(),
                    }
                    let note = match indirection_type {
                        IndirectionType::RawPtr => fluent::lint_improper_ctypes_unsized_ptr,
                        IndirectionType::Ref => fluent::lint_improper_ctypes_unsized_ref,
                        IndirectionType::Box => fluent::lint_improper_ctypes_unsized_box,
                    };
                    return cstr_res.wrap_all(ty, note, None);
                } else {
                    return cstr_res;
                }
            }
        }

        // there are three remaining concerns with the pointer:
        // - is the pointer compatible with a C pointer in the first place? (if not, only send that error message)
        // - is the pointee FFI-safe? (it might not matter, see mere lines below)
        // - does the pointer type contain a non-zero assumption, but has a value given by non-rust code?
        // this block deals with the first two.
        let mut ffi_res = match get_type_sizedness(self.cx, inner_ty) {
            TypeSizedness::UnsizedWithExternType | TypeSizedness::Definite => {
                // FIXME:
                // for now, we consider this to be safe even in the case of a FFI-unsafe pointee
                // this is technically only safe if the pointer is never dereferenced on the non-rust
                // side of the FFI boundary, i.e. if the type is to be treated as opaque
                // there are techniques to flag those pointees as opaque, but not always, so we can only enforce this
                // in some cases.
                FfiResult::FfiSafe

                // // there's a nuance on what this lint should do for
                // // function definitions (`extern "C" fn fn_name(...) {...}`)
                // // versus declarations (`unsafe extern "C" {fn fn_name(...);}`).
                // // This is touched upon in https://github.com/rust-lang/rust/issues/66220
                // // and https://github.com/rust-lang/rust/pull/72700
                // //
                // // The big question is: what does "ABI safety" mean? if you have something translated to a C pointer
                // // (which has a stable layout) but points to FFI-unsafe type, is it safe?
                // // On one hand, the function's ABI will match that of a similar C-declared function API,
                // // on the other, dereferencing the pointer on the other side of the FFI boundary will be painful.
                // // In this code, the opinion on is split between function declarations and function definitions,
                // // with the idea that at least one side of the FFI boundary needs to treat the pointee as an opaque type.
                // // For declarations, we see this as unsafe, but for definitions, we see this as safe.
                // //
                // // For extern function declarations, the actual definition of the function is written somewhere else,
                // // meaning the declaration is free to express this opaqueness with an extern type (opaque caller-side) or a std::ffi::c_void (opaque callee-side)
                // // (or other possibly better tricks, see https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs)
                // // For extern function definitions, however, in the case where the type is opaque caller-side, it is not opaque callee-side,
                // // and having the full type information is necessary to compile the function.
                // if state.is_in_defined_function() {
                //     FfiResult::FfiSafe
                // } else {
                //     return self.visit_type(state, Some(ty), inner_ty).forbid_phantom().wrap_all(
                //         ty,
                //         fluent::lint_improper_ctypes_sized_ptr_to_unsafe_type,
                //         None,
                //     );
                // }
            }
            TypeSizedness::NotYetKnown => {
                // types with sizedness NotYetKnown:
                // - Type params (with `variable: impl Trait` shorthand or not)
                //   (function definitions only, let's see how this interacts with monomorphisation)
                // - Self in trait functions/methods
                //   (FIXME note: function 'declarations' there should be treated as definitions)
                // - Opaque return types
                //   (always FFI-unsafe)
                // - non-exhaustive structs/enums/unions from other crates
                //   (always FFI-unsafe)
                // (for the three first, this is unless there is a `+Sized` bound involved)
                //
                // FIXME: on a side note, we should separate 'true' declarations (non-rust code),
                // 'fake' declarations (in traits, needed to be implemented elsewhere), and definitions.
                // (for instance, definitions should worry about &self with Self:?Sized, but fake declarations shouldn't)

                // whether they are FFI-safe or not does not depend on the indirections involved (&Self, &T, Box<impl Trait>),
                // so let's not wrap the current context around a potential FfiUnsafe type param.
                self.visit_type(state, Some(ty), inner_ty)
            }
            TypeSizedness::UnsizedWithMetadata => {
                let help = match inner_ty.kind() {
                    ty::Str => Some(fluent::lint_improper_ctypes_str_help),
                    ty::Slice(_) => Some(fluent::lint_improper_ctypes_slice_help),
                    _ => None,
                };
                let reason = match indirection_type {
                    IndirectionType::RawPtr => fluent::lint_improper_ctypes_unsized_ptr,
                    IndirectionType::Ref => fluent::lint_improper_ctypes_unsized_ref,
                    IndirectionType::Box => fluent::lint_improper_ctypes_unsized_box,
                };
                return FfiResult::new_with_reason(ty, reason, help);
            }
        };

        // and now the third concern (does the pointer type contain a non-zero assumption, and is the value given by non-rust code?)
        // technically, pointers with non-rust-given values could also be misaligned, pointing to the wrong thing, or outright dangling, but we assume they never are
        ffi_res += if state.value_may_be_unchecked() {
            let has_nonnull_assumption = match indirection_type {
                IndirectionType::RawPtr => false,
                IndirectionType::Ref | IndirectionType::Box => true,
            };
            let has_optionlike_wrapper = if let Some(outer_ty) = outer_ty {
                super::is_outer_optionlike_around_ty(self.cx, outer_ty, ty)
            } else {
                false
            };

            if has_nonnull_assumption && !has_optionlike_wrapper {
                FfiResult::new_with_reason(
                    ty,
                    fluent::lint_improper_ctypes_ptr_validity_reason,
                    Some(fluent::lint_improper_ctypes_ptr_validity_help),
                )
            } else {
                FfiResult::FfiSafe
            }
        } else {
            FfiResult::FfiSafe
        };

        ffi_res
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn visit_variant_fields(
        &mut self,
        state: CTypesVisitorState,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;
        let transparent_with_all_zst_fields = if def.repr().transparent() {
            if let Some(field) = super::transparent_newtype_field(self.cx.tcx, variant) {
                // Transparent newtypes have at most one non-ZST field which needs to be checked..
                let field_ty = get_type_from_field(self.cx, field, args);
                let ffi_res = self.visit_type(state, Some(ty), field_ty);

                // checking that this is not an FfiUnsafe due to an unit type:
                // visit_type should be smart enough to not consider it unsafe if called from here
                #[cfg(debug_assertions)]
                if let FfiUnsafe(ref reasons) = ffi_res {
                    if let (1, Some(FfiUnsafeDetails { ref reason, .. })) =
                        (reasons.len(), reasons.first())
                    {
                        let FfiUnsafeReason { ref ty, .. } = reason.as_ref();
                        debug_assert!(!ty.is_unit());
                    }
                }

                return ffi_res;
            } else {
                // ..or have only ZST fields, which is FFI-unsafe (unless those fields are all
                // `PhantomData`).
                true
            }
        } else {
            false
        };

        let mut all_ffires = FfiSafe;
        // We can't completely trust `repr(C)` markings, so make sure the fields are actually safe.
        let mut all_phantom = !variant.fields.is_empty();
        for field in &variant.fields {
            let field_ty = get_type_from_field(self.cx, field, args);
            all_phantom &= match self.visit_type(state, Some(ty), field_ty) {
                FfiPhantom(..) => true,
                r @ (FfiUnsafe { .. } | FfiSafe) => {
                    all_ffires += r;
                    false
                }
            }
        }

        if let FfiUnsafe(details) = all_ffires {
            FfiUnsafe(details).wrap_all(ty, fluent::lint_improper_ctypes_struct_dueto, None)
        } else if all_phantom {
            FfiPhantom(ty)
        } else if transparent_with_all_zst_fields {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_struct_zst, None)
        } else {
            FfiSafe
        }
    }

    fn visit_struct_union(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        debug_assert!(matches!(def.adt_kind(), AdtKind::Struct | AdtKind::Union));

        if !((def.repr().c() && !def.repr().packed()) || def.repr().transparent()) {
            // FIXME packed reprs prevent C compatibility, right?
            // TODO new message idiot
            return FfiResult::new_with_reason(
                ty,
                if def.is_struct() {
                    fluent::lint_improper_ctypes_struct_layout_reason
                } else {
                    fluent::lint_improper_ctypes_union_layout_reason
                },
                if def.is_struct() {
                    Some(fluent::lint_improper_ctypes_struct_layout_help)
                } else {
                    Some(fluent::lint_improper_ctypes_union_layout_help)
                },
            );
        }

        let is_non_exhaustive = def.non_enum_variant().is_field_list_non_exhaustive();
        if is_non_exhaustive && !def.did().is_local() {
            // note: we are not overriding the CItemKind here because this error only occurs
            // on structs/unions from different crates
            return FfiResult::new_with_reason(
                ty,
                if def.is_struct() {
                    fluent::lint_improper_ctypes_struct_non_exhaustive
                } else {
                    fluent::lint_improper_ctypes_union_non_exhaustive
                },
                None,
            );
        }

        // from now on in the function, we lint the actual insides of the struct/union: if something is wrong,
        // then the "fault" comes from inside the struct itself.
        // even if we add more details to the lint, the initial line must specify that the FFI-unsafety is because of the struct
        // - if the struct is from the same crate, there is another warning on its definition anyway
        //   (unless it's about Boxes and references without Option<_>
        //    which is partly why we keep the details as to why that struct is FFI-unsafe)
        // - if the struct is from another crate, then there's not much that can be done anyways
        // because of this, we use ``override_cause_ty`` iff outer_type.is_some().
        // TODO: redo this comment
        // if outer_ty.is_some() || !state.is_being_defined() then this enum is visited in the middle of another lint,
        // so we override the "cause type" of the lint
        // (for more detail, see comment in ``visit_struct_union`` before its call to ``with_itemkind_override_and_cache``)
        let override_cause_ty =
            if state.is_being_defined() { outer_ty.and(Some(ty)) } else { Some(ty) };
        self.with_itemkind_override_and_cache(override_cause_ty, CItemKind::AdtDef, |this| {
            if def.non_enum_variant().fields.is_empty() {
                FfiResult::new_with_reason(
                    ty,
                    if def.is_struct() {
                        fluent::lint_improper_ctypes_struct_fieldless_reason
                    } else {
                        fluent::lint_improper_ctypes_union_fieldless_reason
                    },
                    if def.is_struct() {
                        Some(fluent::lint_improper_ctypes_struct_fieldless_help)
                    } else {
                        Some(fluent::lint_improper_ctypes_union_fieldless_help)
                    },
                )
            } else {
                this.visit_variant_fields(state, ty, def, def.non_enum_variant(), args)
            }
        })
    }

    fn visit_enum(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        debug_assert!(matches!(def.adt_kind(), AdtKind::Enum));
        use FfiResult::*;

        if def.variants().is_empty() {
            // Empty enums are implicitely handled as the never type:
            // FIXME think about the FFI-safety of functions that use that
            return FfiSafe;
        }
        // Check for a repr() attribute to specify the size of the
        // discriminant.
        if !def.repr().c() && !def.repr().transparent() && def.repr().int.is_none() {
            // Special-case types like `Option<extern fn()>` and `Result<extern fn(), ()>`
            if let Some(inner_ty) = repr_nullable_ptr(self.cx.tcx, self.cx.typing_env(), ty) {
                return self.visit_type(state, Some(ty), inner_ty);
            }

            return FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_enum_repr_reason,
                Some(fluent::lint_improper_ctypes_enum_repr_help),
            );
        }

        let non_local_def = non_local_and_non_exhaustive(def);
        // Check the contained variants.

        let (mut nonexhaustive_flag, mut nonexhaustive_variant_flag) = (false, false);
        def.variants().iter().for_each(|variant| {
            let (nonex_enum, nonex_var) = flag_non_exhaustive_variant(non_local_def, variant);
            nonexhaustive_flag |= nonex_enum;
            nonexhaustive_variant_flag |= nonex_var;
        });

        // "nonexhaustive" lints only happen outside of the crate defining the enum, so no CItemKind override
        // (meaning: the fault lies in the function call, not the enum)
        if nonexhaustive_flag {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_non_exhaustive, None)
        } else if nonexhaustive_variant_flag {
            FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_non_exhaustive_variant,
                None,
            )
        } else {
            // if outer_ty.is_some() || !state.is_being_defined() then this enum is visited in the middle of another lint,
            // so we override the "cause type" of the lint
            // (for more detail, see comment in ``visit_struct_union`` before its call to ``with_itemkind_override_and_cache``)
            let override_cause_ty =
                if state.is_being_defined() { outer_ty.and(Some(ty)) } else { Some(ty) };
            self.with_itemkind_override_and_cache(override_cause_ty, CItemKind::AdtDef, |this| {
                def.variants()
                    .iter()
                    .map(|variant| {
                        this.visit_variant_fields(state, ty, def, variant, args)
                            // FIXME: check that enums allow any (up to all) variants to be phantoms?
                            // (previous code says no, but I don't know why? the problem with phantoms is that they're ZSTs, right?)
                            .forbid_phantom()
                    })
                    .reduce(|r1, r2| r1 + r2)
                    .unwrap() // always at least one variant if we hit this branch
            })
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn visit_type(
        &mut self,
        state: CTypesVisitorState,
        outer_ty: Option<Ty<'tcx>>,
        ty: Ty<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        let tcx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !self.ty_cache.insert(ty) {
            return FfiSafe;
        }

        match *ty.kind() {
            ty::Adt(def, args) => {
                if let Some(inner_ty) = ty.boxed_ty() {
                    return self.visit_indirection(
                        state,
                        outer_ty,
                        ty,
                        inner_ty,
                        IndirectionType::Box,
                    );
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                // TODO: dedup lints with def.did().is_local()
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        // I thought CStr (not CString) here could only be reached in non-compiling code:
                        //   - not using an indirection would cause a compile error (this lint *currently* seems to not get triggered on such non-compiling code)
                        //   - and using one would cause the lint to catch on the indirection before reaching its pointee
                        // but function *pointers* don't seem to have the same no-unsized-parameters requirement to compile
                        if let Some(sym::cstring_type | sym::cstr_type) =
                            tcx.get_diagnostic_name(def.did())
                        {
                            return self.visit_cstr(outer_ty, ty);
                        }
                        self.visit_struct_union(state, outer_ty, ty, def, args)
                    }
                    AdtKind::Enum => self.visit_enum(state, outer_ty, ty, def, args),
                }
            }

            ty::Char => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_char_reason,
                Some(fluent::lint_improper_ctypes_char_help),
            ),

            ty::Pat(pat_ty, _) => {
                if state.value_may_be_unchecked() {
                    // you would think that int-range pattern types that exclude 0 would have Option layout optimisation
                    // they don't (see tests/ui/type/pattern_types/range_patterns.stderr)
                    // so there's no need to allow Option<pattern_type!(u32 in 1..)>.
                    debug_assert!(matches!(
                        pat_ty.kind(),
                        ty::Int(..) | ty::Uint(..) | ty::Float(..)
                    ));
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_pat_intrange_reason,
                        Some(fluent::lint_improper_ctypes_pat_intrange_help),
                    )
                } else if let ty::Int(_) | ty::Uint(_) = pat_ty.kind() {
                    self.visit_numeric(pat_ty)
                } else {
                    bug!(
                        "this lint was written when pattern types could only be integers constrained to ranges"
                    )
                }
            }

            // types which likely have a stable representation, depending on the target architecture
            ty::Int(..) | ty::Uint(..) | ty::Float(..) => self.visit_numeric(ty),

            // Primitive types with a stable representation.
            ty::Bool | ty::Never => FfiSafe,

            ty::Slice(_) => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_slice_reason,
                Some(fluent::lint_improper_ctypes_slice_help),
            ),

            ty::Dynamic(..) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_dyn, None)
            }

            ty::Str => FfiResult::new_with_reason(
                ty,
                fluent::lint_improper_ctypes_str_reason,
                Some(fluent::lint_improper_ctypes_str_help),
            ),

            ty::Tuple(tuple) => {
                let empty_and_safe = if tuple.is_empty() {
                    if let Some(outer_ty) = outer_ty {
                        match outer_ty.kind() {
                            // `()` fields are FFI-safe!
                            ty::Adt(..) => true,
                            ty::RawPtr(..) => true,
                            // most of those are not even reachable,
                            // but let's not worry about checking that here
                            _ => false,
                        }
                    } else {
                        // C functions can return void
                        state.is_in_function_return()
                    }
                } else {
                    false
                };

                if empty_and_safe {
                    FfiSafe
                } else {
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_tuple_reason,
                        Some(fluent::lint_improper_ctypes_tuple_help),
                    )
                }
            }

            ty::RawPtr(ty, _)
                if match ty.kind() {
                    ty::Tuple(tuple) => tuple.is_empty(),
                    _ => false,
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(inner_ty, _) => {
                return self.visit_indirection(
                    state,
                    outer_ty,
                    ty,
                    inner_ty,
                    IndirectionType::RawPtr,
                );
            }
            ty::Ref(_, inner_ty, _) => {
                return self.visit_indirection(state, outer_ty, ty, inner_ty, IndirectionType::Ref);
            }

            ty::Array(inner_ty, _) => {
                if outer_ty.is_none() && state.is_in_function() {
                    // C doesn't really support passing arrays by value - the only way to pass an array by value
                    // is through a struct.
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_array_reason,
                        Some(fluent::lint_improper_ctypes_array_help),
                    )
                } else {
                    self.visit_type(state, Some(ty), inner_ty)
                }
            }

            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if fn_abi_is_internal(sig.abi()) {
                    FfiResult::new_with_reason(
                        ty,
                        fluent::lint_improper_ctypes_fnptr_reason,
                        Some(fluent::lint_improper_ctypes_fnptr_help),
                    )
                } else {
                    self.visit_fnptr(state, outer_ty, ty, sig)
                }
            }

            ty::Foreign(..) => FfiSafe,

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach this branch.
            // TODO: to help with the generation of multiple lints from single types,
            // can we rely solely on this and discard the separate check for opaque types?
            // I think so but I don't have an exact list of how they arise.
            // nope, it seems that tests/ui/lint/opaque-ty-ffi-normalization-cycle.rs relies on pre-normalisation checking
            // ...but is it the behaviour we want?
            ty::Alias(ty::Opaque, ..) => {
                FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_opaque, None)
            }

            // `extern "C" fn` function definitions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            // function pointers can do the same
            ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent, ..) => FfiSafe,

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Alias(ty::Weak, ..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Placeholder(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }

    #[allow(dead_code)]
    fn check_for_opaque_ty(&mut self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        struct ProhibitOpaqueTypes;
        impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for ProhibitOpaqueTypes {
            type Result = ControlFlow<Ty<'tcx>>;

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if !ty.has_opaque_types() {
                    return ControlFlow::Continue(());
                }

                if let ty::Alias(ty::Opaque, ..) = ty.kind() {
                    ControlFlow::Break(ty)
                } else {
                    ty.super_visit_with(self)
                }
            }
        }

        if let Some(ty) = ty.visit_with(&mut ProhibitOpaqueTypes).break_value() {
            FfiResult::new_with_reason(ty, fluent::lint_improper_ctypes_opaque, None)
        } else {
            FfiResult::FfiSafe
        }
    }

    fn check_for_type(&mut self, state: CTypesVisitorState, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        //use FfiResult::*;
        let ty = normalize_if_possible(self.cx, ty);

        // TODO dedup upaque_ty logic across visits
        //let mut ffi_res = self.check_for_opaque_ty(ty);

        let ffi_res = self.visit_type(state, None, ty);

        // remove the lints that we detect have been raised prior
        // let consider_safe_anyway = if let FfiUnsafe(ref mut reasons) = ffi_res {
        //     reasons.retain(|FfiUnsafeDetails{override_itemkind,reason: _, was_cached: _}|{override_itemkind.is_none()});
        //     reasons.len() == 0
        // } else {
        //     false
        // };
        // if consider_safe_anyway {
        //     ffi_res = FfiSafe
        // }

        ffi_res
    }

    fn check_for_fnptr(&mut self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = normalize_if_possible(self.cx, ty);

        // TODO dedup upaque_ty logic across visits
        // match self.check_for_opaque_ty(ty) {
        //     FfiResult::FfiSafe => (),
        //     ffi_res @ _ => return ffi_res,
        // }

        match *ty.kind() {
            ty::FnPtr(sig_tys, hdr) => {
                let sig = sig_tys.with(hdr);
                if fn_abi_is_internal(sig.abi()) {
                    bug!(
                        "expected to inspect the type of an `extern \"ABI\"` FnPtr, not an internal-ABI one"
                    )
                } else {
                    self.visit_fnptr(CTypesVisitorState::AdtDef, None, ty, sig)
                }
            }
            r @ _ => {
                bug!("expected to inspect the type of an `extern \"ABI\"` FnPtr, not {:?}", r,)
            }
        }
    }

    fn check_for_adtdef(&mut self, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        let ty = normalize_if_possible(self.cx, ty);

        // TODO dedup upaque_ty logic across visits
        // match self.check_for_opaque_ty(ty) {
        //     FfiResult::FfiSafe => (),
        //     ffi_res @ _ => return ffi_res,
        // }

        match *ty.kind() {
            ty::Adt(def, args) => {
                if !def.did().is_local() {
                    bug!(
                        "check_adtdef expected to visit a locally-defined struct/enum/union not {:?}",
                        def
                    );
                }
                // TODO: check how this will behave when checking the stdlib, especially wrt Box,PhantomData and CStr (CString is not repr(C))

                // question: how does this behave when running for "special" ADTs in the stdlib?
                // answer: none of CStr, CString, Box, and PhantomData are repr(C)
                let state = CTypesVisitorState::AdtDef;
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        self.visit_struct_union(state, None, ty, def, args)
                    }
                    AdtKind::Enum => self.visit_enum(state, None, ty, def, args),
                }
            }
            r @ _ => {
                bug!("expected to inspect the type of an `extern \"ABI\"` FnPtr, not {:?}", r,)
            }
        }
    }

    fn check_arg_for_power_alignment(&self, cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
        // Structs (under repr(C)) follow the power alignment rule if:
        //   - the first field of the struct is a floating-point type that
        //     is greater than 4-bytes, or
        //   - the first field of the struct is an aggregate whose
        //     recursively first field is a floating-point type greater than
        //     4 bytes.
        let tcx = cx.tcx;
        if tcx.sess.target.os != "aix" {
            return false;
        }
        if ty.is_floating_point() && ty.primitive_size(tcx).bytes() > 4 {
            return true;
        } else if let Adt(adt_def, _) = ty.kind()
            && adt_def.is_struct()
        {
            let struct_variant = adt_def.variant(VariantIdx::ZERO);
            // Within a nested struct, all fields are examined to correctly
            // report if any fields after the nested struct within the
            // original struct are misaligned.
            for struct_field in &struct_variant.fields {
                let field_ty = tcx.type_of(struct_field.did).instantiate_identity();
                if self.check_arg_for_power_alignment(cx, field_ty) {
                    return true;
                }
            }
        }
        return false;
    }

    fn check_struct_for_power_alignment(
        &self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>,
        adt_def: AdtDef<'tcx>,
    ) {
        debug_assert!(adt_def.repr().c() && !adt_def.repr().packed());
        if cx.tcx.sess.target.os == "aix" && !adt_def.all_fields().next().is_none() {
            let struct_variant_data = item.expect_struct().0;
            for (index, first_field_def) in struct_variant_data.fields().iter().enumerate() {
                // Struct fields (after the first field) are checked for the
                // power alignment rule, as fields after the first are likely
                // to be the fields that are misaligned.
                if index != 0 {
                    let def_id = first_field_def.def_id;
                    let ty = cx.tcx.type_of(def_id).instantiate_identity();
                    if self.check_arg_for_power_alignment(cx, ty) {
                        cx.emit_span_lint(
                            USES_POWER_ALIGNMENT,
                            first_field_def.span,
                            UsesPowerAlignment,
                        );
                    }
                }
            }
        }
    }
}

/// common structure for functionality that is shared
/// between all `ImproperC*` lint pass structs
#[derive(Clone)]
pub(crate) struct ImproperCTypesLint<'tcx> {
    #[allow(unused)]
    past_lints: FxHashMap<Ty<'tcx>, FfiResult<'tcx>>,
}

impl<'tcx> ImproperCTypesLint<'tcx> {
    pub(crate) fn new() -> Self {
        Self { past_lints: FxHashMap::default() }
    }

    /// Find and check any fn-ptr types with external ABIs in `ty`.
    /// For example, `Option<extern "C" fn()>` checks `extern "C" fn()`
    fn check_type_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        hir_ty: &hir::Ty<'tcx>,
        ty: Ty<'tcx>,
    ) {
        struct FnPtrFinder<'tcx> {
            spans: Vec<Span>,
            tys: Vec<Ty<'tcx>>,
        }

        impl<'tcx> hir::intravisit::Visitor<'_> for FnPtrFinder<'tcx> {
            fn visit_ty(&mut self, ty: &'_ hir::Ty<'_, AmbigArg>) {
                debug!(?ty);
                if let hir::TyKind::BareFn(hir::BareFnTy { abi, .. }) = ty.kind
                    && !fn_abi_is_internal(*abi)
                {
                    self.spans.push(ty.span);
                }

                hir::intravisit::walk_ty(self, ty)
            }
        }

        impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for FnPtrFinder<'tcx> {
            type Result = ControlFlow<Ty<'tcx>>;

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                if let ty::FnPtr(_, hdr) = ty.kind()
                    && !fn_abi_is_internal(hdr.abi)
                {
                    self.tys.push(ty);
                }

                ty.super_visit_with(self)
            }
        }

        let mut visitor = FnPtrFinder { spans: Vec::new(), tys: Vec::new() };
        ty.visit_with(&mut visitor);
        visitor.visit_ty_unambig(hir_ty);

        let all_types = iter::zip(visitor.tys.drain(..), visitor.spans.drain(..));
        all_types.for_each(|(fn_ptr_ty, span)| {
            // FIXME this will probably lead to error deduplication: fix this
            let mut visitor = ImproperCTypesVisitor::new(self, cx);
            let ffi_res = visitor.check_for_fnptr(fn_ptr_ty);

            // even in function *definitions*, `FnPtr`s are always function declarations ...right?
            // (FIXME: we can't do that yet because one of rustc's crates can't compile if we do)
            self.process_ffi_result(cx, span, ffi_res, CItemKind::Callback)
        });
    }

    /// For a function that doesn't need to be "ffi-safe", look for fn-ptr argument/return types
    /// that need to be checked for ffi-safety
    fn check_fn_for_external_abi_fnptr(
        &mut self,
        cx: &LateContext<'tcx>,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_type_for_external_abi_fnptr(cx, input_hir, *input_ty);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            self.check_type_for_external_abi_fnptr(cx, ret_hir, sig.output());
        }
    }

    /// For a local definition of a #[repr(C)] struct/enum/union, check that it is indeed FFI-safe
    fn check_reprc_adt(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>,
        adt_def: AdtDef<'tcx>,
    ) {
        debug_assert!(adt_def.repr().c() && !adt_def.repr().packed());

        let ty = cx.tcx.type_of(item.owner_id).instantiate_identity();
        let mut visitor = ImproperCTypesVisitor::new(self, cx);

        // match item.kind {
        //     hir::ItemKind::Struct(var_data, generics) => {
        //         var_data.
        //     },
        //     hir::ItemKind::Enum(def, generics) => {},
        //     hir::ItemKind::Union(var_data) => {},
        //     _ => bug!("not the right itemkind in check_adt")
        // }

        // FIXME: this following call is awkward.
        // is there a way to perform its logic in MIR space rather than HIR space?
        visitor.check_struct_for_power_alignment(cx, item, adt_def);
        let ffi_res = visitor.check_for_adtdef(ty);

        self.process_ffi_result(cx, item.span, ffi_res, CItemKind::AdtDef);
    }

    /// Check that an extern "ABI" static variable is of a ffi-safe type
    fn check_foreign_static(&mut self, cx: &LateContext<'tcx>, id: hir::OwnerId, span: Span) {
        let ty = cx.tcx.type_of(id).instantiate_identity();
        let mut visitor = ImproperCTypesVisitor::new(self, cx);
        let ffi_res = visitor.check_for_type(CTypesVisitorState::StaticTy, ty);
        self.process_ffi_result(cx, span, ffi_res, CItemKind::ImportedExtern);
    }

    /// Check if a function's argument types and result type are "ffi-safe".
    fn check_foreign_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_mode: CItemKind,
        def_id: LocalDefId,
        decl: &'tcx hir::FnDecl<'_>,
    ) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let sig = cx.tcx.instantiate_bound_regions_with_erased(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            let mut visitor = ImproperCTypesVisitor::new(self, cx);
            let visit_state = match fn_mode {
                CItemKind::ExportedFunction => CTypesVisitorState::ArgumentTyInDefinition,
                CItemKind::ImportedExtern => CTypesVisitorState::ArgumentTyInDeclaration,
                _ => bug!("check_foreign_fn cannot be called with CItemKind::{:?}", fn_mode),
            };
            let ffi_res = visitor.check_for_type(visit_state, *input_ty);
            self.process_ffi_result(cx, input_hir.span, ffi_res, fn_mode);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            let mut visitor = ImproperCTypesVisitor::new(self, cx);
            let visit_state = match fn_mode {
                CItemKind::ExportedFunction => CTypesVisitorState::ReturnTyInDefinition,
                CItemKind::ImportedExtern => CTypesVisitorState::ReturnTyInDeclaration,
                _ => bug!("check_foreign_fn cannot be called with CItemKind::{:?}", fn_mode),
            };
            let ffi_res = visitor.check_for_type(visit_state, sig.output());
            self.process_ffi_result(cx, ret_hir.span, ffi_res, fn_mode);
        }
    }

    fn process_ffi_result(
        &self,
        cx: &LateContext<'tcx>,
        sp: Span,
        res: FfiResult<'tcx>,
        fn_mode: CItemKind,
    ) {
        match res {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(
                    cx,
                    ty.clone(),
                    sp,
                    vec![ImproperCTypesLayer {
                        ty,
                        note: fluent::lint_improper_ctypes_only_phantomdata,
                        span_note: None, // filled later
                        help: None,
                        inner_ty: None,
                    }],
                    fn_mode,
                );
            }
            FfiResult::FfiUnsafe(reasons) => {
                printifenv!("unsafe! reason count: {}", reasons.len());
                for details in reasons {
                    printifenv!("{:?}: {:?}", sp, details);
                    // TODO here! use override and co
                    let mut ffiresult_recursor = ControlFlow::Continue(details.reason.as_ref());
                    let mut cimproper_layers: Vec<ImproperCTypesLayer<'_>> = vec![];

                    // this whole while block converts the arbitrarily-deep
                    // FfiResult stack to an ImproperCTypesLayer Vec
                    while let ControlFlow::Continue(FfiUnsafeReason {
                        ref ty,
                        ref note,
                        ref help,
                        ref inner,
                    }) = ffiresult_recursor
                    {
                        if let Some(layer) = cimproper_layers.last_mut() {
                            layer.inner_ty = Some(ty.clone());
                        }
                        cimproper_layers.push(ImproperCTypesLayer {
                            ty: ty.clone(),
                            inner_ty: None,
                            help: help.clone(),
                            note: note.clone(),
                            span_note: None, // filled later
                        });

                        if let Some(inner) = inner {
                            ffiresult_recursor = ControlFlow::Continue(inner.as_ref());
                        } else {
                            ffiresult_recursor = ControlFlow::Break(());
                        }
                    }
                    let cause_ty = if let Some(cause_ty) = details.override_cause_ty {
                        cause_ty
                    } else {
                        // should always have at least one type
                        cimproper_layers.last().unwrap().ty.clone()
                    };
                    self.emit_ffi_unsafe_type_lint(cx, cause_ty, sp, cimproper_layers, fn_mode);
                }
            }
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &self,
        cx: &LateContext<'tcx>,
        ty: Ty<'tcx>,
        sp: Span,
        mut reasons: Vec<ImproperCTypesLayer<'tcx>>,
        fn_mode: CItemKind,
    ) {
        let lint = match fn_mode {
            CItemKind::ImportedExtern => IMPROPER_CTYPES,
            CItemKind::ExportedFunction => IMPROPER_C_FN_DEFINITIONS,
            CItemKind::AdtDef => IMPROPER_CTYPE_DEFINITIONS,
            CItemKind::Callback => IMPROPER_C_CALLBACKS,
        };
        let desc = match fn_mode {
            CItemKind::ImportedExtern => "`extern` block",
            CItemKind::ExportedFunction => "`extern` fn",
            CItemKind::Callback => "`extern` callback",
            CItemKind::AdtDef => "repr(C) type",
        };
        for reason in reasons.iter_mut() {
            reason.span_note = if let ty::Adt(def, _) = reason.ty.kind()
                && let Some(sp) = cx.tcx.hir().span_if_local(def.did())
            {
                Some(sp)
            } else {
                None
            };
        }

        cx.emit_span_lint(lint, sp, ImproperCTypes { ty, desc, label: sp, reasons });
    }
}

/// IMPROPER_CTYPES checks items that are part of a header to a non-rust library
/// Namely, functions and static variables in `extern "<abi>" { }`,
/// if `<abi>` is external (e.g. "C").
///
/// `IMPROPER_C_CALLBACKS` checks for function pointers marked with an external ABI.
/// (fields of type `extern "<abi>" fn`, where e.g. `<abi>` is `C`)
/// these pointers are searched in all other items which contain types
/// (e.g.functions, struct definitions, etc)
///
/// `IMPROPER_C_FN_DEFINITIONS` checks rust-defined functions that are marked
/// to be used from the other side of a FFI boundary.
/// In other words, `extern "<abi>" fn` definitions and trait-method declarations.
/// This only matters if `<abi>` is external (e.g. `C`).
///
/// `IMPROPER_CTYPE_DEFINITIONS` checks structs/enums/unions marked with `repr(C)`,
/// assuming they are to have a fully C-compatible layout.
///
/// and now combinatorics for pointees
impl<'tcx> LateLintPass<'tcx> for ImproperCTypesLint<'tcx> {
    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, it: &hir::ForeignItem<'tcx>) {
        printifenv!("\ncheck_foreign_item: {:?}", it);
        let abi = cx.tcx.hir().get_foreign_abi(it.hir_id());

        match it.kind {
            hir::ForeignItemKind::Fn(sig, _, _) => {
                if fn_abi_is_internal(abi) {
                    self.check_fn_for_external_abi_fnptr(cx, it.owner_id.def_id, sig.decl)
                } else {
                    self.check_foreign_fn(
                        cx,
                        CItemKind::ImportedExtern,
                        it.owner_id.def_id,
                        sig.decl,
                    );
                }
            }
            hir::ForeignItemKind::Static(ty, _, _) if !fn_abi_is_internal(abi) => {
                self.check_foreign_static(cx, it.owner_id, ty.span);
            }
            hir::ForeignItemKind::Static(..) | hir::ForeignItemKind::Type => (),
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        printifenv!("\ncheck_item: {:?}", item);
        match item.kind {
            hir::ItemKind::Static(ty, ..)
            | hir::ItemKind::Const(ty, ..)
            | hir::ItemKind::TyAlias(ty, ..) => {
                self.check_type_for_external_abi_fnptr(
                    cx,
                    ty,
                    cx.tcx.type_of(item.owner_id).instantiate_identity(),
                );
            }
            // See `check_fn` for declarations, `check_foreign_items` for definitions in extern blocks
            hir::ItemKind::Fn { .. } => {}
            hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) | hir::ItemKind::Enum(..) => {
                // looking for extern FnPtr:s is delegated to `check_field_def`.
                let adt_def: AdtDef<'tcx> = cx.tcx.adt_def(item.owner_id.to_def_id());
                //printifenv!("\nadt repr: {:?}", adt_def.repr());

                if adt_def.repr().c() && !adt_def.repr().packed() {
                    self.check_reprc_adt(cx, item, adt_def);
                }
            }

            // Doesn't define something that can contain a external type to be checked.
            hir::ItemKind::Impl(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::GlobalAsm(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::ExternCrate(..) => {}
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx hir::FieldDef<'tcx>) {
        printifenv!("\ncheck_field_def: {:?}", field);
        self.check_type_for_external_abi_fnptr(
            cx,
            field.ty,
            cx.tcx.type_of(field.def_id).instantiate_identity(),
        );
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        _: &'tcx hir::Body<'_>,
        _: Span,
        id: LocalDefId,
    ) {
        printifenv!("\ncheck_fn: {:?}///{:?}", kind, decl);
        use hir::intravisit::FnKind;

        let abi = match kind {
            FnKind::ItemFn(_, _, header, ..) => header.abi,
            FnKind::Method(_, sig, ..) => sig.header.abi,
            _ => return,
        };

        if fn_abi_is_internal(abi) {
            self.check_fn_for_external_abi_fnptr(cx, id, decl);
        } else {
            self.check_foreign_fn(cx, CItemKind::ExportedFunction, id, decl);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, tr_it: &hir::TraitItem<'tcx>) {
        printifenv!("\ncheck_trait_item: {:?}", tr_it);
        match tr_it.kind {
            hir::TraitItemKind::Const(hir_ty, _) => {
                let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
            }
            hir::TraitItemKind::Fn(sig, trait_fn) => {
                match trait_fn {
                    // if the method is defined here,
                    // there is a matching ``LateLintPass::check_fn`` call,
                    // let's not redo that work
                    hir::TraitFn::Provided(_) => return,
                    hir::TraitFn::Required(_) => (),
                }
                let local_id = tr_it.owner_id.def_id;
                if fn_abi_is_internal(sig.header.abi) {
                    self.check_fn_for_external_abi_fnptr(cx, local_id, sig.decl);
                } else {
                    self.check_foreign_fn(cx, CItemKind::ExportedFunction, local_id, sig.decl);
                }
                // sig.span
            }
            hir::TraitItemKind::Type(_, ty_maybe) => {
                if let Some(hir_ty) = ty_maybe {
                    let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                    self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
                }
            }
        }
    }
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, im_it: &hir::ImplItem<'tcx>) {
        printifenv!("\ncheck_impl_item {:?}", im_it);
        // note: we do not skip these checks eventhough they might generate dupe warnings because:
        // - the corresponding trait might be in another crate
        // - the corresponding trait might have some templating involved, so only the impl has the full type information
        match im_it.kind {
            hir::ImplItemKind::Type(hir_ty) => {
                let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
            }
            hir::ImplItemKind::Fn(_sig, _) => {
                // see ``LateLintPass::check_fn``
            }
            hir::ImplItemKind::Const(hir_ty, _) => {
                let ty = cx.tcx.type_of(hir_ty.hir_id.owner.def_id).instantiate_identity();
                self.check_type_for_external_abi_fnptr(cx, hir_ty, ty);
            }
        }
    }
}

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    /// (in other words, declarations of items defined in foreign code)
    ///
    /// ### Example
    ///
    /// ```rust
    /// extern "C" {
    ///     static STATIC: String;
    ///     fn some_func(a:String);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used in `extern`
    /// blocks are safe and follow certain rules to ensure proper
    /// compatibility with the foreign interfaces. This lint is issued when it
    /// detects a probable mistake in a definition. The lint usually should
    /// provide a description of the issue, along with possibly a hint on how
    /// to resolve it.
    IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

declare_lint! {
    /// The `improper_c_fn_definitions` lint detects incorrect use of
    /// [`extern` function] definitions.
    /// (in other words, functions to be used by foreign code)
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub extern "C" fn str_type(p: &str) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    IMPROPER_C_FN_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign item definitions"
}

declare_lint! {
    /// The `improper_c_callbacks` lint detects incorrect use of
    /// [`extern` function] pointers.
    /// (in other words, function signatures for callbacks)
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub fn str_emmiter(call_me_back: extern "C" fn(&str)) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    IMPROPER_C_CALLBACKS,
    Warn,
    "proper use of libc types in foreign-code-compatible callbacks"
}

declare_lint! {
    /// The `improper_ctype_definitions` lint detects incorrect use of types in
    /// foreign-compatible structs, enums, and union definitions.
    ///
    /// ### Example
    ///
    /// ```rust
    /// repr(C) struct StringWrapper{
    ///     length: usize,
    ///     strung: &str,
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types designed to be
    /// compatible with foreign interfaces follow certain rules to be safe.
    /// This lint is issued when it detects a probable mistake in a definition.
    /// The lint usually should provide a description of the issue,
    /// along with possibly a hint on how to resolve it.
    IMPROPER_CTYPE_DEFINITIONS,
    Warn,
    "proper use of libc types when defining foreign-code-compatible structs"
}

declare_lint! {
    /// The `uses_power_alignment` lint detects specific `repr(C)`
    /// aggregates on AIX.
    /// In its platform C ABI, AIX uses the "power" (as in PowerPC) alignment
    /// rule (detailed in https://www.ibm.com/docs/en/xl-c-and-cpp-aix/16.1?topic=data-using-alignment-modes#alignment),
    /// which can also be set for XLC by `#pragma align(power)` or
    /// `-qalign=power`. Aggregates with a floating-point type as the
    /// recursively first field (as in "at offset 0") modify the layout of
    /// *subsequent* fields of the associated structs to use an alignment value
    /// where the floating-point type is aligned on a 4-byte boundary.
    ///
    /// The power alignment rule for structs needed for C compatibility is
    /// unimplementable within `repr(C)` in the compiler without building in
    /// handling of references to packed fields and infectious nested layouts,
    /// so a warning is produced in these situations.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-powerpc64-ibm-aix)
    /// #[repr(C)]
    /// pub struct Floats {
    ///     a: f64,
    ///     b: u8,
    ///     c: f64,
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: repr(C) does not follow the power alignment rule. This may affect platform C ABI compatibility for this type
    ///  --> <source>:5:3
    ///   |
    /// 5 |   c: f64,
    ///   |   ^^^^^^
    ///   |
    ///   = note: `#[warn(uses_power_alignment)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The power alignment rule specifies that the above struct has the
    /// following alignment:
    ///  - offset_of!(Floats, a) == 0
    ///  - offset_of!(Floats, b) == 8
    ///  - offset_of!(Floats, c) == 12
    /// However, rust currently aligns `c` at offset_of!(Floats, c) == 16.
    /// Thus, a warning should be produced for the above struct in this case.
    USES_POWER_ALIGNMENT,
    Warn,
    "Structs do not follow the power alignment rule under repr(C)"
}

impl_lint_pass!(ImproperCTypesLint<'_> => [
    IMPROPER_CTYPES,
    IMPROPER_C_FN_DEFINITIONS,
    IMPROPER_C_CALLBACKS,
    IMPROPER_CTYPE_DEFINITIONS,
    USES_POWER_ALIGNMENT,
]);
