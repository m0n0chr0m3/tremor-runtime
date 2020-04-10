// Copyright 2018-2020, Wayfair GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// NOTE: we use a lot of arguments here, we are aware of that but tough luck
// FIXME: investigate if re-writing would make code better
#![allow(clippy::too_many_arguments)]
// FIXME possible optimisations:
// * P001 [x] re-write `let x = merge x of ... end` to a mutable merge that does not require cloing `x`
// * P002 [x] don't construct data for expressions that return value is never used
// * P003 [x] don't clone as part of a match statement (we should never ever mutate in case or when)
// * P004 [x] turn local variables into a pre-defined vector to improve access
// * P005 [x] turn literals into BorrowedValues so we don't need to re-cast them - constants could be pre-laoded
// * P008 [x] We should not need to clone values in the for comprehension twice.

// FIXME todo
// * 101 [ ] `%{x > 3}` and other comparisons
// * 102 [x] Remove the need for `()` around when clauses that contain binary ops

// NOTE: For env / end
#![allow(clippy::similar_names)]

mod expr;
mod imut_expr;

pub(crate) use self::expr::Cont;
use crate::ast::*;
use crate::errors::*;
use crate::stry;
use crate::EventContext;
use simd_json::borrowed::{Object, Value};
use simd_json::prelude::*;
use simd_json::StaticNode;
use std::borrow::Borrow;
use std::borrow::Cow;
use std::convert::TryInto;
use std::iter::Iterator;

/// constant `true` value
pub const TRUE: Value<'static> = Value::Static(StaticNode::Bool(true));
/// constant `false` value
pub const FALSE: Value<'static> = Value::Static(StaticNode::Bool(false));
/// constant `null` value
pub const NULL: Value<'static> = Value::Static(StaticNode::Null);

macro_rules! static_bool {
    ($e:expr) => {
        #[allow(clippy::if_not_else)]
        {
            if $e {
                Cow::Borrowed(&TRUE)
            } else {
                Cow::Borrowed(&FALSE)
            }
        }
    };
}

/// Interpreter environment
pub struct Env<'run, 'event, 'script>
where
    'script: 'event,
    'event: 'run,
{
    /// Context of the event
    pub context: &'run EventContext,
    /// Constatns
    pub consts: &'run [Value<'event>],
    /// Aggregates
    pub aggrs: &'run [InvokeAggrFn<'script>],
    /// Node metadata
    pub meta: &'run NodeMetas<'script>,
}

/// Local variable stack
#[derive(Default, Debug)]
pub struct LocalStack<'stack> {
    values: Vec<Option<Value<'stack>>>,
}

impl<'stack> LocalStack<'stack> {
    /// Creates a stack with a given size
    pub fn with_size(size: usize) -> Self {
        Self {
            values: vec![None; size],
        }
    }
}

/// The type of an aggregation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AggrType {
    /// This is a normal execution
    Tick,
    /// This us an emit event
    Emit,
}

/// Execution options for a script.
#[derive(Clone, Copy, Debug)]
pub struct ExecOpts {
    /// Is a result needed
    pub result_needed: bool,
    /// If this is an aggregation or a normal execution
    pub aggr: AggrType,
}

impl ExecOpts {
    pub(crate) fn without_result(mut self) -> Self {
        self.result_needed = false;
        self
    }
    pub(crate) fn with_result(mut self) -> Self {
        self.result_needed = true;
        self
    }
}

/// A grouping of data that is commonly passed through interpreter methods.
#[allow(missing_docs)] // FIXME: Don't approve a PR with this line! ;)
#[derive(Clone, Copy)]
pub struct InterpreterContext<'run, 'event, 'script> {
    pub env: &'run Env<'run, 'event, 'script>,
    pub event: &'run Value<'event>,
    pub state: &'run Value<'static>,
    pub meta: &'run Value<'event>,
    pub local: &'run LocalStack<'event>,
}

impl<'run, 'event, 'script> InterpreterContext<'run, 'event, 'script> {
    /// Groups all five fields into a struct
    pub fn of(
        env: &'run Env<'run, 'event, 'script>,
        event: &'run Value<'event>,
        state: &'run Value<'static>,
        meta: &'run Value<'event>,
        local: &'run LocalStack<'event>,
    ) -> Self {
        InterpreterContext {
            env,
            event,
            state,
            meta,
            local,
        }
    }
}

#[inline]
#[allow(clippy::cast_precision_loss)]
fn val_eq<'event>(lhs: &Value<'event>, rhs: &Value<'event>) -> bool {
    // FIXME Consider Tony Garnock-Jones perserves w.r.t. forcing a total ordering
    // across builtin types if/when extending for 'lt' and 'gt' variants
    //
    use Value::*;
    let error = std::f64::EPSILON;
    match (lhs, rhs) {
        (Object(l), Object(r)) => {
            if l.len() == r.len() {
                l.iter()
                    .all(|(k, lv)| r.get(k).map(|rv| val_eq(lv, rv)) == Some(true))
            } else {
                false
            }
        }
        (Array(l), Array(r)) => {
            if l.len() == r.len() {
                l.iter().zip(r.iter()).all(|(l, r)| val_eq(l, r))
            } else {
                false
            }
        }
        (Static(StaticNode::Bool(l)), Static(StaticNode::Bool(r))) => *l == *r,
        (Static(StaticNode::Null), Static(StaticNode::Null)) => true,
        (String(l), String(r)) => *l == *r,
        (l, r) => {
            if let (Some(l), Some(r)) = (l.as_u64(), r.as_u64()) {
                l == r
            } else if let (Some(l), Some(r)) = (l.as_i64(), r.as_i64()) {
                l == r
            } else if let (Some(l), Some(r)) = (l.cast_f64(), r.cast_f64()) {
                (l - r).abs() < error
            } else {
                false
            }
        }
    }
}

#[allow(
    clippy::cognitive_complexity,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
#[inline]
pub(crate) fn exec_binary<'run, 'event, OuterExpr, InnerExpr>(
    outer: &'run OuterExpr,
    inner: &'run InnerExpr,
    node_meta: &'run NodeMetas,
    op: BinOpKind,
    lhs: &Value<'event>,
    rhs: &Value<'event>,
) -> Result<Cow<'run, Value<'event>>>
where
    OuterExpr: BaseExpr,
    InnerExpr: BaseExpr,
    'event: 'run,
{
    // Lazy Heinz doesn't want to write that 10000 times
    // - snot badger - Darach
    use BinOpKind::*;
    use Value::*;
    match (&op, lhs, rhs) {
        (Eq, Static(StaticNode::Null), Static(StaticNode::Null)) => Ok(static_bool!(true)),
        (NotEq, Static(StaticNode::Null), Static(StaticNode::Null)) => Ok(static_bool!(false)),

        // FIXME - do we want this?
        // This is to make sure that == in a expression
        // and a record pattern behaves the same.
        (Eq, l, r) => Ok(static_bool!(val_eq(l, r))),

        (NotEq, l, r) =>
        {
            #[allow(clippy::if_not_else)]
            Ok(static_bool!(!val_eq(l, r)))
        }

        (op, Static(StaticNode::Bool(l)), Static(StaticNode::Bool(r))) => match op {
            And => Ok(static_bool!(*l && *r)),
            Or => Ok(static_bool!(*l || *r)),
            #[allow(clippy::if_not_else)]
            Xor =>
            {
                #[allow(clippy::if_not_else)]
                Ok(static_bool!(*l != *r))
            }
            BitAnd => Ok(static_bool!(*l & *r)),
            BitOr => Ok(static_bool!(*l | *r)),
            BitXor => Ok(static_bool!(*l ^ *r)),
            _ => error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta),
        },
        (op, String(l), String(r)) => match op {
            Gt => Ok(static_bool!(l > r)),
            Gte => Ok(static_bool!(l >= r)),
            Lt => Ok(static_bool!(l < r)),
            Lte => Ok(static_bool!(l <= r)),
            Add => Ok(Cow::Owned(format!("{}{}", *l, *r).into())),
            _ => error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta),
        },
        (op, l, r) => {
            if let (Some(l), Some(r)) = (l.as_u64(), r.as_u64()) {
                match op {
                    BitAnd => Ok(Cow::Owned(Value::from(l & r))),
                    BitOr => Ok(Cow::Owned(Value::from(l | r))),
                    BitXor => Ok(Cow::Owned(Value::from(l ^ r))),
                    Gt => Ok(static_bool!(l > r)),
                    Gte => Ok(static_bool!(l >= r)),
                    Lt => Ok(static_bool!(l < r)),
                    Lte => Ok(static_bool!(l <= r)),
                    Add => Ok(Cow::Owned(Value::from(l + r))),
                    Sub if l >= r => Ok(Cow::Owned(Value::from(l - r))),
                    Sub => {
                        // Handle substraction that would turn this into a negative
                        // to do that we calculate r-i (the inverse) and then
                        // try to turn this into a i64 and negate it;
                        let d = r - l;

                        if let Some(res) = d.try_into().ok().and_then(i64::checked_neg) {
                            Ok(Cow::Owned(Value::from(res)))
                        } else {
                            error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta)
                        }
                    }
                    Mul => Ok(Cow::Owned(Value::from(l * r))),
                    Div => Ok(Cow::Owned(Value::from((l as f64) / (r as f64)))),
                    Mod => Ok(Cow::Owned(Value::from(l % r))),
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    RBitShiftSigned => match (l).checked_shr(r as u32) {
                        Some(n) => Ok(Cow::Owned(Value::from(n))),
                        None => error_invalid_bitshift(outer, inner, &node_meta),
                    },
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    RBitShiftUnsigned => match (l as u64).checked_shr(r as u32) {
                        #[allow(clippy::cast_possible_wrap)]
                        Some(n) => Ok(Cow::Owned(Value::from(n as i64))),
                        None => error_invalid_bitshift(outer, inner, &node_meta),
                    },
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    LBitShift => match (l).checked_shl(r as u32) {
                        Some(n) => Ok(Cow::Owned(Value::from(n))),
                        None => error_invalid_bitshift(outer, inner, &node_meta),
                    },
                    _ => error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta),
                }
            } else if let (Some(l), Some(r)) = (l.as_i64(), r.as_i64()) {
                match op {
                    BitAnd => Ok(Cow::Owned(Value::from(l & r))),
                    BitOr => Ok(Cow::Owned(Value::from(l | r))),
                    BitXor => Ok(Cow::Owned(Value::from(l ^ r))),
                    Gt => Ok(static_bool!(l > r)),
                    Gte => Ok(static_bool!(l >= r)),
                    Lt => Ok(static_bool!(l < r)),
                    Lte => Ok(static_bool!(l <= r)),
                    Add => Ok(Cow::Owned(Value::from(l + r))),
                    Sub => Ok(Cow::Owned(Value::from(l - r))),
                    Mul => Ok(Cow::Owned(Value::from(l * r))),
                    Div => Ok(Cow::Owned(Value::from((l as f64) / (r as f64)))),
                    Mod => Ok(Cow::Owned(Value::from(l % r))),
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    RBitShiftSigned => match (l).checked_shr(r as u32) {
                        Some(n) => Ok(Cow::Owned(Value::from(n))),
                        None => error_invalid_bitshift(outer, inner, &node_meta),
                    },
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    RBitShiftUnsigned => match (l as u64).checked_shr(r as u32) {
                        #[allow(clippy::cast_possible_wrap)]
                        Some(n) => Ok(Cow::Owned(Value::from(n as i64))),
                        None => error_invalid_bitshift(outer, inner, &node_meta),
                    },
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    LBitShift => match (l).checked_shl(r as u32) {
                        Some(n) => Ok(Cow::Owned(Value::from(n))),
                        None => error_invalid_bitshift(outer, inner, &node_meta),
                    },
                    _ => error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta),
                }
            } else if let (Some(l), Some(r)) = (l.cast_f64(), r.cast_f64()) {
                match op {
                    Gte => Ok(static_bool!(l >= r)),
                    Gt => Ok(static_bool!(l > r)),
                    Lt => Ok(static_bool!(l < r)),
                    Lte => Ok(static_bool!(l <= r)),
                    Add => Ok(Cow::Owned(Value::from(l + r))),
                    Sub => Ok(Cow::Owned(Value::from(l - r))),
                    Mul => Ok(Cow::Owned(Value::from(l * r))),
                    Div => Ok(Cow::Owned(Value::from(l / r))),
                    _ => error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta),
                }
            } else {
                error_invalid_binary(outer, inner, *op, lhs, rhs, &node_meta)
            }
        }
    }
}

#[inline]
pub(crate) fn exec_unary<'run, 'event: 'run>(
    op: UnaryOpKind,
    val: &Value<'event>,
) -> Option<Cow<'run, Value<'event>>> {
    // Lazy Heinz doesn't want to write that 10000 times
    // - snot badger - Darach
    use UnaryOpKind::*;
    if let Some(x) = val.as_f64() {
        match &op {
            Minus => Some(Cow::Owned(Value::from(-x))),
            Plus => Some(Cow::Owned(Value::from(x))),
            _ => None,
        }
    } else if let Some(x) = val.as_u64() {
        match &op {
            Minus => x
                .try_into()
                .ok()
                .and_then(i64::checked_neg)
                .map(Value::from)
                .map(Cow::Owned),
            Plus => Some(Cow::Owned(Value::from(x))),
            BitNot => Some(Cow::Owned(Value::from(!x))),
            _ => None,
        }
    } else if let Some(x) = val.as_i64() {
        match &op {
            Minus => x
                .try_into()
                .ok()
                .and_then(i64::checked_neg)
                .map(Value::from)
                .map(Cow::Owned),
            Plus => Some(Cow::Owned(Value::from(x))),
            BitNot => Some(Cow::Owned(Value::from(!x))),
            _ => None,
        }
    } else if let Some(x) = val.as_bool() {
        match &op {
            BitNot | Not => Some(static_bool!(!x)),
            _ => None,
        }
    } else {
        None
    }
}

#[inline]
#[allow(clippy::too_many_lines)]
pub(crate) fn resolve<'run, 'event, 'script, Expr>(
    outer: &'script Expr,
    opts: ExecOpts,
    ictx: InterpreterContext<'run, 'event, 'script>,
    path: &'script Path,
) -> Result<Cow<'run, Value<'event>>>
where
    Expr: BaseExpr,
    'script: 'event,
    'event: 'run,
{
    // Fetch the base of the path
    // TODO: Extract this into a method on `Path`?
    let base_value: &Value = match path {
        Path::Local(lpath) => match ictx.local.values.get(lpath.idx) {
            Some(Some(l)) => l,
            Some(None) => {
                return error_bad_key(
                    outer,
                    lpath,
                    &path,
                    ictx.env.meta.name_dflt(lpath.mid).to_string(),
                    vec![],
                    &ictx.env.meta,
                );
            }

            _ => return error_oops(outer, "Use of unknown local value", &ictx.env.meta),
        },
        Path::Const(lpath) => match ictx.env.consts.get(lpath.idx) {
            Some(v) => v,
            _ => return error_oops(outer, "Use of uninitialized constant", &ictx.env.meta),
        },
        Path::Meta(_path) => ictx.meta,
        Path::Event(_path) => ictx.event,
        Path::State(_path) => ictx.state,
    };

    // Resolve the targeted value by applying all path segments
    let mut subrange: Option<&[Value]> = None;
    let mut current = base_value;
    for segment in path.segments() {
        match segment {
            // Next segment is an identifier: lookup the identifier on `current`, if it's an object
            Segment::Id { mid, key, .. } => {
                if let Some(c) = key.lookup(current) {
                    current = c;
                    subrange = None;
                    continue;
                } else if let Some(o) = current.as_object() {
                    return error_bad_key(
                        outer,
                        segment, //&Expr::dummy(*start, *end),
                        &path,
                        ictx.env.meta.name_dflt(*mid).to_string(),
                        o.keys().map(ToString::to_string).collect(),
                        &ictx.env.meta,
                    );
                } else {
                    return error_need_obj(outer, segment, current.value_type(), &ictx.env.meta);
                }
            }
            // Next segment is an index: index into `current`, if it's an array
            Segment::Idx { idx, .. } => {
                if let Some(a) = current.as_array() {
                    let range_to_consider = subrange.unwrap_or_else(|| a.as_slice());
                    let idx = *idx;

                    if let Some(c) = range_to_consider.get(idx) {
                        current = c;
                        subrange = None;
                        continue;
                    } else {
                        return error_array_out_of_bound(
                            outer,
                            segment,
                            &path,
                            idx..idx,
                            range_to_consider.len(),
                            &ictx.env.meta,
                        );
                    }
                } else {
                    return error_need_arr(outer, segment, current.value_type(), &ictx.env.meta);
                }
            }
            // Next segment is an index range: index into `current`, if it's an array
            Segment::Range {
                range_start,
                range_end,
                ..
            } => {
                if let Some(a) = current.as_array() {
                    let range_to_consider = subrange.unwrap_or_else(|| a.as_slice());

                    // Helper closure for evaluating an expression to an `usize` index
                    let as_index = |expr: &ImutExprInt| -> Result<usize> {
                        let val = stry!(expr.run_with_context(opts, ictx));
                        if let Some(n) = val.as_usize() {
                            Ok(n)
                        } else if val.is_i64() {
                            // TODO: Receive reviewer feedback.
                            // Note for review: ideally, we'd use something like `is_integer()`
                            // to give the best possible error message in this case, but no such
                            // method exists on `Values` (and in `ValueTrait`). I'm using `is_i64`
                            // as a compromise. Perhaps `is_i128` is better, but it ultimately
                            // suffers the same problem. For JSON data, `i64` is plenty fine, but to
                            // be really safe/future-proof, something more robust is needed.
                            error_bad_array_index(
                                outer,
                                expr,
                                &path,
                                val.borrow(),
                                range_to_consider.len(),
                                &ictx.env.meta,
                            )
                        } else {
                            error_need_int(outer, expr, val.value_type(), &ictx.env.meta)
                        }
                    };

                    let range_start = as_index(range_start)?;
                    let range_end = as_index(range_end)?;

                    if range_end < range_start {
                        return error_decreasing_range(
                            outer,
                            segment,
                            &path,
                            range_start,
                            range_end,
                            &ictx.env.meta,
                        );
                    } else if range_end > range_to_consider.len() {
                        return error_array_out_of_bound(
                            outer,
                            segment,
                            &path,
                            range_start..range_end,
                            range_to_consider.len(),
                            &ictx.env.meta,
                        );
                    } else {
                        subrange = Some(&range_to_consider[range_start..range_end]);
                        continue;
                    }
                } else {
                    return error_need_arr(outer, segment, current.value_type(), &ictx.env.meta);
                }
            }
            // Next segment is an expression: run `expr` to know which key it signifies at runtime
            Segment::Element { expr, .. } => {
                let key = stry!(expr.run_with_context(opts, ictx));

                match (current, key.borrow()) {
                    // The segment resolved to an identifier, and `current` is an object: lookup
                    (Value::Object(o), Value::String(id)) => {
                        if let Some(v) = o.get(id) {
                            current = v;
                            subrange = None;
                            continue;
                        } else {
                            return error_bad_key(
                                outer,
                                segment,
                                &path,
                                id.to_string(),
                                o.keys().map(ToString::to_string).collect(),
                                &ictx.env.meta,
                            );
                        }
                    }
                    // The segment did not resolve to an identifier, but `current` is an object: err
                    (Value::Object(_), other) => {
                        return error_need_str(outer, segment, other.value_type(), &ictx.env.meta)
                    }
                    // If `current` is an array, the segment has to be an index
                    (Value::Array(a), idx) => {
                        if let Some(idx) = idx.as_usize() {
                            let range_to_consider = subrange.unwrap_or_else(|| a.as_slice());

                            if let Some(v) = range_to_consider.get(idx) {
                                current = v;
                                subrange = None;
                                continue;
                            } else {
                                return error_array_out_of_bound(
                                    outer,
                                    segment,
                                    &path,
                                    idx..idx,
                                    range_to_consider.len(),
                                    &ictx.env.meta,
                                );
                            }
                        } else {
                            return error_need_int(
                                outer,
                                segment,
                                idx.value_type(),
                                &ictx.env.meta,
                            );
                        }
                    }
                    // The segment resolved to an identifier, but `current` isn't an object: err
                    (other, key) if key.is_str() => {
                        return error_need_obj(outer, segment, other.value_type(), &ictx.env.meta);
                    }
                    // The segment resolved to an index, but `current` isn't an array: err
                    (other, key) if key.is_usize() => {
                        return error_need_arr(outer, segment, other.value_type(), &ictx.env.meta);
                    }
                    // Anything else: err
                    _ => return error_oops(outer, "Bad path segments", &ictx.env.meta),
                }
            }
        }
    }

    if let Some(range_to_consider) = subrange {
        Ok(Cow::Owned(Value::Array(range_to_consider.to_vec())))
    } else {
        Ok(Cow::Borrowed(current))
    }
}

fn merge_values<'run, 'event, 'script, Outer, Inner>(
    outer: &'script Outer,
    inner: &Inner,
    value: &'run mut Value<'event>,
    replacement: &'run Value<'event>,
) -> Result<()>
where
    Outer: BaseExpr,
    Inner: BaseExpr,
    'script: 'event,
    'event: 'run,
{
    if let (Some(rep), Some(map)) = (replacement.as_object(), value.as_object_mut()) {
        for (k, v) in rep {
            if v.is_null() {
                map.remove(k);
            } else if let Some(k) = map.get_mut(k) {
                stry!(merge_values(outer, inner, k, v))
            } else {
                //NOTE: We got to clone here since we're duplicating values
                map.insert(k.clone(), v.clone());
            }
        }
    } else {
        // If one of the two isn't a map we can't merge so we simplhy
        // write the replacement into the target.
        // NOTE: We got to clone here since we're duplicating values
        *value = replacement.clone();
    }
    Ok(())
}

#[inline]
fn patch_value<'run, 'event, 'script, Expr>(
    _outer: &'script Expr,
    opts: ExecOpts,
    ictx: InterpreterContext<'run, 'event, 'script>,
    value: &'run mut Value<'event>,
    expr: &'script Patch,
) -> Result<()>
where
    Expr: BaseExpr,
    'script: 'event,
    'event: 'run,
{
    let patch_expr = expr;
    for op in &expr.operations {
        // NOTE: This if is inside the for loop to prevent obj to be updated
        // between iterations and possibly lead to dangling pointers
        if let Some(ref mut obj) = value.as_object_mut() {
            match op {
                PatchOperation::Insert { ident, expr } => {
                    let new_key = stry!(ident.eval_to_string(opts, ictx));
                    let new_value = stry!(expr.run_with_context(opts, ictx));
                    if obj.contains_key(&new_key) {
                        return error_patch_key_exists(
                            patch_expr,
                            ident,
                            new_key.to_string(),
                            &ictx.env.meta,
                        );
                    } else {
                        obj.insert(new_key, new_value.into_owned());
                    }
                }
                PatchOperation::Update { ident, expr } => {
                    let new_key = stry!(ident.eval_to_string(opts, ictx));
                    let new_value = stry!(expr.run_with_context(opts, ictx));
                    if obj.contains_key(&new_key) {
                        obj.insert(new_key, new_value.into_owned());
                    } else {
                        return error_patch_update_key_missing(
                            patch_expr,
                            expr,
                            new_key.to_string(),
                            &ictx.env.meta,
                        );
                    }
                }
                PatchOperation::Upsert { ident, expr } => {
                    let new_key = stry!(ident.eval_to_string(opts, ictx));
                    let new_value = stry!(expr.run_with_context(opts, ictx));
                    obj.insert(new_key, new_value.into_owned());
                }
                PatchOperation::Erase { ident } => {
                    let new_key = stry!(ident.eval_to_string(opts, ictx));
                    obj.remove(&new_key);
                }
                PatchOperation::Move { from, to } => {
                    let from = stry!(from.eval_to_string(opts, ictx));
                    let to = stry!(to.eval_to_string(opts, ictx));

                    if obj.contains_key(&to) {
                        return error_patch_key_exists(
                            patch_expr,
                            expr,
                            to.to_string(),
                            &ictx.env.meta,
                        );
                    }
                    if let Some(old) = obj.remove(&from) {
                        obj.insert(to, old);
                    }
                }
                PatchOperation::Copy { from, to } => {
                    let from = stry!(from.eval_to_string(opts, ictx));
                    let to = stry!(to.eval_to_string(opts, ictx));

                    if obj.contains_key(&to) {
                        return error_patch_key_exists(
                            patch_expr,
                            expr,
                            to.to_string(),
                            &ictx.env.meta,
                        );
                    }
                    if let Some(old) = obj.get(&from) {
                        let old = old.clone();
                        obj.insert(to, old);
                    }
                }
                PatchOperation::Merge { ident, expr } => {
                    let new_key = stry!(ident.eval_to_string(opts, ictx));
                    let merge_spec = stry!(expr.run_with_context(opts, ictx));

                    match obj.get_mut(&new_key) {
                        Some(value @ Value::Object(_)) => {
                            stry!(merge_values(patch_expr, expr, value, &merge_spec));
                        }
                        Some(other) => {
                            return error_patch_merge_type_conflict(
                                patch_expr,
                                ident,
                                new_key.to_string(),
                                &other,
                                &ictx.env.meta,
                            );
                        }
                        None => {
                            let mut new_value = Value::from(Object::new());
                            stry!(merge_values(patch_expr, expr, &mut new_value, &merge_spec));
                            obj.insert(new_key, new_value);
                        }
                    }
                }
                PatchOperation::TupleMerge { expr } => {
                    let merge_spec = stry!(expr.run_with_context(opts, ictx));

                    stry!(merge_values(patch_expr, expr, value, &merge_spec));
                }
            }
        } else {
            return error_need_obj(patch_expr, &expr.target, value.value_type(), &ictx.env.meta);
        }
    }
    Ok(())
}

#[inline]
fn test_guard<'run, 'event, 'script, Expr>(
    outer: &'script Expr,
    opts: ExecOpts,
    ictx: InterpreterContext<'run, 'event, 'script>,
    guard: &'script Option<ImutExprInt<'script>>,
) -> Result<bool>
where
    Expr: BaseExpr,

    'script: 'event,
    'event: 'run,
{
    if let Some(guard) = guard {
        let test = stry!(guard.run_with_context(opts, ictx));
        if let Some(b) = test.as_bool() {
            Ok(b)
        } else {
            error_guard_not_bool(outer, guard, &test, &ictx.env.meta)
        }
    } else {
        Ok(true)
    }
}

#[inline]
#[allow(clippy::too_many_lines)]
fn test_predicate_expr<'run, 'event, 'script, Expr>(
    outer: &'script Expr,
    opts: ExecOpts,
    ictx: InterpreterContext<'run, 'event, 'script>,
    target: &'run Value<'event>,
    pattern: &'script Pattern<'script>,
    guard: &'run Option<ImutExprInt<'script>>,
) -> Result<bool>
where
    Expr: BaseExpr,

    'script: 'event,
    'event: 'run,
{
    match pattern {
        Pattern::Record(ref rp) => {
            if match_rp_expr(outer, opts.without_result(), ictx, &target, &rp)?.is_some() {
                test_guard(outer, opts, ictx, guard)
            } else {
                Ok(false)
            }
        }
        Pattern::Array(ref ap) => {
            if match_ap_expr(outer, opts.without_result(), ictx, &target, &ap)?.is_some() {
                test_guard(outer, opts, ictx, guard)
            } else {
                Ok(false)
            }
        }
        Pattern::Expr(ref expr) => {
            let v = stry!(expr.run_with_context(opts, ictx));
            let vb: &Value = v.borrow();
            if val_eq(target, vb) {
                test_guard(outer, opts, ictx, guard)
            } else {
                Ok(false)
            }
        }
        Pattern::Assign(ref a) => {
            match *a.pattern {
                Pattern::Array(ref ap) => {
                    if let Some(v) = match_ap_expr(outer, opts.with_result(), ictx, &target, &ap)? {
                        // we need to assign prior to the guard so we can check
                        // against the pattern expressions
                        set_local_shadow(outer, ictx.local, &ictx.env.meta, a.idx, v)?;
                        test_guard(outer, opts, ictx, guard)
                    } else {
                        Ok(false)
                    }
                }
                Pattern::Record(ref rp) => {
                    if let Some(v) = match_rp_expr(outer, opts.with_result(), ictx, &target, &rp)? {
                        // we need to assign prior to the guard so we can check
                        // against the pattern expressions
                        set_local_shadow(outer, ictx.local, &ictx.env.meta, a.idx, v)?;

                        test_guard(outer, opts, ictx, guard)
                    } else {
                        Ok(false)
                    }
                }
                Pattern::Expr(ref expr) => {
                    let v = stry!(expr.run_with_context(opts, ictx));
                    let vb: &Value = v.borrow();
                    if val_eq(target, vb) {
                        // we need to assign prior to the guard so we can check
                        // against the pattern expressions
                        set_local_shadow(outer, ictx.local, &ictx.env.meta, a.idx, v.into_owned())?;

                        test_guard(outer, opts, ictx, guard)
                    } else {
                        Ok(false)
                    }
                }
                _ => error_oops(outer, "Unimplemented pattern", &ictx.env.meta),
            }
        }
        Pattern::Default => Ok(true),
    }
}

#[inline]
#[allow(clippy::too_many_lines)]
fn match_rp_expr<'run, 'event, 'script, Expr>(
    outer: &'script Expr,
    opts: ExecOpts,
    ictx: InterpreterContext<'run, 'event, 'script>,
    target: &'run Value<'event>,
    rp: &'script RecordPattern,
) -> Result<Option<Value<'event>>>
where
    Expr: BaseExpr,
    'script: 'event,
    'event: 'run,
{
    let mut acc = Value::from(Object::with_capacity(if opts.result_needed {
        rp.fields.len()
    } else {
        0
    }));
    for pp in &rp.fields {
        let known_key = pp.key();

        match pp {
            PredicatePattern::FieldPresent { .. } => {
                if let Some(v) = known_key.lookup(target) {
                    if opts.result_needed {
                        known_key.insert(&mut acc, v.clone())?;
                    }
                    continue;
                } else {
                    return Ok(None);
                }
            }
            PredicatePattern::FieldAbsent { .. } => {
                if known_key.lookup(target).is_some() {
                    return Ok(None);
                } else {
                    continue;
                }
            }
            PredicatePattern::TildeEq { test, .. } => {
                let testee = if let Some(v) = known_key.lookup(target) {
                    v
                } else {
                    return Ok(None);
                };
                if let Ok(x) =
                    test.extractor
                        .extract(opts.result_needed, &testee, &ictx.env.context)
                {
                    if opts.result_needed {
                        known_key.insert(&mut acc, x)?;
                    }
                } else {
                    return Ok(None);
                }
            }
            PredicatePattern::Eq { rhs, not, .. } => {
                let testee = if let Some(v) = known_key.lookup(target) {
                    v
                } else {
                    return Ok(None);
                };

                let rhs = stry!(rhs.run_with_context(opts, ictx));
                let vb: &Value = rhs.borrow();
                let r = val_eq(testee, vb);
                let m = if *not { !r } else { r };

                if m {
                    continue;
                } else {
                    return Ok(None);
                }
            }
            PredicatePattern::RecordPatternEq { pattern, .. } => {
                let testee = if let Some(v) = known_key.lookup(target) {
                    v
                } else {
                    return Ok(None);
                };

                if testee.is_object() {
                    if let Some(m) = match_rp_expr(outer, opts, ictx, testee, pattern)? {
                        if opts.result_needed {
                            known_key.insert(&mut acc, m)?;
                        }
                        continue;
                    } else {
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            }
            PredicatePattern::ArrayPatternEq { pattern, .. } => {
                let testee = if let Some(v) = known_key.lookup(target) {
                    v
                } else {
                    return Ok(None);
                };

                if testee.is_array() {
                    if let Some(r) = match_ap_expr(outer, opts, ictx, testee, pattern)? {
                        if opts.result_needed {
                            known_key.insert(&mut acc, r)?;
                        }

                        continue;
                    } else {
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            }
        }
    }

    Ok(Some(acc))
}

#[inline]
fn match_ap_expr<'run, 'event, 'script, Expr>(
    outer: &'script Expr,
    opts: ExecOpts,
    ictx: InterpreterContext<'run, 'event, 'script>,
    target: &'run Value<'event>,
    ap: &'script ArrayPattern,
) -> Result<Option<Value<'event>>>
where
    Expr: BaseExpr,

    'script: 'event,
    'event: 'run,
{
    if let Some(a) = target.as_array() {
        let mut acc = Vec::with_capacity(if opts.result_needed { a.len() } else { 0 });
        let mut idx: u64 = 0;
        for candidate in a {
            'inner: for expr in &ap.exprs {
                match expr {
                    ArrayPredicatePattern::Expr(e) => {
                        let r = stry!(e.run_with_context(opts, ictx));
                        let vb: &Value = r.borrow();

                        // NOTE: We are creating a new value here so we have to clone
                        if val_eq(candidate, vb) && opts.result_needed {
                            acc.push(Value::Array(vec![Value::from(idx), r.into_owned()]));
                        }
                    }
                    ArrayPredicatePattern::Tilde(test) => {
                        if let Ok(r) = test.extractor.extract(
                            opts.result_needed,
                            &candidate,
                            &ictx.env.context,
                        ) {
                            if opts.result_needed {
                                acc.push(Value::Array(vec![Value::from(idx), r]));
                            }
                        } else {
                            continue 'inner;
                        }
                    }
                    ArrayPredicatePattern::Record(rp) => {
                        if let Some(r) = match_rp_expr(outer, opts, ictx, candidate, rp)? {
                            if opts.result_needed {
                                acc.push(Value::Array(vec![Value::from(idx), r]))
                            };
                        } else {
                            continue 'inner;
                        }
                    }
                }
            }
            idx += 1;
        }
        Ok(Some(Value::Array(acc)))
    } else {
        Ok(None)
    }
}

#[inline]
#[allow(mutable_transmutes, clippy::transmute_ptr_to_ptr)]
fn set_local_shadow<'run, 'event, 'script, Expr>(
    outer: &'script Expr,
    local: &'run LocalStack<'event>,
    node_meta: &'run NodeMetas,
    idx: usize,
    v: Value<'event>,
) -> Result<()>
where
    Expr: BaseExpr,
    'script: 'event,
    'event: 'run,
{
    use std::mem;
    // This is icky do we want it?
    // it is only used
    let local: &mut LocalStack = unsafe { mem::transmute(local) };
    if let Some(d) = local.values.get_mut(idx) {
        *d = Some(v);
        Ok(())
    } else {
        error_oops(outer, "Unknown local variable", &node_meta)
    }
}

//FIXME Do we really want this here?
impl<'script> GroupBy<'script> {
    /// Creates groups based on an event.
    pub fn generate_groups<'run, 'event>(
        &'script self,
        ectx: &'run EventContext,
        event: &'run Value<'event>,
        state: &'run Value<'static>,
        node_meta: &'run NodeMetas<'script>,
        meta: &'run Value<'event>,
    ) -> Result<Vec<Vec<Value<'event>>>>
    where
        'script: 'event,
        'event: 'run,
    {
        let mut groups = Vec::with_capacity(16);
        self.0
            .generate_groups(ectx, event, state, node_meta, meta, &mut groups)?;
        Ok(groups)
    }
}

//FIXME Do we really want this here?
impl<'script> GroupByInt<'script> {
    pub(crate) fn generate_groups<'run, 'event>(
        &'script self,
        ectx: &'run EventContext,
        event: &'run Value<'event>,
        state: &'run Value<'static>,
        node_meta: &'run NodeMetas<'script>,
        meta: &'run Value<'event>,
        groups: &'run mut Vec<Vec<Value<'event>>>,
    ) -> Result<()>
    where
        'script: 'event,
        'event: 'run,
    {
        const NO_AGGRS: [InvokeAggrFn<'static>; 0] = [];
        let opts = ExecOpts {
            result_needed: true,
            aggr: AggrType::Emit,
        };
        // FIXME
        let consts = vec![];
        let local_stack = LocalStack::with_size(0);
        let env = Env {
            consts: &consts,
            context: ectx,
            aggrs: &NO_AGGRS,
            meta: node_meta,
        };

        match self {
            GroupByInt::Expr { expr, .. } => {
                let v = expr
                    .run(opts, &env, event, state, meta, &local_stack)?
                    .into_owned();
                if let Some((last_group, other_groups)) = groups.split_last_mut() {
                    other_groups.iter_mut().for_each(|g| g.push(v.clone()));
                    last_group.push(v)
                } else {
                    // No last group existed, i.e, `groups` was empty. Push a new group:
                    groups.push(vec![v]);
                }
                Ok(())
            }

            GroupByInt::Set { items, .. } => {
                for item in items {
                    item.0
                        .generate_groups(ectx, event, state, node_meta, meta, groups)?
                }

                // set(event.measurement, each(record::keys(event.fields)))
                // GroupBy::Set(items: [GroupBy::Expr(..), GroupBy::Each(GroupBy::Expr(..))])
                // [[7]]
                // [[7, "a"], [7, "b"]]

                // GroupBy::Set(items: [GroupBy::Each(GroupBy::Expr(..)), GroupBy::Expr(..)])
                // [["a"], ["b"]]
                // [["a", 7], ["b", 7]]

                // GroupBy::Set(items: [GroupBy::Each(GroupBy::Expr(..)), GroupBy::Each(GroupBy::Expr(..))])
                // [["a"], ["b"]]
                // [["a", 7], ["b", 7], ["a", 8], ["b", 8]]
                Ok(())
            }
            GroupByInt::Each { expr, .. } => {
                let v = expr
                    .run(opts, &env, event, state, meta, &local_stack)?
                    .into_owned();
                if let Some(each) = v.as_array() {
                    if groups.is_empty() {
                        for e in each {
                            groups.push(vec![e.clone()]);
                        }
                    } else {
                        let mut new_groups = Vec::with_capacity(each.len() * groups.len());
                        for g in groups.drain(..) {
                            for e in each {
                                let mut g = g.clone();
                                g.push(e.clone());
                                new_groups.push(g);
                            }
                        }
                        std::mem::swap(groups, &mut new_groups);
                    }
                    Ok(())
                } else {
                    error_need_arr(self, self, v.value_type(), &env.meta)
                }
            }
        }
    }
}
