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

use super::{
    exec_binary, exec_unary, merge_values, patch_value, resolve, set_local_shadow, test_guard,
    test_predicate_expr, AggrType, Env, ExecOpts, InterpreterContext, LocalStack, FALSE, TRUE,
};

use crate::ast::*;
use crate::errors::*;
use crate::registry::{Registry, TremorAggrFnWrapper};
use crate::stry;
use simd_json::prelude::*;
use simd_json::value::borrowed::{Object, Value};
use std::borrow::Borrow;
use std::borrow::Cow;
use std::mem;

impl<'run, 'event, 'script> ImutExpr<'script>
where
    'script: 'event,
    'event: 'run,
{
    /// Evaluates the expression
    pub fn run(
        &'script self,
        opts: ExecOpts,
        env: &'run Env<'run, 'event, 'script>,
        event: &'run Value<'event>,
        state: &'run Value<'static>,
        meta: &'run Value<'event>,
        local: &'run LocalStack<'event>,
    ) -> Result<Cow<'run, Value<'event>>> {
        self.run_with_context(InterpreterContext::of(opts, env, event, state, meta, local))
    }

    /// Evaluates the expression, with the `InterpreterContext` grouped into a struct.
    pub fn run_with_context(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        self.0.run_with_context(ictx)
    }
}
impl<'run, 'event, 'script> ImutExprInt<'script>
where
    'script: 'event,
    'event: 'run,
{
    #[inline]
    pub fn eval_to_string(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
    ) -> Result<Cow<'event, str>> {
        match stry!(self.run_with_context(ictx)).borrow() {
            Value::String(s) => Ok(s.clone()),
            other => error_need_obj(self, self, other.value_type(), &ictx.env.meta),
        }
    }

    #[inline]
    pub fn run(
        &'script self,
        opts: ExecOpts,
        env: &'run Env<'run, 'event, 'script>,
        event: &'run Value<'event>,
        state: &'run Value<'static>,
        meta: &'run Value<'event>,
        local: &'run LocalStack<'event>,
    ) -> Result<Cow<'run, Value<'event>>> {
        self.run_with_context(InterpreterContext::of(opts, env, event, state, meta, local))
    }

    pub fn run_with_context(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        match self {
            ImutExprInt::Literal(literal) => Ok(Cow::Borrowed(&literal.value)),
            ImutExprInt::Path(path) => resolve(self, ictx, path),
            ImutExprInt::Present { path, .. } => self.present(ictx, path),
            ImutExprInt::Record(ref record) => {
                let mut object: Object = Object::with_capacity(record.fields.len());

                for field in &record.fields {
                    let result = stry!(field.value.run_with_context(ictx));
                    let name = stry!(field.name.eval_to_string(ictx));
                    object.insert(name, result.into_owned());
                }

                Ok(Cow::Owned(Value::from(object)))
            }
            ImutExprInt::List(ref list) => {
                let mut r: Vec<Value<'event>> = Vec::with_capacity(list.exprs.len());
                for expr in &list.exprs {
                    r.push(stry!(expr.run_with_context(ictx)).into_owned());
                }
                Ok(Cow::Owned(Value::Array(r)))
            }
            ImutExprInt::Invoke1(ref call) => self.invoke1(ictx, call),
            ImutExprInt::Invoke2(ref call) => self.invoke2(ictx, call),
            ImutExprInt::Invoke3(ref call) => self.invoke3(ictx, call),
            ImutExprInt::Invoke(ref call) => self.invoke(ictx, call),
            ImutExprInt::InvokeAggr(ref call) => self.emit_aggr(ictx.opts, ictx.env, call),
            ImutExprInt::Patch(ref expr) => self.patch(ictx, expr),
            ImutExprInt::Merge(ref expr) => self.merge(ictx, expr),
            ImutExprInt::Local {
                idx,
                mid,
                is_const: false,
            } => match ictx.local.values.get(*idx) {
                Some(Some(l)) => Ok(Cow::Borrowed(l)),
                Some(None) => {
                    let path: Path = Path::Local(LocalPath {
                        is_const: false,
                        idx: *idx,
                        mid: *mid,
                        segments: vec![],
                    });
                    //TODO: get root key
                    error_bad_key(
                        self,
                        self,
                        &path,
                        ictx.env.meta.name_dflt(*mid).to_string(),
                        vec![],
                        &ictx.env.meta,
                    )
                }

                _ => error_oops(self, "Unknown local variable", &ictx.env.meta),
            },
            ImutExprInt::Local {
                idx,
                is_const: true,
                ..
            } => match ictx.env.consts.get(*idx) {
                Some(v) => Ok(Cow::Borrowed(v)),
                _ => error_oops(self, "Unknown const variable", &ictx.env.meta),
            },
            ImutExprInt::Unary(ref expr) => self.unary(ictx, expr),
            ImutExprInt::Binary(ref expr) => self.binary(ictx, expr),
            ImutExprInt::Match(ref expr) => self.match_expr(ictx, expr),
            ImutExprInt::Comprehension(ref expr) => self.comprehension(ictx, expr),
        }
    }

    fn comprehension(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script ImutComprehension,
    ) -> Result<Cow<'run, Value<'event>>> {
        let mut value_vec = vec![];
        let target = &expr.target;
        let cases = &expr.cases;
        let target_value = stry!(target.run_with_context(ictx));

        if let Some(target_map) = target_value.as_object() {
            // Record comprehension case
            value_vec.reserve(target_map.len());
            // NOTE: Since we we are going to create new data from this
            // object we are cloning it.
            // This is also required since we might mutate. If we restruct
            // mutation in the future we could get rid of this.

            'comprehension_outer: for (k, v) in target_map.clone() {
                set_local_shadow(
                    self,
                    ictx.local,
                    &ictx.env.meta,
                    expr.key_id,
                    Value::String(k),
                )?;
                set_local_shadow(self, ictx.local, &ictx.env.meta, expr.val_id, v)?;
                for e in cases {
                    if stry!(test_guard(self, ictx, &e.guard)) {
                        let v = stry!(self.execute_effectors(ictx, e, &e.exprs));
                        // NOTE: We are creating a new value so we have to clone;
                        value_vec.push(v.into_owned());
                        continue 'comprehension_outer;
                    }
                }
            }
        } else if let Some(target_array) = target_value.as_array() {
            // Array comprehension case

            value_vec.reserve(target_array.len());

            // NOTE: Since we we are going to create new data from this
            // object we are cloning it.
            // This is also required since we might mutate. If we restruct
            // mutation in the future we could get rid of this.

            let mut count = 0;
            'comp_array_outer: for x in target_array.clone() {
                set_local_shadow(self, ictx.local, &ictx.env.meta, expr.key_id, count.into())?;
                set_local_shadow(self, ictx.local, &ictx.env.meta, expr.val_id, x)?;

                for e in cases {
                    if stry!(test_guard(self, ictx, &e.guard)) {
                        let v = stry!(self.execute_effectors(ictx, e, &e.exprs));

                        value_vec.push(v.into_owned());
                        count += 1;
                        continue 'comp_array_outer;
                    }
                }
                count += 1;
            }
        }
        Ok(Cow::Owned(Value::Array(value_vec)))
    }

    #[inline]
    fn execute_effectors<T: BaseExpr>(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        inner: &'script T,
        effectors: &'script [ImutExpr<'script>],
    ) -> Result<Cow<'run, Value<'event>>> {
        // Since we don't have side effects we don't need to run anything but the last effector!
        if let Some(effector) = effectors.last() {
            effector.run_with_context(ictx)
        } else {
            error_missing_effector(self, inner, &ictx.env.meta)
        }
    }

    fn match_expr(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script ImutMatch,
    ) -> Result<Cow<'run, Value<'event>>> {
        let target = stry!(expr.target.run_with_context(ictx));

        for predicate in &expr.patterns {
            if stry!(test_predicate_expr(
                self,
                ictx.clone(),
                &target,
                &predicate.pattern,
                &predicate.guard,
            )) {
                return self.execute_effectors(ictx.clone(), predicate, &predicate.exprs);
            }
        }
        error_no_clause_hit(self, &ictx.env.meta)
    }

    fn binary(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script BinExpr<'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        let lhs = stry!(expr.lhs.run_with_context(ictx));
        let rhs = stry!(expr.rhs.run_with_context(ictx));
        exec_binary(self, expr, &ictx.env.meta, expr.kind, &lhs, &rhs)
    }

    fn unary(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script UnaryExpr<'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        let rhs = stry!(expr.expr.run_with_context(ictx));
        // TODO align this implemenation to be similar to exec_binary?
        match exec_unary(expr.kind, &rhs) {
            Some(v) => Ok(v),
            None => error_invalid_unary(self, &expr.expr, expr.kind, &rhs, &ictx.env.meta),
        }
    }

    #[allow(clippy::too_many_lines)]
    // FIXME: Quite some overlap with `interpreter::resolve` and `assign`
    fn present(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        path: &'script Path,
    ) -> Result<Cow<'run, Value<'event>>> {
        let mut subrange: Option<(usize, usize)> = None;
        let mut current: &Value = match path {
            Path::Local(path) => match ictx.local.values.get(path.idx) {
                Some(Some(l)) => l,
                Some(None) => return Ok(Cow::Borrowed(&FALSE)),
                _ => return error_oops(self, "Unknown local variable", &ictx.env.meta),
            },
            Path::Const(path) => match ictx.env.consts.get(path.idx) {
                Some(v) => v,
                _ => return error_oops(self, "Unknown constant variable", &ictx.env.meta),
            },
            Path::Meta(_path) => ictx.meta,
            Path::Event(_path) => ictx.event,
            Path::State(_path) => ictx.state,
        };

        for segment in path.segments() {
            match segment {
                Segment::Id { key, .. } => {
                    if let Some(c) = key.lookup(current) {
                        current = c;
                        subrange = None;
                        continue;
                    } else {
                        return Ok(Cow::Borrowed(&FALSE));
                    }
                }
                Segment::Idx { idx, .. } => {
                    if let Some(a) = current.as_array() {
                        let (start, end) = if let Some((start, end)) = subrange {
                            // We check range on setting the subrange!
                            (start, end)
                        } else {
                            (0, a.len())
                        };
                        let idx = *idx as usize + start;
                        if idx >= end {
                            // We exceed the sub range
                            return Ok(Cow::Borrowed(&FALSE));
                        }
                        if let Some(c) = a.get(idx) {
                            current = c;
                            subrange = None;
                            continue;
                        } else {
                            return Ok(Cow::Borrowed(&FALSE));
                        }
                    } else {
                        return Ok(Cow::Borrowed(&FALSE));
                    }
                }

                Segment::Element { expr, .. } => {
                    let next = match (current, stry!(expr.run_with_context(ictx)).borrow()) {
                        (Value::Array(a), idx) => {
                            if let Some(idx) = idx.as_usize() {
                                let (start, end) = if let Some((start, end)) = subrange {
                                    // We check range on setting the subrange!
                                    (start, end)
                                } else {
                                    (0, a.len())
                                };
                                let idx = idx + start;
                                if idx >= end {
                                    // We exceed the sub range
                                    return Ok(Cow::Borrowed(&FALSE));
                                }
                                a.get(idx)
                            } else {
                                return Ok(Cow::Borrowed(&FALSE));
                            }
                        }
                        (Value::Object(o), Value::String(id)) => o.get(id),
                        _other => return Ok(Cow::Borrowed(&FALSE)),
                    };
                    if let Some(next) = next {
                        current = next;
                        subrange = None;
                        continue;
                    } else {
                        return Ok(Cow::Borrowed(&FALSE));
                    }
                }
                Segment::Range {
                    range_start,
                    range_end,
                    ..
                } => {
                    if let Some(a) = current.as_array() {
                        let (start, end) = if let Some((start, end)) = subrange {
                            // We check range on setting the subrange!
                            (start, end)
                        } else {
                            (0, a.len())
                        };

                        let s = stry!(range_start.run_with_context(ictx));

                        if let Some(range_start) = s.as_usize() {
                            let range_start = range_start + start;

                            let e = stry!(range_end.run_with_context(ictx));

                            if let Some(range_end) = e.as_usize() {
                                let range_end = range_end + start;
                                // We're exceeding the array
                                if range_end >= end {
                                    return Ok(Cow::Borrowed(&FALSE));
                                } else {
                                    subrange = Some((range_start, range_end));
                                    continue;
                                }
                            }
                        }
                        return Ok(Cow::Borrowed(&FALSE));
                    } else {
                        return Ok(Cow::Borrowed(&FALSE));
                    }
                }
            }
        }

        Ok(Cow::Borrowed(&TRUE))
    }

    // TODO: Can we convince Rust to generate the 3 or 4 versions of this method from one template?
    fn invoke1(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        unsafe {
            let v = stry!(expr.args.get_unchecked(0).run_with_context(ictx));
            expr.invocable
                .invoke(&ictx.env.context, &[v.borrow()])
                .map(Cow::Owned)
                .map_err(|e| {
                    let r: Option<&Registry> = None;
                    e.into_err(self, self, r, &ictx.env.meta)
                })
        }
    }

    fn invoke2(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        unsafe {
            let v1 = stry!(expr.args.get_unchecked(0).run_with_context(ictx));
            let v2 = stry!(expr.args.get_unchecked(1).run_with_context(ictx));
            expr.invocable
                .invoke(&ictx.env.context, &[v1.borrow(), v2.borrow()])
                .map(Cow::Owned)
                .map_err(|e| {
                    let r: Option<&Registry> = None;
                    e.into_err(self, self, r, &ictx.env.meta)
                })
        }
    }

    fn invoke3(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        unsafe {
            let v1 = stry!(expr.args.get_unchecked(0).run_with_context(ictx));
            let v2 = stry!(expr.args.get_unchecked(1).run_with_context(ictx));
            let v3 = stry!(expr.args.get_unchecked(2).run_with_context(ictx));
            expr.invocable
                .invoke(&ictx.env.context, &[v1.borrow(), v2.borrow(), v3.borrow()])
                .map(Cow::Owned)
                .map_err(|e| {
                    let r: Option<&Registry> = None;
                    e.into_err(self, self, r, &ictx.env.meta)
                })
        }
    }

    fn invoke(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        let argv: Vec<Cow<'run, _>> = expr
            .args
            .iter()
            .map(|arg| arg.run_with_context(ictx))
            .collect::<Result<_>>()?;

        // Construct a view into `argv`, since `invoke` expects a slice of references, not Cows.
        let argv1: Vec<&Value> = argv.iter().map(Cow::borrow).collect();

        expr.invocable
            .invoke(&ictx.env.context, &argv1)
            .map(Cow::Owned)
            .map_err(|e| {
                let r: Option<&Registry> = None;
                e.into_err(self, self, r, &ictx.env.meta)
            })
    }

    #[allow(mutable_transmutes, clippy::transmute_ptr_to_ptr)]
    fn emit_aggr(
        &'script self,
        opts: ExecOpts,
        env: &'run Env<'run, 'event, 'script>,
        expr: &'script InvokeAggr,
    ) -> Result<Cow<'run, Value<'event>>> {
        if opts.aggr != AggrType::Emit {
            return error_oops(
                self,
                "Trying to emit aggreagate outside of emit context",
                &env.meta,
            );
        }

        unsafe {
            // FIXME?
            let invocable: &mut TremorAggrFnWrapper =
                mem::transmute(&env.aggrs[expr.aggr_id].invocable);
            let r = invocable.emit().map(Cow::Owned).map_err(|e| {
                let r: Option<&Registry> = None;
                e.into_err(self, self, r, &env.meta)
            })?;
            Ok(r)
        }
    }

    fn patch(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script Patch,
    ) -> Result<Cow<'run, Value<'event>>> {
        // NOTE: We clone this since we patch it - this should be not mutated but cloned

        let mut value = stry!(expr.target.run_with_context(ictx)).into_owned();
        stry!(patch_value(self, ictx, &mut value, expr));
        Ok(Cow::Owned(value))
    }

    fn merge(
        &'script self,
        ictx: InterpreterContext<'run, 'event, 'script>,
        expr: &'script Merge,
    ) -> Result<Cow<'run, Value<'event>>> {
        // NOTE: We got to clone here since we're going to change the value
        let value = stry!(expr.target.run_with_context(ictx));

        if value.is_object() {
            // Make sure we clone the data so we don't mutate it in place
            let mut value = value.into_owned();
            let replacement = stry!(expr.expr.run_with_context(ictx));

            if replacement.is_object() {
                stry!(merge_values(self, &expr.expr, &mut value, &replacement));
                Ok(Cow::Owned(value))
            } else {
                error_need_obj(self, &expr.expr, replacement.value_type(), &ictx.env.meta)
            }
        } else {
            error_need_obj(self, &expr.target, value.value_type(), &ictx.env.meta)
        }
    }
}
