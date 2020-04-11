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
        self.run_with_context(
            opts,
            &InterpreterContext::of(env, event, state, meta, local),
        )
    }

    /// Evaluates the expression, with the `InterpreterContext` grouped into a struct.
    pub fn run_with_context(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        self.0.run_with_context(opts, ictx)
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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
    ) -> Result<Cow<'event, str>> {
        match stry!(self.run_with_context(opts, ictx)).borrow() {
            Value::String(s) => Ok(s.clone()),
            other => error_need_obj(self, self, other.value_type(), &ictx.env.meta),
        }
    }

    #[inline]
    pub fn eval_to_index<Expr>(
        &'script self,
        outer: &'script Expr,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        path: &'script Path,
        range: &[Value],
    ) -> Result<usize>
    where
        Expr: BaseExpr,
    {
        let val = stry!(self.run_with_context(opts, ictx));
        if let Some(n) = val.as_usize() {
            Ok(n)
        } else if val.is_i64() {
            // TODO: As soon as value-trait v0.1.8 is used, switch this `is_i64` to `is_integer`.
            error_bad_array_index(outer, self, path, val.borrow(), range.len(), &ictx.env.meta)
        } else {
            error_need_int(outer, self, val.value_type(), &ictx.env.meta)
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
        self.run_with_context(
            opts,
            &InterpreterContext::of(env, event, state, meta, local),
        )
    }

    pub fn run_with_context(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        match self {
            ImutExprInt::Literal(literal) => Ok(Cow::Borrowed(&literal.value)),
            ImutExprInt::Path(path) => resolve(self, opts, ictx, path),
            ImutExprInt::Present { path, .. } => self.present(opts, ictx, path),
            ImutExprInt::Record(ref record) => {
                let mut object: Object = Object::with_capacity(record.fields.len());

                for field in &record.fields {
                    let result = stry!(field.value.run_with_context(opts, ictx));
                    let name = stry!(field.name.eval_to_string(opts, ictx));
                    object.insert(name, result.into_owned());
                }

                Ok(Cow::Owned(Value::from(object)))
            }
            ImutExprInt::List(ref list) => {
                let mut r: Vec<Value<'event>> = Vec::with_capacity(list.exprs.len());
                for expr in &list.exprs {
                    r.push(stry!(expr.run_with_context(opts, ictx)).into_owned());
                }
                Ok(Cow::Owned(Value::Array(r)))
            }
            ImutExprInt::Invoke1(ref call) => self.invoke1(opts, ictx, call),
            ImutExprInt::Invoke2(ref call) => self.invoke2(opts, ictx, call),
            ImutExprInt::Invoke3(ref call) => self.invoke3(opts, ictx, call),
            ImutExprInt::Invoke(ref call) => self.invoke(opts, ictx, call),
            ImutExprInt::InvokeAggr(ref call) => self.emit_aggr(opts, ictx.env, call),
            ImutExprInt::Patch(ref expr) => self.patch(opts, ictx, expr),
            ImutExprInt::Merge(ref expr) => self.merge(opts, ictx, expr),
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
            ImutExprInt::Unary(ref expr) => self.unary(opts, ictx, expr),
            ImutExprInt::Binary(ref expr) => self.binary(opts, ictx, expr),
            ImutExprInt::Match(ref expr) => self.match_expr(opts, ictx, expr),
            ImutExprInt::Comprehension(ref expr) => self.comprehension(opts, ictx, expr),
        }
    }

    fn comprehension(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script ImutComprehension,
    ) -> Result<Cow<'run, Value<'event>>> {
        let mut value_vec = vec![];
        let target = &expr.target;
        let cases = &expr.cases;
        let target_value = stry!(target.run_with_context(opts, ictx));

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
                    if stry!(test_guard(self, opts, ictx, &e.guard)) {
                        let v = stry!(self.execute_effectors(opts, ictx, e, &e.exprs));
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
                    if stry!(test_guard(self, opts, ictx, &e.guard)) {
                        let v = stry!(self.execute_effectors(opts, ictx, e, &e.exprs));

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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        inner: &'script T,
        effectors: &'script [ImutExpr<'script>],
    ) -> Result<Cow<'run, Value<'event>>> {
        // Since we don't have side effects we don't need to run anything but the last effector!
        if let Some(effector) = effectors.last() {
            effector.run_with_context(opts, ictx)
        } else {
            error_missing_effector(self, inner, &ictx.env.meta)
        }
    }

    fn match_expr(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script ImutMatch,
    ) -> Result<Cow<'run, Value<'event>>> {
        let target = stry!(expr.target.run_with_context(opts, ictx));

        for predicate in &expr.patterns {
            if stry!(test_predicate_expr(
                self,
                opts,
                ictx,
                &target,
                &predicate.pattern,
                &predicate.guard,
            )) {
                return self.execute_effectors(opts, ictx, predicate, &predicate.exprs);
            }
        }
        error_no_clause_hit(self, &ictx.env.meta)
    }

    fn binary(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script BinExpr<'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        let lhs = stry!(expr.lhs.run_with_context(opts, ictx));
        let rhs = stry!(expr.rhs.run_with_context(opts, ictx));
        exec_binary(self, expr, &ictx.env.meta, expr.kind, &lhs, &rhs)
    }

    fn unary(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script UnaryExpr<'script>,
    ) -> Result<Cow<'run, Value<'event>>> {
        let rhs = stry!(expr.expr.run_with_context(opts, ictx));
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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        path: &'script Path,
    ) -> Result<Cow<'run, Value<'event>>> {
        // Fetch the base of the path
        // TODO: Extract this into a method on `Path`?
        let base_value: &Value = match path {
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

        // Resolve the targeted value by applying all path segments
        let mut subrange: Option<&[Value]> = None;
        let mut current = base_value;
        for segment in path.segments() {
            match segment {
                // Next segment is an identifier: lookup the identifier on `current`, if it's an object
                Segment::Id { key, .. } => {
                    if let Some(c) = key.lookup(current) {
                        current = c;
                        subrange = None;
                        continue;
                    } else {
                        // No field for that id: not present
                        return Ok(Cow::Borrowed(&FALSE));
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
                            // No element at the index: not present
                            return Ok(Cow::Borrowed(&FALSE));
                        }
                    } else {
                        // FIXME: Indexing into something that isn't an array: should this return
                        // false, or return an error?
                        return Ok(Cow::Borrowed(&FALSE));
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

                        let range_start = range_start.eval_to_index(
                            self,
                            opts,
                            ictx,
                            path,
                            &range_to_consider,
                        )?;
                        let range_end =
                            range_end.eval_to_index(self, opts, ictx, path, &range_to_consider)?;
                        if range_end < range_start {
                            return error_decreasing_range(
                                self,
                                segment,
                                &path,
                                range_start,
                                range_end,
                                &ictx.env.meta,
                            );
                        } else if range_end > range_to_consider.len() {
                            // Index is out of array bounds: not present
                            return Ok(Cow::Borrowed(&FALSE));
                        } else {
                            subrange = Some(&range_to_consider[range_start..range_end]);
                            continue;
                        }
                    } else {
                        // FIXME: Indexing into something that isn't an array: should this return
                        // false, or return an error? (Should probably have the same answer as the
                        // same question a couple of lines higher in this function.)
                        return Ok(Cow::Borrowed(&FALSE));
                    }
                }
                // Next segment is an expression: run `expr` to know which key it signifies at runtime
                Segment::Element { expr, .. } => {
                    let key = stry!(expr.run_with_context(opts, ictx));

                    let next = match (current, key.borrow()) {
                        // The segment resolved to an identifier, and `current` is an object: lookup
                        (Value::Object(o), Value::String(id)) => o.get(id),
                        // If `current` is an array, the segment has to be an index
                        (Value::Array(a), idx) => {
                            // FIXME: Same logic as in `eval_to_index` to handle out-of-usize-range
                            // (e.g., negative numbers) indices.

                            if let Some(idx) = idx.as_usize() {
                                let range_to_consider = subrange.unwrap_or_else(|| a.as_slice());
                                range_to_consider.get(idx)
                            } else {
                                // Index is out of array bounds: not present
                                return Ok(Cow::Borrowed(&FALSE));
                            }
                        }
                        // Anything else: not present
                        // TODO: Double-check this reasoning, are there some cases where we'd
                        // want to return an error anyway?
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
            }
        }

        Ok(Cow::Borrowed(&TRUE))
    }

    // TODO: Can we convince Rust to generate the 3 or 4 versions of this method from one template?
    fn invoke1(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        unsafe {
            let v = stry!(expr.args.get_unchecked(0).run_with_context(opts, ictx));
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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        unsafe {
            let v1 = stry!(expr.args.get_unchecked(0).run_with_context(opts, ictx));
            let v2 = stry!(expr.args.get_unchecked(1).run_with_context(opts, ictx));
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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        unsafe {
            let v1 = stry!(expr.args.get_unchecked(0).run_with_context(opts, ictx));
            let v2 = stry!(expr.args.get_unchecked(1).run_with_context(opts, ictx));
            let v3 = stry!(expr.args.get_unchecked(2).run_with_context(opts, ictx));
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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script Invoke,
    ) -> Result<Cow<'run, Value<'event>>> {
        let argv: Vec<Cow<'run, _>> = expr
            .args
            .iter()
            .map(|arg| arg.run_with_context(opts, ictx))
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
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script Patch,
    ) -> Result<Cow<'run, Value<'event>>> {
        // NOTE: We clone this since we patch it - this should be not mutated but cloned

        let mut value = stry!(expr.target.run_with_context(opts, ictx)).into_owned();
        stry!(patch_value(self, opts, ictx, &mut value, expr));
        Ok(Cow::Owned(value))
    }

    fn merge(
        &'script self,
        opts: ExecOpts,
        ictx: &InterpreterContext<'run, 'event, 'script>,
        expr: &'script Merge,
    ) -> Result<Cow<'run, Value<'event>>> {
        // NOTE: We got to clone here since we're going to change the value
        let value = stry!(expr.target.run_with_context(opts, ictx));

        if value.is_object() {
            // Make sure we clone the data so we don't mutate it in place
            let mut value = value.into_owned();
            let replacement = stry!(expr.expr.run_with_context(opts, ictx));

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
