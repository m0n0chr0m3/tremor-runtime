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

//! # Exit offramp terminates runtime
//!
//! The exit offramp terminates the runtime with a system exit status
//!
//! ## Configuration
//!
//! This operator takes no configuration

use crate::offramp::prelude::*;
use halfbrown::HashMap;
use simd_json::StaticNode;
use tremor_script::prelude::*;

pub struct Exit {
    pipelines: HashMap<TremorURL, pipeline::Addr>,
    postprocessors: Postprocessors,
}

impl offramp::Impl for Exit {
    fn from_config(_config: &Option<OpConfig>) -> Result<Box<dyn Offramp>> {
        Ok(Box::new(Self {
            pipelines: HashMap::new(),
            postprocessors: vec![],
        }))
    }
}

impl Offramp for Exit {
    fn on_event(&mut self, _codec: &Box<dyn Codec>, _input: String, event: Event) -> Result<()> {
        for (value, _meta) in event.value_meta_iter() {
            if let Some(Value::Static(StaticNode::I64(status))) = value.get("exit") {
                #[allow(clippy::cast_possible_truncation)]
                // ALLOW: this is the supposed to exit
                std::process::exit(*status as i32);
            } else {
                return Err("Unexpected event received in exit offramp".into());
            }
        }
        Ok(())
    }
    fn add_pipeline(&mut self, id: TremorURL, addr: pipeline::Addr) {
        self.pipelines.insert(id, addr);
    }
    fn remove_pipeline(&mut self, id: TremorURL) -> bool {
        self.pipelines.remove(&id);
        self.pipelines.is_empty()
    }
    fn default_codec(&self) -> &str {
        "json"
    }
    fn start(&mut self, _codec: &Box<dyn Codec>, postprocessors: &[String]) -> Result<()> {
        self.postprocessors = make_postprocessors(postprocessors)?;
        Ok(())
    }
}
