// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! What stands in the engine's `model` field on a speech server.
//!
//! A `qwen3tts` model carries its own talker and runs it through
//! [`super::Pipeline`]; there is no language model beside it to serve the
//! text endpoints, which the HTTP layer refuses before they reach this.
//! The engine still holds a [`ModelForward`] for what reads one in passing —
//! `/props`' shape, the slots — so this one describes the talker's shape
//! and runs nothing.

use crate::engine::arch::ModelForward;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::ModelConfig;
use anyhow::{Result, bail};

pub struct Placeholder {
    config: ModelConfig,
}

impl Placeholder {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

const REFUSAL: &str = "this server is serving a speech model (qwen3tts), which generates no text; use \
     /v1/audio/speech";

impl ModelForward for Placeholder {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new(0, capacity, 1)
    }

    fn forward(
        &self,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        bail!(REFUSAL)
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        bail!(REFUSAL)
    }
}
