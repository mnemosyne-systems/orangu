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

pub mod arch;
pub mod attention;
pub mod backend;
pub mod chat_template;
pub mod constraint;
pub mod decode_stages;
pub mod dense_residency;
pub mod env;
pub mod expert_read;
pub mod expert_store;
pub mod expert_tier;
pub mod footprint;
pub mod generate;
pub mod iq_grids;
pub mod kv_cache;
pub mod kv_pool;
pub mod loader;
pub mod metrics;
pub mod moe_stats;
pub mod page_cache;
pub mod placement;
pub mod plan;
pub mod prefix_cache;
pub mod prefix_index;
pub mod quant;
pub mod route_ahead;
pub mod sampling;
pub mod scheduler;
pub mod slot_store;
pub mod tensor;
pub mod tokenizer;
pub mod tool_calls;
pub mod vecdot;
