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

//! Spreading one model's layers over a tree of `orangu-server` nodes —
//! the `[workers]` section.
//!
//! - [`protocol`]: the frames a parent and a worker exchange.
//! - [`auth`]: proving `[workers].secret` without sending it.
//! - [`identity`]: making sure every node has the same model, quantization
//!   included.
//! - [`pipeline`]: one node's slice — its own layers, then its workers' —
//!   and the top-level node's [`pipeline::DelegatingModel`].
//! - [`stage`]: the parent side of a forward; [`link`]: its TCP connection.
//! - [`session`]: the worker side of a forward.
//! - [`plan`]: dividing a range of layers by memory.
//! - [`node`]: one process's place in the tree, both halves.
//! - [`clock`]: keeping a node's cores at full clock while a request runs.
//! - [`transport`]: plain TCP or TLS between nodes.

pub mod auth;
pub mod clock;
pub mod fetch;
pub mod guard;
pub mod identity;
pub mod link;
pub mod metrics;
pub mod node;
pub mod pipeline;
pub mod plan;
pub mod protocol;
pub mod room;
pub mod session;
pub mod speed;
pub mod stage;
pub mod transport;
