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

//! The parts of a `[workers]` channel every kind of node speaks: the
//! frames, proving the secret, the connection, and finding the other nodes.
//! `orangu-server` builds its tree of layers on them; `orangu-gguf` joins
//! the same channel as a node of its own kind.
//!
//! - [`protocol`]: the frames two nodes exchange.
//! - [`auth`]: proving `[workers].secret` without sending it.
//! - [`transport`]: plain TCP or TLS between nodes.
//! - [`channel`]: finding the other nodes of `[workers].name`.

pub mod auth;
pub mod channel;
pub mod protocol;
pub mod transport;
