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

//! Capturing a CPU profile and rendering it as a flamegraph.
//!
//! This lives in the library rather than in one tool because two of them
//! need it and the alternative was a copy. `orangu-bench` profiles a
//! *server* over the window it measured the rate of; `orangu-gguf`
//! profiles *itself* over the training steps it ran. The mechanism is the
//! same either way — bracket a known window, sample a pid, collapse and
//! render in process — and it was worth exactly one implementation.
//!
//! [`profile::Recorder`] is that mechanism; [`flamegraph`] is everything
//! downstream of `perf script`, which is to say everything that would
//! otherwise be a shell pipeline and a Perl script.

pub mod flamegraph;
pub mod profile;
