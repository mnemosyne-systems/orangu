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

//! The status line's `PLAN` badge: whether the active tab is in plan mode.

use std::sync::atomic::{AtomicBool, Ordering};

static SHOWN: AtomicBool = AtomicBool::new(false);

/// Shows the badge when `on`, hides it otherwise.
pub fn show(on: bool) {
    SHOWN.store(on, Ordering::Relaxed);
}

/// Whether the status line shows the badge.
pub fn shown() -> bool {
    SHOWN.load(Ordering::Relaxed)
}
