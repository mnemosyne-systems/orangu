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

use crate::*;

mod auto;
mod interactive;

pub(crate) use auto::*;
pub(crate) use interactive::*;

/// Copy `text` to the system clipboard.
pub(crate) fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new().context("failed to access the clipboard")?;
    clipboard
        .set_text(text.to_string())
        .context("failed to write to the clipboard")?;
    Ok(())
}

/// The human-readable status label for a file in the exit summary.
pub(crate) fn review_status_label(status: ReviewStatus) -> &'static str {
    match status {
        ReviewStatus::Approved => "Approved",
        ReviewStatus::Rejected => "Rejected",
        ReviewStatus::Unreviewed => "No review",
    }
}
