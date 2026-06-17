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

//! Workspace tabs.
//!
//! A *workspace* is a directory orangu is open on. The [`WorkspaceManager`]
//! holds the ordered list of open workspaces and tracks which one is active, so
//! several projects can be kept open in a single orangu instance instead of
//! running one instance per project.
//!
//! This module is only the model and the bookkeeping — the list, the active
//! tab, and the open/close/switch operations over them. Giving each tab its own
//! session, conversation and pending queue, the status line, the key bindings
//! and the `/workspace` command are wired up on top of this in later steps.
//!
//! Two rules from the design shape the operations here:
//!
//! * There is always at least one workspace. Closing the last one is rejected —
//!   only `/quit` ends orangu.
//! * Closing a tab renumbers the ones after it: tab numbers are just positions
//!   in the list (1-based for the user), so removing one shifts the rest down.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Where the workspace tab bar is drawn, set by `workspaces` in the `[orangu]`
/// section of `orangu.conf`. Parsing is case-insensitive; an unset value
/// defaults to [`Top`](Self::Top). The status-line renderer reads this to place
/// the tabs; the model itself does not care where they appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum WorkspacePlacement {
    /// Tabs across the top, above the transcript. The default.
    #[default]
    Top,
    /// Tabs across the bottom, under the status bar.
    Bottom,
    /// Tabs down the left edge.
    Left,
    /// Tabs down the right edge.
    Right,
}

impl FromStr for WorkspacePlacement {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "top" => Ok(Self::Top),
            "bottom" => Ok(Self::Bottom),
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            other => Err(format!(
                "expected one of top|bottom|left|right, found '{other}'"
            )),
        }
    }
}

/// A single workspace tab.
///
/// Identified by its directory. The per-tab runtime state (its session,
/// conversation, pending queue and optional server/model override) will hang
/// off this type as the feature is built out; for now a workspace is just the
/// directory it is open on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    path: PathBuf,
}

impl Workspace {
    /// Create a workspace open on `path`.
    ///
    /// The path is taken as-is; callers are expected to pass an absolute,
    /// normalized directory (the binary resolves it the same way it resolves
    /// the `--workspace` argument).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The directory this workspace is open on.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The ordered set of open workspace tabs and which one is active.
///
/// Invariant: `workspaces` is never empty and `active` is always a valid index
/// into it. Every method preserves both.
#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    workspaces: Vec<Workspace>,
    active: usize,
}

impl WorkspaceManager {
    /// Start with a single workspace, which becomes the active tab (tab 1).
    pub fn new(initial: Workspace) -> Self {
        Self {
            workspaces: vec![initial],
            active: 0,
        }
    }

    /// Number of open workspaces.
    pub fn len(&self) -> usize {
        self.workspaces.len()
    }

    /// Always `false`: there is always at least one workspace. Present so the
    /// type pairs `len` with `is_empty`, as Rust convention expects.
    pub fn is_empty(&self) -> bool {
        self.workspaces.is_empty()
    }

    /// Index of the active workspace (0-based). The user-facing tab number is
    /// this plus one.
    pub fn active_index(&self) -> usize {
        self.active
    }

    /// The active workspace.
    pub fn active(&self) -> &Workspace {
        &self.workspaces[self.active]
    }

    /// The active workspace, mutably.
    pub fn active_mut(&mut self) -> &mut Workspace {
        &mut self.workspaces[self.active]
    }

    /// All open workspaces, in tab order.
    pub fn workspaces(&self) -> &[Workspace] {
        &self.workspaces
    }

    /// The workspace at `index`, if any.
    pub fn get(&self, index: usize) -> Option<&Workspace> {
        self.workspaces.get(index)
    }

    /// The index of the tab open on `path`, if one is.
    pub fn position_of(&self, path: &Path) -> Option<usize> {
        self.workspaces.iter().position(|w| w.path() == path)
    }

    /// Open `workspace` as a new tab to the right of the others and make it
    /// active. Returns the new tab's index.
    ///
    /// This does not deduplicate; opening the same directory twice yields two
    /// tabs. Use [`open_or_switch`](Self::open_or_switch) for the
    /// `/workspace <path>` behaviour, where an already-open directory is
    /// switched to rather than opened again.
    pub fn open(&mut self, workspace: Workspace) -> usize {
        self.workspaces.push(workspace);
        self.active = self.workspaces.len() - 1;
        self.active
    }

    /// Switch to the tab open on `workspace`'s path if there already is one,
    /// otherwise open a new tab for it. Either way the matching tab ends up
    /// active. Returns its index.
    pub fn open_or_switch(&mut self, workspace: Workspace) -> usize {
        match self.position_of(workspace.path()) {
            Some(index) => {
                self.active = index;
                index
            }
            None => self.open(workspace),
        }
    }

    /// Make the tab at `index` active. Returns `false` (and changes nothing) if
    /// `index` is out of range.
    pub fn switch_to(&mut self, index: usize) -> bool {
        if index >= self.workspaces.len() {
            return false;
        }
        self.active = index;
        true
    }

    /// Move focus to the next tab on the right, wrapping from the last tab back
    /// to the first.
    pub fn focus_next(&mut self) {
        self.active = (self.active + 1) % self.workspaces.len();
    }

    /// Move focus to the previous tab on the left, wrapping from the first tab
    /// round to the last.
    pub fn focus_previous(&mut self) {
        let len = self.workspaces.len();
        self.active = (self.active + len - 1) % len;
    }

    /// Close the tab at `index`.
    ///
    /// Returns `false` (and changes nothing) when `index` is out of range or it
    /// is the only open tab — the last workspace is never closed, only `/quit`
    /// ends orangu.
    ///
    /// The tabs after the closed one shift down by one (their numbers drop).
    /// The active tab follows sensibly: closing a tab to the left of the active
    /// one keeps the same tab active under its new number; closing the active
    /// tab moves focus to its right neighbour, or to the new last tab when the
    /// active tab was the rightmost.
    pub fn close(&mut self, index: usize) -> bool {
        if index >= self.workspaces.len() || self.workspaces.len() == 1 {
            return false;
        }
        self.workspaces.remove(index);
        if self.active > index || self.active >= self.workspaces.len() {
            self.active -= 1;
        }
        true
    }

    /// Close the active tab. See [`close`](Self::close); returns `false` when
    /// the active tab is the only one open.
    pub fn close_active(&mut self) -> bool {
        self.close(self.active)
    }
}

#[cfg(test)]
mod tests {
    use super::{Workspace, WorkspaceManager, WorkspacePlacement};
    use std::path::{Path, PathBuf};

    fn ws(path: &str) -> Workspace {
        Workspace::new(PathBuf::from(path))
    }

    fn manager(paths: &[&str]) -> WorkspaceManager {
        let mut manager = WorkspaceManager::new(ws(paths[0]));
        for path in &paths[1..] {
            manager.open(ws(path));
        }
        manager
    }

    /// The paths of the open tabs, in order — handy for asserting the layout.
    fn layout(manager: &WorkspaceManager) -> Vec<&Path> {
        manager.workspaces().iter().map(Workspace::path).collect()
    }

    #[test]
    fn placement_parses_case_insensitively() {
        assert_eq!("top".parse(), Ok(WorkspacePlacement::Top));
        assert_eq!("Bottom".parse(), Ok(WorkspacePlacement::Bottom));
        assert_eq!("  LEFT ".parse(), Ok(WorkspacePlacement::Left));
        assert_eq!("Right".parse(), Ok(WorkspacePlacement::Right));
    }

    #[test]
    fn placement_defaults_to_top() {
        assert_eq!(WorkspacePlacement::default(), WorkspacePlacement::Top);
    }

    #[test]
    fn placement_rejects_unknown_values() {
        let err = "middle".parse::<WorkspacePlacement>().unwrap_err();
        assert!(
            err.contains("top|bottom|left|right"),
            "unexpected error: {err}"
        );
        assert!(err.contains("middle"), "unexpected error: {err}");
    }

    #[test]
    fn starts_with_one_active_tab() {
        let manager = WorkspaceManager::new(ws("/a"));
        assert_eq!(manager.len(), 1);
        assert!(!manager.is_empty());
        assert_eq!(manager.active_index(), 0);
        assert_eq!(manager.active().path(), Path::new("/a"));
    }

    #[test]
    fn open_appends_to_the_right_and_activates() {
        let mut manager = WorkspaceManager::new(ws("/a"));
        let index = manager.open(ws("/b"));
        assert_eq!(index, 1);
        assert_eq!(manager.active_index(), 1);
        assert_eq!(layout(&manager), [Path::new("/a"), Path::new("/b")]);
    }

    #[test]
    fn open_does_not_deduplicate() {
        let mut manager = WorkspaceManager::new(ws("/a"));
        manager.open(ws("/a"));
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn position_of_finds_open_paths_only() {
        let manager = manager(&["/a", "/b"]);
        assert_eq!(manager.position_of(Path::new("/a")), Some(0));
        assert_eq!(manager.position_of(Path::new("/b")), Some(1));
        assert_eq!(manager.position_of(Path::new("/c")), None);
    }

    #[test]
    fn open_or_switch_switches_when_already_open() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        manager.switch_to(0);
        let index = manager.open_or_switch(ws("/b"));
        assert_eq!(index, 1);
        assert_eq!(manager.active_index(), 1);
        // No new tab was opened.
        assert_eq!(manager.len(), 3);
    }

    #[test]
    fn open_or_switch_opens_when_not_present() {
        let mut manager = manager(&["/a", "/b"]);
        let index = manager.open_or_switch(ws("/c"));
        assert_eq!(index, 2);
        assert_eq!(manager.active_index(), 2);
        assert_eq!(manager.len(), 3);
    }

    #[test]
    fn switch_to_rejects_out_of_range() {
        let mut manager = manager(&["/a", "/b"]);
        assert!(manager.switch_to(1));
        assert_eq!(manager.active_index(), 1);
        assert!(!manager.switch_to(2));
        assert_eq!(manager.active_index(), 1);
    }

    #[test]
    fn focus_next_and_previous_wrap_around() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        assert_eq!(manager.active_index(), 2);
        manager.focus_next(); // wraps 2 -> 0
        assert_eq!(manager.active_index(), 0);
        manager.focus_previous(); // wraps 0 -> 2
        assert_eq!(manager.active_index(), 2);
        manager.focus_previous(); // 2 -> 1
        assert_eq!(manager.active_index(), 1);
        manager.focus_next(); // 1 -> 2
        assert_eq!(manager.active_index(), 2);
    }

    #[test]
    fn focus_navigation_is_a_no_op_with_one_tab() {
        let mut manager = WorkspaceManager::new(ws("/a"));
        manager.focus_next();
        assert_eq!(manager.active_index(), 0);
        manager.focus_previous();
        assert_eq!(manager.active_index(), 0);
    }

    #[test]
    fn close_rejects_the_last_tab() {
        let mut manager = WorkspaceManager::new(ws("/a"));
        assert!(!manager.close(0));
        assert!(!manager.close_active());
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.active().path(), Path::new("/a"));
    }

    #[test]
    fn close_rejects_out_of_range() {
        let mut manager = manager(&["/a", "/b"]);
        assert!(!manager.close(5));
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn closing_a_tab_renumbers_the_rest() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        // Close the middle tab; /c shifts down from index 2 to index 1.
        assert!(manager.close(1));
        assert_eq!(layout(&manager), [Path::new("/a"), Path::new("/c")]);
        assert_eq!(manager.position_of(Path::new("/c")), Some(1));
    }

    #[test]
    fn closing_a_tab_left_of_active_keeps_the_same_tab_active() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        manager.switch_to(2); // active = /c
        assert!(manager.close(0)); // remove /a
        // /c is still active, now at its shifted-down index.
        assert_eq!(manager.active().path(), Path::new("/c"));
        assert_eq!(manager.active_index(), 1);
    }

    #[test]
    fn closing_the_active_middle_tab_focuses_the_right_neighbour() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        manager.switch_to(1); // active = /b
        assert!(manager.close_active());
        // /c took /b's slot and becomes active.
        assert_eq!(manager.active().path(), Path::new("/c"));
        assert_eq!(manager.active_index(), 1);
    }

    #[test]
    fn closing_the_active_last_tab_focuses_the_new_last() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        assert_eq!(manager.active_index(), 2); // active = /c (rightmost)
        assert!(manager.close_active());
        assert_eq!(manager.active().path(), Path::new("/b"));
        assert_eq!(manager.active_index(), 1);
    }

    #[test]
    fn closing_a_tab_right_of_active_leaves_active_untouched() {
        let mut manager = manager(&["/a", "/b", "/c"]);
        manager.switch_to(0); // active = /a
        assert!(manager.close(2)); // remove /c
        assert_eq!(manager.active().path(), Path::new("/a"));
        assert_eq!(manager.active_index(), 0);
    }
}
