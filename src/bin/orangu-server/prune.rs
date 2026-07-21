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

//! `orangu-server prune`: deletes chat sessions from
//! `~/.orangu/server/sessions/` (`web::sessions`). Needs no config file and
//! loads no model — a pure filesystem operation against a fixed path, the
//! same shape as `system`/`suggest`.
//!
//! Every invocation, regardless of its own argument, first sweeps away
//! every non-active session with an empty chat history
//! (`sessions::sweep_empty_sessions`) — junk from a "New Chat" click that
//! was never used, or a leftover from an interrupted write — and every stale
//! persisted slot KV-cache file (`engine::slot_store::sweep_stale_slot_files`,
//! the `<fingerprint>/slots/*.bin` snapshots written by
//! `POST /slots/{id}?action=save`) before doing anything else. What's left is
//! then handled by argument:
//!
//! - No argument: prints the remaining sessions as a numbered table, sorted
//!   newest-updated-first, and prompts for an `NR` or `all`.
//! - `all`: deletes every remaining **non-active** session. A session is
//!   "active" when its `session.json` (`sessions::is_active`) names a pid
//!   that's still running the process that last touched it — written by
//!   the server that owns it, read here by this separate CLI invocation,
//!   so a session started long after some other still-running server's own
//!   startup is still correctly protected.
//! - An `NR` (from this command's own listing) or a full session id: prunes
//!   that one session, refusing (not erroring) if it's active.

use crate::web::sessions::{self, PruneEntry};
use anyhow::{Context, Result, anyhow};
use std::{
    io::Write,
    time::{SystemTime, UNIX_EPOCH},
};

/// Slot KV-cache files untouched for this long are treated as abandoned and
/// swept on every `prune` run. Conservative because the sweep is unconditional
/// and a file's client-side session can't be checked (see
/// `engine::slot_store::sweep_stale_slot_files`) — but these are pure caches,
/// so the only cost of over-eager deletion is a one-time reprefill.
const SLOT_FILE_MAX_AGE_DAYS: u64 = 30;

pub fn run(identifier: Option<String>, yes: bool) -> Result<()> {
    let removed_empty = sessions::sweep_empty_sessions()?;
    if removed_empty > 0 {
        let plural = if removed_empty == 1 { "" } else { "s" };
        println!("Removed {removed_empty} empty session{plural}.");
    }

    let slots = crate::engine::slot_store::sweep_stale_slot_files(std::time::Duration::from_secs(
        SLOT_FILE_MAX_AGE_DAYS * 24 * 60 * 60,
    ));
    if slots.removed > 0 {
        let plural = if slots.removed == 1 { "" } else { "s" };
        println!(
            "Removed {} stale slot cache file{plural} ({}).",
            slots.removed,
            format_bytes(slots.bytes),
        );
    }

    let entries = sessions::list_sessions_for_prune()?;

    match identifier.as_deref() {
        None => prune_interactive(&entries, yes),
        Some(id) if id.eq_ignore_ascii_case("all") => prune_all(&entries, yes),
        Some(id) => prune_one(resolve_entry(&entries, id)?, yes),
    }
}

fn prune_interactive(entries: &[PruneEntry], yes: bool) -> Result<()> {
    if entries.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }
    print_table(entries);

    print!("\nPrune (NR or 'all', empty to cancel): ");
    std::io::stdout().flush().ok();
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read selection")?;
    let input = input.trim();
    if input.is_empty() {
        println!("Aborted. Nothing pruned.");
        return Ok(());
    }
    if input.eq_ignore_ascii_case("all") {
        return prune_all(entries, yes);
    }
    prune_one(resolve_entry(entries, input)?, yes)
}

/// Resolves an `NR` (from this command's own listing, one-indexed like
/// `list`'s/`delete`'s own `NR`) or a full session id against `entries`.
fn resolve_entry<'a>(entries: &'a [PruneEntry], identifier: &str) -> Result<&'a PruneEntry> {
    if let Ok(nr) = identifier.parse::<usize>() {
        let count = entries.len();
        return nr
            .checked_sub(1)
            .and_then(|index| entries.get(index))
            .ok_or_else(|| anyhow!("no session with NR {nr} ({count} session(s) listed)"));
    }
    entries
        .iter()
        .find(|e| e.id == identifier)
        .ok_or_else(|| anyhow!("no session '{identifier}' found (not an NR or a known session id)"))
}

fn prune_one(entry: &PruneEntry, yes: bool) -> Result<()> {
    if entry.active {
        println!(
            "Session '{}' is active (in use by a running orangu-server) — not pruned.",
            entry.id
        );
        return Ok(());
    }
    if !yes {
        let confirmed = crate::confirm(&format!(
            "Delete session '{}' ({}, {})? [y/N]: ",
            entry.id,
            display_title(entry),
            message_count_label(entry.message_count),
        ))?;
        if !confirmed {
            println!("Aborted. Nothing pruned.");
            return Ok(());
        }
    }
    sessions::delete_session_dir(&entry.id)?;
    println!("Pruned session '{}'", entry.id);
    Ok(())
}

fn prune_all(entries: &[PruneEntry], yes: bool) -> Result<()> {
    let (active, inactive): (Vec<_>, Vec<_>) = entries.iter().partition(|e| e.active);
    if inactive.is_empty() {
        println!("No non-active sessions to prune.");
        if !active.is_empty() {
            println!("({} active session(s) skipped)", active.len());
        }
        return Ok(());
    }

    println!("This will delete {} session(s):", inactive.len());
    for entry in &inactive {
        println!("  {} {}", entry.id, display_title(entry));
    }
    if !active.is_empty() {
        let plural = if active.len() == 1 { "" } else { "s" };
        println!("Skipping {} active session{plural}.", active.len());
    }

    if !yes {
        let confirmed = crate::confirm(&format!("Delete {} session(s)? [y/N]: ", inactive.len()))?;
        if !confirmed {
            println!("Aborted. Nothing pruned.");
            return Ok(());
        }
    }

    let mut pruned = 0usize;
    for entry in &inactive {
        match sessions::delete_session_dir(&entry.id) {
            Ok(()) => pruned += 1,
            Err(err) => eprintln!("failed to delete session '{}': {err:#}", entry.id),
        }
    }
    println!("Pruned {pruned} session(s).");
    Ok(())
}

fn display_title(entry: &PruneEntry) -> &str {
    if entry.title.is_empty() {
        "(untitled)"
    } else {
        &entry.title
    }
}

fn message_count_label(count: usize) -> String {
    format!("{count} message{}", if count == 1 { "" } else { "s" })
}

fn print_table(entries: &[PruneEntry]) {
    let nr_width = entries.len().to_string().len().max("NR".len());
    let id_width = entries
        .iter()
        .map(|e| e.id.len())
        .max()
        .unwrap_or(0)
        .max("ID".len());
    let title_width = entries
        .iter()
        .map(|e| display_title(e).chars().count())
        .max()
        .unwrap_or(0)
        .max("TITLE".len());

    println!(
        "{:>nr_width$}  {:<id_width$}  {:<title_width$}  MESSAGES  UPDATED",
        "NR", "ID", "TITLE"
    );
    for (index, entry) in entries.iter().enumerate() {
        let nr = index + 1;
        let active = if entry.active { "  (active)" } else { "" };
        println!(
            "{nr:>nr_width$}  {:<id_width$}  {:<title_width$}  {:>8}  {}{active}",
            entry.id,
            display_title(entry),
            entry.message_count,
            format_relative(entry.updated_at),
        );
    }
}

/// A short human-readable byte size (`512 B`, `3.4 MB`, `1.2 GB`) for the
/// reclaimed-space line — binary units, one decimal past `KB`.
fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < KB * KB {
        format!("{:.1} KB", b / KB)
    } else if b < KB * KB * KB {
        format!("{:.1} MB", b / (KB * KB))
    } else {
        format!("{:.1} GB", b / (KB * KB * KB))
    }
}

/// A short, human-scale "how long ago" for `updated_at` — deliberately not
/// a calendar date/time (which would need either a date/time dependency or
/// the same hand-rolled calendar math `web::current_year` already uses just
/// for a year) since relative recency is what matters for picking a session
/// to prune, not its exact timestamp.
fn format_relative(unix_ts: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let elapsed = now.saturating_sub(unix_ts);
    if elapsed < 60 {
        "just now".to_string()
    } else if elapsed < 3600 {
        format!("{}m ago", elapsed / 60)
    } else if elapsed < 86400 {
        format!("{}h ago", elapsed / 3600)
    } else {
        format!("{}d ago", elapsed / 86400)
    }
}
