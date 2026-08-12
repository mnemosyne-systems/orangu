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

use std::path::Path;
use strum::IntoEnumIterator;

use super::*;
use crate::commands::NATURAL_LANGUAGE_BINDINGS;

/// The grey inline ghost suffix to draw after the cursor for `input`, or `None`
/// when there is nothing to hint. Slash commands take priority over the merge
/// flow's next step and the natural-language bindings (with `ghost_index`
/// picking which cycled candidate to preview), and structured argument
/// completions — branches, tags, files, models, servers — fall last. Only
/// hinted while the cursor sits at the end of the typed text. Shared by the
/// main prompt and the `/review` / `/auto_review` input windows so all three
/// preview completions the same way.
pub fn input_ghost_suffix(
    input: &str,
    cursor: usize,
    ghost_index: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
    skills: &orangu::skills::SkillRegistry,
) -> Option<String> {
    if cursor != input.len() {
        return None;
    }
    command_ghost_suffix(input, skills)
        .or_else(|| ghost_suffix_at(input, ghost_index, workspace))
        .or_else(|| {
            completion_ghost_suffix(
                input,
                cursor,
                workspace,
                server_names,
                available_models,
                skills,
            )
        })
}

/// Returns the trailing characters needed to finish the slash command the user
/// is part-way through typing, e.g. `/q` -> `uit` (completing `/quit`). This is
/// the grey "ghost" hint shown inline after the cursor; pressing Tab fills it in.
///
/// Returns `None` unless `input` is a lone slash-command prefix still being
/// typed (no whitespace yet) that matches a known command. The first matching
/// command in [`COMMANDS`] wins, so the suggestion narrows as more letters are
/// typed. An already-complete command yields `None`.
pub fn command_ghost_suffix(input: &str, skills: &orangu::skills::SkillRegistry) -> Option<String> {
    if !input.starts_with('/') || input.chars().any(char::is_whitespace) {
        return None;
    }
    if let Some(candidate) = crate::slash_command::SlashCommand::iter()
        .map(|cmd| cmd.command())
        .find(|command| command.starts_with(input))
    {
        return candidate
            .strip_prefix(input)
            .filter(|rest| !rest.is_empty())
            .map(|s| s.to_string());
    }
    for skill in skills.all() {
        let cmd = format!("/{}", skill.name);
        if cmd.starts_with(input) {
            return cmd
                .strip_prefix(input)
                .filter(|rest| !rest.is_empty())
                .map(|s| s.to_string());
        }
    }
    None
}

/// Every natural-language binding the user's part-typed input could still grow
/// into, as the trailing characters needed to complete each one. For input `c`
/// this yields `urrent model`, `ode review`, `heckout `, ... (completing
/// `current model`, `code review`, `checkout `, ...). The list drives the grey "ghost" hint and
/// its Shift+Tab cycling; index 0 is what `natural_language_ghost_suffix`
/// returns.
///
/// Matching is ASCII case-insensitive, mirroring the parser, and candidates keep
/// [`NATURAL_LANGUAGE_BINDINGS`] (parser priority) order. Bindings that differ
/// only by trailing whitespace (e.g. `checkout ` vs `checkout`) render
/// identically, so only the first is kept. Empty input, slash input, and input
/// that already spells a complete binding (e.g. `status`, `diff`) yield an
/// empty list — there is nothing left to hint.
pub fn natural_language_ghost_candidates(input: &str) -> Vec<&'static str> {
    if input.is_empty() || input.starts_with('/') {
        return Vec::new();
    }
    if NATURAL_LANGUAGE_BINDINGS
        .iter()
        .any(|binding| binding.eq_ignore_ascii_case(input))
    {
        return Vec::new();
    }
    let mut seen: Vec<&str> = Vec::new();
    let mut candidates: Vec<&'static str> = Vec::new();
    for binding in NATURAL_LANGUAGE_BINDINGS {
        if binding.len() <= input.len()
            || !binding.as_bytes()[..input.len()].eq_ignore_ascii_case(input.as_bytes())
        {
            continue;
        }
        let suffix = &binding[input.len()..];
        if suffix.trim().is_empty() {
            continue;
        }
        let key = binding.trim_end();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        candidates.push(suffix);
    }
    candidates
}

/// Everything the part-typed `input` could still grow into, as the trailing
/// characters that complete each candidate — the merge flow's remaining steps
/// first (see [`flow::flow_candidates`]), then the natural-language bindings.
/// This is the list the grey ghost previews and Shift+Tab cycles through, so
/// the next step of the flow is always what the prompt offers first: on an
/// empty line it hints `switch to main` / `merge <branch>` / ... outright,
/// where a binding alone would have nothing to say.
///
/// A flow step that spells out a binding already offered (e.g. `push`) is kept
/// once, in the flow's position.
pub fn ghost_candidates(input: &str, workspace: &Path) -> Vec<String> {
    let mut candidates: Vec<String> = flow::flow_candidates(input, workspace)
        .iter()
        .filter_map(|command| command.get(input.len()..))
        .filter(|suffix| !suffix.is_empty())
        .map(str::to_string)
        .collect();
    for suffix in natural_language_ghost_candidates(input) {
        if !candidates.iter().any(|seen| seen == suffix) {
            candidates.push(suffix.to_string());
        }
    }
    candidates
}

/// The ghost suffix to preview at cycle position `index`, wrapping around
/// [`ghost_candidates`]. Index 0 is what the prompt draws; Shift+Tab advances
/// it and Tab accepts whatever is shown.
pub fn ghost_suffix_at(input: &str, index: usize, workspace: &Path) -> Option<String> {
    let candidates = ghost_candidates(input, workspace);
    if candidates.is_empty() {
        return None;
    }
    Some(candidates[index % candidates.len()].clone())
}

/// The leading single word of a natural-language ghost `suffix`, including the
/// whitespace that trails it, so Tab accepts a multi-word binding one word at a
/// time. For `"h force"` (completing `push` then `force`) this is `"h "`; for a
/// suffix with no internal whitespace such as `"onnect"` it is the whole suffix.
/// Keeping the trailing space matters: accepting `pus` -> `push ` leaves the
/// ghost alive so the next word (`force`) can be previewed and accepted in turn.
pub fn first_ghost_word(suffix: &str) -> &str {
    let Some(word_start) = suffix.find(|ch: char| !ch.is_whitespace()) else {
        return suffix;
    };
    let Some(rel_end) = suffix[word_start..].find(char::is_whitespace) else {
        return suffix;
    };
    let word_end = word_start + rel_end;
    let next = suffix[word_end..]
        .find(|ch: char| !ch.is_whitespace())
        .map(|index| word_end + index)
        .unwrap_or(suffix.len());
    &suffix[..next]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::parse_local_command;
    use crate::completion::flow::MergeFlowStep;
    use crate::completion::flow::test_support::begin;
    use crate::input::{InputState, apply_completion, cycle_ghost_suggestion};
    use crate::test_support::exclusive_committer_prompt as exclusive;
    use tempfile::tempdir;

    /// The whole prompt hint for `input`, the way the screen draws it.
    fn ghost_for(input: &str, workspace: &Path) -> Option<String> {
        input_ghost_suffix(
            input,
            input.len(),
            0,
            workspace,
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(Path::new("/")),
        )
    }

    #[test]
    fn merge_flow_leads_the_ghost_from_the_empty_prompt_on() {
        let _guard = exclusive();
        let workspace = tempdir().expect("workspace");
        begin(231, "pr-231", MergeFlowStep::Switch, workspace.path());

        // The empty prompt hints the step to take next — a binding alone has
        // nothing to say there.
        assert_eq!(
            ghost_for("", workspace.path()).as_deref(),
            Some("switch to main")
        );
        // Shift+Tab walks the optional steps and then the rest of the flow,
        // all before the bindings.
        assert_eq!(
            ghost_candidates("", workspace.path())[..5],
            [
                "switch to main",
                "build",
                "review",
                "auto review",
                "merge pr-231"
            ]
        );
        // A part-typed line keeps the flow's step ahead of the binding that
        // shares its prefix, and completes the remembered branch...
        assert_eq!(
            ghost_for("m", workspace.path()).as_deref(),
            Some("erge pr-231")
        );
        // ...including after the binding itself is complete, where the plain
        // binding ghost stops.
        assert_eq!(
            ghost_for("merge", workspace.path()).as_deref(),
            Some(" pr-231")
        );
        // Slash commands still own the hint.
        assert_eq!(ghost_for("/q", workspace.path()).as_deref(), Some("uit"));
    }

    #[test]
    fn tab_accepts_the_flow_step_one_word_at_a_time() {
        let _guard = exclusive();
        let workspace = tempdir().expect("workspace");
        begin(231, "pr-231", MergeFlowStep::Comment, workspace.path());

        let mut input_state = InputState::default();
        let skills = orangu::skills::SkillRegistry::discover(Path::new("/"));
        for expected in ["comment ", "comment on ", "comment on 231 "] {
            apply_completion(&mut input_state, workspace.path(), &[], &[], &skills);
            assert_eq!(input_state.as_str(), expected);
        }
        apply_completion(&mut input_state, workspace.path(), &[], &[], &skills);
        assert_eq!(input_state.as_str(), "comment on 231 merged.md");
        assert!(parse_local_command(input_state.as_str()).is_some());
    }

    #[test]
    fn get_comments_ghost_offers_issue_and_pull_request() {
        // After `get comments for ` the ghost hint cycles between the two
        // targets; once a target is partially typed only it remains.
        assert_eq!(
            natural_language_ghost_candidates("get comments for "),
            vec!["issue ", "pull request "]
        );
        assert_eq!(
            natural_language_ghost_candidates("get comments for p"),
            vec!["ull request "]
        );
    }

    #[test]
    fn suggests_ghost_suffix_for_partial_slash_commands() {
        // A unique prefix completes to the rest of the command.
        assert_eq!(
            command_ghost_suffix(
                "/q",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            Some("uit".to_string())
        );
        assert_eq!(
            command_ghost_suffix(
                "/qui",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            Some("t".to_string())
        );

        // The first matching command wins, so the hint narrows as letters arrive.
        assert_eq!(
            command_ghost_suffix(
                "/",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            Some("help".to_string())
        );

        // A fully typed command and unmatched prefixes have nothing to suggest.
        assert_eq!(
            command_ghost_suffix(
                "/quit",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            None
        );
        assert_eq!(
            command_ghost_suffix(
                "/zzz",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            None
        );

        // Once an argument is being typed (whitespace) the name hint stops.
        assert_eq!(
            command_ghost_suffix(
                "/quit ",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            None
        );
        assert_eq!(
            command_ghost_suffix(
                "not a command",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            None
        );
    }

    #[test]
    fn suggests_ghost_suffix_for_partial_natural_language_bindings() {
        // The rendered hint is cycle position 0.
        let ghost = |input| natural_language_ghost_candidates(input).first().copied();

        // A partial verb completes to the rest of the binding.
        assert_eq!(ghost("discon"), Some("nect"));
        assert_eq!(ghost("rebas"), Some("e"));

        // Argument-taking prefixes complete through their trailing space.
        assert_eq!(ghost("diff a"), Some("gainst "));
        assert_eq!(ghost("use s"), Some("erver "));

        // Matching is case-insensitive; the suggested suffix is canonical.
        assert_eq!(ghost("DIF"), Some("f"));

        // A complete binding has nothing left to hint, even when a longer
        // binding shares its prefix (e.g. "diff" vs "diff against ").
        assert_eq!(ghost("commit"), None);
        assert_eq!(ghost("merge"), None);
        assert_eq!(ghost("diff"), None);

        // Still hinted while the binding is incomplete.
        assert_eq!(ghost("c"), Some("urrent model"));

        // Empty input, slash input, and unknown prefixes suggest nothing.
        assert_eq!(ghost(""), None);
        assert_eq!(ghost("/q"), None);
        assert_eq!(ghost("xyzzy"), None);
    }

    #[test]
    fn first_ghost_word_accepts_one_word_at_a_time() {
        // A multi-word suffix yields just the leading word plus its trailing
        // space, so "pus" -> "push " (with "force" left to preview next).
        assert_eq!(first_ghost_word("h force"), "h ");
        assert_eq!(first_ghost_word("comment on "), "comment ");
        // A single-word suffix is taken whole, trailing space and all.
        assert_eq!(first_ghost_word("onnect"), "onnect");
        assert_eq!(first_ghost_word("gainst "), "gainst ");
        // Degenerate suffixes are returned untouched.
        assert_eq!(first_ghost_word(""), "");
        assert_eq!(first_ghost_word("force"), "force");
    }

    #[test]
    fn shift_tab_cycles_through_natural_language_candidates() {
        // "c" matches several bindings; cycling walks them in priority order and
        // wraps back to the first. Bindings differing only by trailing whitespace
        // (e.g. "checkout " vs "checkout") collapse to one entry.
        let workspace = tempdir().expect("workspace");
        let candidates = natural_language_ghost_candidates("c");
        assert!(
            candidates.len() > 1,
            "expected multiple candidates for \"c\", got {candidates:?}"
        );
        // With no merge flow in progress the cycled list is the bindings' own.
        let cycled = |index| ghost_suffix_at("c", index, workspace.path());
        assert_eq!(cycled(0).as_deref(), Some(candidates[0]));
        assert_eq!(cycled(1).as_deref(), Some(candidates[1]));
        // Index wraps around the candidate list.
        assert_eq!(cycled(candidates.len()).as_deref(), Some(candidates[0]));

        // The whole list completes "c" to distinct, real commands.
        for suffix in candidates {
            let completed = format!("c{suffix}");
            assert!(
                parse_local_command(completed.trim()).is_some()
                    || parse_local_command(&format!("{completed}1")).is_some()
                    || parse_local_command(&format!("{completed}1 2")).is_some(),
                "cycled candidate {completed:?} does not parse"
            );
        }
    }

    #[test]
    fn tab_accepts_natural_language_ghost_suggestion() {
        let workspace = tempdir().expect("workspace");

        // Tab fills in the ghosted binding one word at a time, so a multi-word
        // binding grows with each press rather than landing all at once. Typing
        // "pus" completes to "push " (with "force" then previewed as the ghost),
        // and the next Tab accepts that word too.
        let mut input_state = InputState::default();
        input_state.set_buffer("pus".to_string());
        apply_completion(
            &mut input_state,
            workspace.path(),
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
        );
        assert_eq!(input_state.as_str(), "push ");
        assert_eq!(input_state.cursor(), "push ".len());
        assert_eq!(
            natural_language_ghost_candidates("push ").first().copied(),
            Some("force")
        );
        apply_completion(
            &mut input_state,
            workspace.path(),
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
        );
        assert_eq!(input_state.as_str(), "push force");

        // A fully typed binding has no ghost, so Tab leaves it untouched.
        let mut input_state = InputState::default();
        input_state.set_buffer("commit".to_string());
        apply_completion(
            &mut input_state,
            workspace.path(),
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
        );
        assert_eq!(input_state.as_str(), "commit");

        // The binding ghost wins over a same-prefixed filename: typing "c" with
        // a "contrib/" directory present completes to "current " (the first word
        // of "current model"), not "contrib/".
        let repo = tempdir().expect("repo");
        std::fs::create_dir(repo.path().join("contrib")).expect("contrib dir");
        let mut input_state = InputState::default();
        input_state.set_buffer("c".to_string());
        apply_completion(
            &mut input_state,
            repo.path(),
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
        );
        assert_eq!(input_state.as_str(), "current ");

        // Shift+Tab advances the preview; Tab then accepts the first word of the
        // shown candidate (word-at-a-time).
        let mut input_state = InputState::default();
        input_state.set_buffer("c".to_string());
        let second = format!(
            "c{}",
            first_ghost_word(natural_language_ghost_candidates("c")[1])
        );
        cycle_ghost_suggestion(&mut input_state, workspace.path());
        assert_eq!(input_state.ghost_index, 1);
        apply_completion(
            &mut input_state,
            workspace.path(),
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
        );
        assert_eq!(input_state.as_str(), second);

        // Editing the line resets the cycle back to the first candidate.
        let mut input_state = InputState::default();
        input_state.set_buffer("c".to_string());
        cycle_ghost_suggestion(&mut input_state, workspace.path());
        input_state.insert_char('o');
        assert_eq!(input_state.ghost_index, 0);
    }
}
