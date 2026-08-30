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

pub const BASH: &str = r#"# bash completion for orangu
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu -s > ~/.local/share/bash-completion/completions/orangu

_orangu() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    local word loop_mode=false
    for word in "${COMP_WORDS[@]:1:COMP_CWORD}"; do
        [[ "$word" == "loop" ]] && loop_mode=true
    done
    if $loop_mode; then
        case "$prev" in
            --file)
                COMPREPLY=( $(compgen -f -- "$cur") )
                compopt -o filenames 2>/dev/null
                return 0
                ;;
        esac
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "--turns --time --until --file --check --review --help" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "status pause resume clear" -- "$cur") )
        fi
        return 0
    fi

    case "$prev" in
        -c|--config|--workflow)
            # Configuration or YAML workflow file.
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        -w|--workspace)
            # Workspace root: unique workspaces from past sessions in
            # ~/.orangu/sessions, extracted from each session's metadata.
            local sessions_dir="${HOME}/.orangu/sessions"
            if [[ -d "$sessions_dir" ]]; then
                local workspaces
                workspaces=$(sed -n 's/.*"workspace":"\([^"]*\)".*/\1/p' \
                    "$sessions_dir"/*/metadata 2>/dev/null | sort -u)
                COMPREPLY=( $(compgen -W "$workspaces" -- "$cur") )
            fi
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        -r|--resume)
            # Session UUID: scan ~/.orangu/sessions, newest first
            local sessions_dir="${HOME}/.orangu/sessions"
            if [[ -d "$sessions_dir" ]]; then
                local uuids
                uuids=$(command ls -1t "$sessions_dir" 2>/dev/null)
                COMPREPLY=( $(compgen -W "$uuids" -- "$cur") )
            fi
            return 0
            ;;
        -t|--theme)
            COMPREPLY=( $(compgen -W "classic modern_dark modern_light oranguday tokyonight rosepine-moon random" -- "$cur") $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "-c --config -t --theme -w --workspace -r --resume -a --all -p --prompt --workflow --dry-run -q --quiet -l --list -i --init -s --shell-completions -h --help" -- "$cur") )
        return 0
    fi
    COMPREPLY=( $(compgen -W "loop" -- "$cur") )
}

complete -F _orangu orangu
"#;

pub const ZSH: &str = r#"#compdef orangu
# zsh completion for orangu
#
# Quick setup — add to ~/.zshrc:
#   eval "$(orangu -s)"
#
# Or write once to your fpath directory:
#   orangu -s > ~/.zsh/completions/_orangu
#   # ~/.zshrc: fpath=(~/.zsh/completions $fpath) && autoload -Uz compinit && compinit

# Completes session UUIDs from ~/.orangu/sessions, newest first.
_orangu_sessions() {
    local sessions_dir="${HOME}/.orangu/sessions"
    local -a uuids
    [[ -d $sessions_dir ]] && uuids=( $sessions_dir/*(/Nom:t) )
    _describe -t sessions 'session' uuids
}

# Completes unique workspace roots from past sessions in ~/.orangu/sessions,
# extracted from each session's metadata.
_orangu_workspaces() {
    local sessions_dir="${HOME}/.orangu/sessions"
    [[ -d $sessions_dir ]] || return
    local -a workspaces
    workspaces=( ${(fu)"$(sed -n 's/.*"workspace":"\([^"]*\)".*/\1/p' \
        $sessions_dir/*/metadata(N) 2>/dev/null)"} )
    compadd -a workspaces
}

_orangu_themes() {
    local -a themes
    themes=(classic modern_dark modern_light oranguday tokyonight rosepine-moon random)
    compadd -a themes
    _files -g '*.theme'
}

_orangu() {
    _arguments -s \
        '(-c --config)'{-c,--config}'[Path to the configuration file (orangu.conf)]:config file:_files' \
        '(-t --theme)'{-t,--theme}'[Override the TUI theme with a name or .theme file]:theme:_orangu_themes' \
        '(-w --workspace)'{-w,--workspace}'[Workspace root for local tools]:workspace:_orangu_workspaces' \
        '(-r --resume)'{-r,--resume}'[Resume a session by UUID]:session uuid:_orangu_sessions' \
        '(-a --all)'{-a,--all}'[Reopen the workspace tabs from the previous run]' \
        '(-p --prompt)'{-p,--prompt}'[Run one prompt or command, print the result and exit]:prompt:' \
        '--workflow[Validate and execute every job in a YAML workflow]:workflow file:_files' \
        '--dry-run[Validate the workflow without executing it]' \
        '(-q --quiet)'{-q,--quiet}'[Print nothing on success; the exit code is the result]' \
        '(-l --list)'{-l,--list}'[List all stored sessions as a table and exit]' \
        '(-i --init)'{-i,--init}'[Interactively create ~/.orangu/orangu.conf and exit]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '1:command:(loop)' \
        '*::argument:->arguments'

    if [[ $state == arguments && $words[2] == loop ]]; then
        _arguments \
            '--turns[Stop after a number of work-and-review iterations]:count:' \
            '--time[Stop after an active-time duration]:duration:' \
            '--until[Continue until review verifies a condition]:condition:' \
            '--file[Load a reusable loop definition]:loop file:_files' \
            '*--check[Run a validation command after each work phase]:command:' \
            '*--review[Add a review criterion]:criterion:' \
            '1:action:(status pause resume clear)' \
            '*:objective:'
    fi
}

_orangu "$@"
"#;

pub const FISH: &str = r#"# fish completion for orangu
#
# Quick setup — add to ~/.config/fish/config.fish:
#   orangu -s | source
#
# Or write once to the fish completions directory:
#   orangu -s > ~/.config/fish/completions/orangu.fish

# Completes session UUIDs from ~/.orangu/sessions, newest first.
function __orangu_sessions
    set -l sessions_dir "$HOME/.orangu/sessions"
    test -d "$sessions_dir"; or return
    path basename (path sort --reverse --key=mtime $sessions_dir/*/)
end

# Completes unique workspace roots from past sessions in ~/.orangu/sessions,
# extracted from each session's metadata.
function __orangu_workspaces
    set -l sessions_dir "$HOME/.orangu/sessions"
    test -d "$sessions_dir"; or return
    sed -n 's/.*"workspace":"\([^"]*\)".*/\1/p' $sessions_dir/*/metadata 2>/dev/null | sort -u
end

function __orangu_using_loop
    contains -- loop (commandline -opc)
end

complete -c orangu -s c -l config           -r                          -d 'Path to the configuration file (orangu.conf)'
complete -c orangu -s t -l theme             -r -a 'classic modern_dark modern_light oranguday tokyonight rosepine-moon random' -d 'Override the TUI theme with a name or .theme file'
complete -c orangu -s t -l theme             -r -a '(__fish_complete_path)' -d 'Theme file'
complete -c orangu -s w -l workspace         -x -a '(__orangu_workspaces)' -d 'Workspace root for local tools'
complete -c orangu -s r -l resume            -x -a '(__orangu_sessions)'   -d 'Resume a session by UUID'
complete -c orangu -s a -l all                                            -d 'Reopen the workspace tabs from the previous run'
complete -c orangu -s p -l prompt            -x                           -d 'Run one prompt or command, print the result and exit'
complete -c orangu      -l workflow          -r -a '(__fish_complete_path)' -d 'Validate and execute every job in a YAML workflow'
complete -c orangu      -l dry-run          -d 'Validate the workflow without executing it'
complete -c orangu -s q -l quiet                                          -d 'Print nothing on success; the exit code is the result'
complete -c orangu -s l -l list                                           -d 'List all stored sessions as a table and exit'
complete -c orangu -s i -l init                                           -d 'Interactively create ~/.orangu/orangu.conf and exit'
complete -c orangu -s s -l shell-completions                              -d 'Print shell completion script for the detected shell and exit'
complete -c orangu -s h -l help                                           -d 'Print help'
complete -c orangu -n 'not __orangu_using_loop' -f -a loop                -d 'Run a bounded code-and-review loop'
complete -c orangu -n '__orangu_using_loop' -l turns -r                   -d 'Stop after a number of work-and-review iterations'
complete -c orangu -n '__orangu_using_loop' -l time -r                    -d 'Stop after an active-time duration'
complete -c orangu -n '__orangu_using_loop' -l until -r                   -d 'Continue until review verifies a condition'
complete -c orangu -n '__orangu_using_loop' -l file -r -a '(__fish_complete_path)' -d 'Load a reusable loop definition'
complete -c orangu -n '__orangu_using_loop' -l check -r                   -d 'Run a validation command after each work phase'
complete -c orangu -n '__orangu_using_loop' -l review -r                  -d 'Add a review criterion'
complete -c orangu -n '__orangu_using_loop' -f -a 'status pause resume clear' -d 'Manage the saved loop'
"#;

#[cfg(test)]
mod tests {
    use super::{BASH, FISH, ZSH};

    /// The scripts are hand-written, so a new command-line option reaches them
    /// only if someone remembers. Ask clap what the options actually are.
    #[test]
    fn every_shell_completes_every_command_line_option() {
        use clap::CommandFactory;

        for argument in crate::Args::command().get_arguments() {
            // Hidden options — `--build-manual`, `--build-cheatsheet` — are
            // repository development tools, not workspace features: they stay
            // out of `--help` and out of the completions with it.
            if argument.is_hide_set() {
                continue;
            }
            let Some(long) = argument.get_long() else {
                continue;
            };
            // fish spells the long form `-l <name>`; bash and zsh spell it out.
            for (shell, script, needle) in [
                ("bash", BASH, format!("--{long}")),
                ("zsh", ZSH, format!("--{long}")),
                ("fish", FISH, format!("-l {long}")),
            ] {
                assert!(
                    script.contains(&needle),
                    "{shell} completion omits the option: --{long}"
                );
            }

            let Some(short) = argument.get_short() else {
                continue;
            };
            // zsh pairs the two forms and fish flags the short one; bash lists
            // it as a bare word, so match whole words there rather than a
            // substring (`-t` lives inside `--theme`).
            assert!(
                BASH.split(|c: char| c.is_whitespace() || c == '"')
                    .any(|word| word == format!("-{short}")),
                "bash completion omits the short option: -{short}"
            );
            for (shell, script, needle) in [
                ("zsh", ZSH, format!("{{-{short},--{long}}}")),
                ("fish", FISH, format!("-s {short} ")),
            ] {
                assert!(
                    script.contains(&needle),
                    "{shell} completion omits the short option: -{short}"
                );
            }
        }
    }

    #[test]
    fn every_shell_completes_all_built_in_themes() {
        // The scripts are emitted verbatim, so their theme lists are the one
        // place that can silently fall behind `BUILT_IN_THEMES`. Adding a
        // shipped theme must add it here too.
        for theme in orangu::tui::Theme::built_in_theme_names() {
            for (shell, script) in [("bash", BASH), ("zsh", ZSH), ("fish", FISH)] {
                assert!(
                    script.contains(&theme),
                    "{shell} completion omits the built-in theme: {theme}"
                );
            }
        }
    }

    #[test]
    fn every_shell_completes_the_loop_interface() {
        for option in ["turns", "time", "until", "file", "check", "review"] {
            for (shell, script, needle) in [
                ("bash", BASH, format!("--{option}")),
                ("zsh", ZSH, format!("--{option}")),
                ("fish", FISH, format!("-l {option}")),
            ] {
                assert!(
                    script.contains(&needle),
                    "{shell} completion omits --{option}"
                );
            }
        }
        for value in ["loop", "status", "pause", "resume", "clear"] {
            for (shell, script) in [("bash", BASH), ("zsh", ZSH), ("fish", FISH)] {
                assert!(script.contains(value), "{shell} completion omits {value}");
            }
        }
    }
}
