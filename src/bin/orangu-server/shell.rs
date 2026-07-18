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

//! Hand-written shell completion scripts, mirroring `orangu`'s own
//! `-s`/`--shell-completions` (`src/bin/orangu/shell.rs`): the positional
//! `model` argument (and `show`'s own argument) are completed by shelling
//! back out to `orangu-server list` itself and reading its first two
//! columns (NR and MODEL). This only ever depends on `orangu-server` itself
//! being on `$PATH` — no clap-generated completion machinery is involved.
//!
//! `prune`'s own argument is completed differently: directly against
//! `~/.orangu/server/sessions/*` (each entry a UUID directory), newest
//! first, plus the literal `all` — the same trick `orangu`'s own scripts
//! use to complete its `-r`/`--resume`'s session UUIDs
//! (`src/bin/orangu/shell.rs`). No process invocation involved (unlike the
//! model-completion trick above), since `prune` itself may prompt on stdin
//! when run with no argument — piping its output into a completion
//! function the way `list`'s is used for models would risk the completion
//! hanging on that prompt.

pub const BASH: &str = r#"# bash completion for orangu-server
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-server -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-server -s > ~/.local/share/bash-completion/completions/orangu-server

# Completes the positional MODEL argument (and `show`'s own argument) with
# every NR and MODEL from `orangu-server list`'s output.
_orangu_server_models() {
    orangu-server list 2>/dev/null | awk 'NR>1 {print $1; print $2}'
}

# Completes `prune`'s own argument: every session UUID under
# ~/.orangu/server/sessions, newest first, plus the literal "all".
_orangu_server_sessions() {
    local sessions_dir="${HOME}/.orangu/server/sessions"
    local uuids=""
    if [[ -d "$sessions_dir" ]]; then
        uuids=$(command ls -1t "$sessions_dir" 2>/dev/null)
    fi
    printf 'all\n%s\n' "$uuids"
}

_orangu_server() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    if [[ "$prev" == "show" || "$prev" == "delete" ]]; then
        COMPREPLY=( $(compgen -W "$(_orangu_server_models)" -- "$cur") )
        return 0
    fi

    if [[ "$prev" == "prune" ]]; then
        COMPREPLY=( $(compgen -W "$(_orangu_server_sessions)" -- "$cur") )
        return 0
    fi

    case "$prev" in
        -c|--config)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "-c --config -i --init -s --shell-completions -d --daemon \
             --all --code --review --explorer --embedding -y --yes -h --help -V --version" -- "$cur") )
        return 0
    fi

    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "$(_orangu_server_models) system suggest list show download delete prune help" -- "$cur") )
        return 0
    fi
}

complete -F _orangu_server orangu-server
"#;

pub const ZSH: &str = r#"#compdef orangu-server
# zsh completion for orangu-server
#
# Quick setup — add to ~/.zshrc:
#   eval "$(orangu-server -s)"
#
# Or write once to your fpath directory:
#   orangu-server -s > ~/.zsh/completions/_orangu-server
#   # ~/.zshrc: fpath=(~/.zsh/completions $fpath) && autoload -Uz compinit && compinit

# Completes the positional MODEL argument (and `show`'s own argument) with
# every NR and MODEL from `orangu-server list`'s output.
_orangu_server_models() {
    local -a candidates
    candidates=( ${(f)"$(orangu-server list 2>/dev/null | awk 'NR>1 {print $1; print $2}')"} )
    compadd -a candidates
}

# Completes `prune`'s own argument: every session UUID under
# ~/.orangu/server/sessions, newest first, plus the literal "all".
_orangu_server_sessions() {
    local sessions_dir="${HOME}/.orangu/server/sessions"
    local -a uuids
    [[ -d $sessions_dir ]] && uuids=( $sessions_dir/*(/Nom:t) )
    compadd all
    compadd -a uuids
}

_orangu_server_commands() {
    _values 'command' \
        'system[Detect the machine'"'"'s CPU and GPU(s)]' \
        'suggest[Suggest a GGUF model size for this machine'"'"'s hardware]' \
        'list[List every .gguf file under the models directory]' \
        'show[Print a GGUF file'"'"'s full metadata]' \
        'download[Download a GGUF model from Hugging Face]' \
        'delete[Delete a GGUF model from disk]' \
        'prune[Delete chat sessions]' \
        'help[Print this message or the help of the given subcommand(s)]'
}

_orangu_server() {
    local curcontext="$curcontext" state line

    _arguments -C \
        '(-c --config)'{-c,--config}'[Path to the configuration file (orangu-server.conf)]:config file:_files' \
        '(-i --init)'{-i,--init}'[Interactively create ~/.orangu/orangu-server.conf and exit]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-d --daemon)'{-d,--daemon}'[Run in the background, detached from the terminal]' \
        '(--all --code --review --explorer --embedding)--all[General-purpose role (default)]' \
        '(--all --code --review --explorer --embedding)--code[Coding role]' \
        '(--all --code --review --explorer --embedding)--review[Code review role]' \
        '(--all --code --review --explorer --embedding)--explorer[Exploration role]' \
        '(--all --code --review --explorer --embedding)--embedding[Embeddings-only role]' \
        '(-y --yes)'{-y,--yes}'[Skip delete'"'"'s confirmation prompt]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]' \
        '1: :->command' \
        '2: :->arg' \
        && return 0

    case $state in
        command)
            _alternative \
                'models:model:_orangu_server_models' \
                'commands:command:_orangu_server_commands'
            ;;
        arg)
            [[ ${line[1]} == show || ${line[1]} == delete ]] && _orangu_server_models
            [[ ${line[1]} == prune ]] && _orangu_server_sessions
            ;;
    esac
}

_orangu_server "$@"
"#;

pub const FISH: &str = r#"# fish completion for orangu-server
#
# Quick setup — add to ~/.config/fish/config.fish:
#   orangu-server -s | source
#
# Or write once to the fish completions directory:
#   orangu-server -s > ~/.config/fish/completions/orangu-server.fish

# Completes the positional MODEL argument (and `show`'s own argument) with
# every NR and MODEL from `orangu-server list`'s output.
function __orangu_server_models
    orangu-server list 2>/dev/null | awk 'NR>1 {print $1; print $2}'
end

# Completes `prune`'s own argument: every session UUID under
# ~/.orangu/server/sessions, newest first, plus the literal "all".
function __orangu_server_sessions
    echo all
    set -l sessions_dir "$HOME/.orangu/server/sessions"
    test -d "$sessions_dir"; or return
    path basename (path sort --reverse --key=mtime $sessions_dir/*/)
end

complete -c orangu-server -n '__fish_use_subcommand' -a '(__orangu_server_models)'
complete -c orangu-server -n '__fish_use_subcommand' -a system   -d 'Detect the machine\'s CPU and GPU(s)'
complete -c orangu-server -n '__fish_use_subcommand' -a suggest  -d 'Suggest a GGUF model size for this machine\'s hardware'
complete -c orangu-server -n '__fish_use_subcommand' -a list     -d 'List every .gguf file under the models directory'
complete -c orangu-server -n '__fish_use_subcommand' -a show     -d 'Print a GGUF file\'s full metadata'
complete -c orangu-server -n '__fish_use_subcommand' -a download -d 'Download a GGUF model from Hugging Face'
complete -c orangu-server -n '__fish_use_subcommand' -a delete   -d 'Delete a GGUF model from disk'
complete -c orangu-server -n '__fish_use_subcommand' -a prune    -d 'Delete chat sessions'
complete -c orangu-server -n '__fish_use_subcommand' -a help     -d 'Print this message or the help of the given subcommand(s)'
complete -c orangu-server -n '__fish_seen_subcommand_from show' -a '(__orangu_server_models)'
complete -c orangu-server -n '__fish_seen_subcommand_from delete' -a '(__orangu_server_models)'
complete -c orangu-server -n '__fish_seen_subcommand_from prune' -a '(__orangu_server_sessions)'

complete -c orangu-server -s c -l config              -r -d 'Path to the configuration file (orangu-server.conf)'
complete -c orangu-server -s i -l init                    -d 'Interactively create ~/.orangu/orangu-server.conf and exit'
complete -c orangu-server -s s -l shell-completions       -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-server -s d -l daemon                  -d 'Run in the background, detached from the terminal'
complete -c orangu-server      -l all                     -d 'General-purpose role (default)'
complete -c orangu-server      -l code                    -d 'Coding role'
complete -c orangu-server      -l review                  -d 'Code review role'
complete -c orangu-server      -l explorer                -d 'Exploration role'
complete -c orangu-server      -l embedding               -d 'Embeddings-only role'
complete -c orangu-server -s y -l yes                     -d 'Skip delete\'s confirmation prompt'
complete -c orangu-server -s h -l help                    -d 'Print help'
complete -c orangu-server -s V -l version                 -d 'Print version'
"#;
