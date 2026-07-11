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

//! Hand-written shell completion scripts, mirroring `orangu-gguf`'s own
//! `-s`/`--shell-completions` (`src/bin/orangu-gguf/shell.rs`): the
//! positional `model` argument is completed by shelling out to
//! `orangu-gguf list` and reading its first two columns (NR and MODEL) —
//! the same trick `orangu-gguf`'s own `show`/`download` completion uses,
//! reused here since both tools default to the same models directory
//! convention (`~/.cache/huggingface/hub`, see `init.rs`'s
//! `huggingface_cache_dir`). This only ever depends on `orangu-gguf` and
//! `orangu-server` both being on `$PATH` — no clap-generated completion
//! machinery is involved.

pub const BASH: &str = r#"# bash completion for orangu-server
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-server -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-server -s > ~/.local/share/bash-completion/completions/orangu-server

# Completes the positional MODEL argument with every NR and MODEL from
# `orangu-gguf list`'s output.
_orangu_server_models() {
    orangu-gguf list 2>/dev/null | awk 'NR>1 {print $1; print $2}'
}

_orangu_server() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

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
             --all --code --review --explorer --embedding -h --help -V --version" -- "$cur") )
        return 0
    fi

    COMPREPLY=( $(compgen -W "$(_orangu_server_models)" -- "$cur") )
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

# Completes the positional MODEL argument with every NR and MODEL from
# `orangu-gguf list`'s output.
_orangu_server_models() {
    local -a candidates
    candidates=( ${(f)"$(orangu-gguf list 2>/dev/null | awk 'NR>1 {print $1; print $2}')"} )
    compadd -a candidates
}

_orangu_server() {
    _arguments -s \
        '(-c --config)'{-c,--config}'[Path to the configuration file (orangu-server.conf)]:config file:_files' \
        '(-i --init)'{-i,--init}'[Interactively create ~/.orangu/orangu-server.conf and exit]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-d --daemon)'{-d,--daemon}'[Run in the background, detached from the terminal]' \
        '(--all --code --review --explorer --embedding)--all[General-purpose role (default)]' \
        '(--all --code --review --explorer --embedding)--code[Coding role]' \
        '(--all --code --review --explorer --embedding)--review[Code review role]' \
        '(--all --code --review --explorer --embedding)--explorer[Exploration role]' \
        '(--all --code --review --explorer --embedding)--embedding[Embeddings-only role]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]' \
        '1: :_orangu_server_models' \
        && return 0
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

# Completes the positional MODEL argument with every NR and MODEL from
# `orangu-gguf list`'s output.
function __orangu_server_models
    orangu-gguf list 2>/dev/null | awk 'NR>1 {print $1; print $2}'
end

complete -c orangu-server -n '__fish_use_subcommand' -a '(__orangu_server_models)'

complete -c orangu-server -s c -l config              -r -d 'Path to the configuration file (orangu-server.conf)'
complete -c orangu-server -s i -l init                    -d 'Interactively create ~/.orangu/orangu-server.conf and exit'
complete -c orangu-server -s s -l shell-completions       -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-server -s d -l daemon                  -d 'Run in the background, detached from the terminal'
complete -c orangu-server      -l all                     -d 'General-purpose role (default)'
complete -c orangu-server      -l code                    -d 'Coding role'
complete -c orangu-server      -l review                  -d 'Code review role'
complete -c orangu-server      -l explorer                -d 'Exploration role'
complete -c orangu-server      -l embedding               -d 'Embeddings-only role'
complete -c orangu-server -s h -l help                    -d 'Print help'
complete -c orangu-server -s V -l version                 -d 'Print version'
"#;
