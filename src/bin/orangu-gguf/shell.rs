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
//! `-s`/`--shell-completions` (`src/bin/orangu/shell.rs`): `show`'s NR/MODEL
//! argument is completed by shelling back out to `orangu-gguf list` itself
//! and reading its first two columns, the same way `orangu`'s own scripts
//! complete session UUIDs by reading `~/.orangu/sessions` directly. This
//! only ever depends on `orangu-gguf` itself being on `$PATH` — no
//! clap-generated completion machinery is involved.

pub const BASH: &str = r#"# bash completion for orangu-gguf
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-gguf -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-gguf -s > ~/.local/share/bash-completion/completions/orangu-gguf

# Completes `show`'s argument with every NR and MODEL from `list`'s output.
_orangu_gguf_models() {
    orangu-gguf list 2>/dev/null | awk 'NR>1 {print $1; print $2}'
}

_orangu_gguf() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    if [[ "$prev" == "show" ]]; then
        COMPREPLY=( $(compgen -W "$(_orangu_gguf_models)" -- "$cur") )
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
            "-c --config -i --init -s --shell-completions -h --help -V --version" -- "$cur") )
        return 0
    fi

    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "system suggest list show download help" -- "$cur") )
        return 0
    fi
}

complete -F _orangu_gguf orangu-gguf
"#;

pub const ZSH: &str = r#"#compdef orangu-gguf
# zsh completion for orangu-gguf
#
# Quick setup — add to ~/.zshrc:
#   eval "$(orangu-gguf -s)"
#
# Or write once to your fpath directory:
#   orangu-gguf -s > ~/.zsh/completions/_orangu-gguf
#   # ~/.zshrc: fpath=(~/.zsh/completions $fpath) && autoload -Uz compinit && compinit

# Completes `show`'s argument with every NR and MODEL from `list`'s output.
_orangu_gguf_models() {
    local -a candidates
    candidates=( ${(f)"$(orangu-gguf list 2>/dev/null | awk 'NR>1 {print $1; print $2}')"} )
    compadd -a candidates
}

_orangu_gguf() {
    local curcontext="$curcontext" state line

    _arguments -C \
        '(-c --config)'{-c,--config}'[Path to the configuration file (orangu-gguf.conf)]:config file:_files' \
        '(-i --init)'{-i,--init}'[Interactively create ~/.orangu/orangu-gguf.conf and exit]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]' \
        '1: :->command' \
        '2: :->arg' \
        && return 0

    case $state in
        command)
            _values 'command' \
                'system[Detect the machine'"'"'s CPU and GPU(s)]' \
                'suggest[Suggest a GGUF model size for this machine'"'"'s hardware]' \
                'list[List every .gguf file under the models directory]' \
                'show[Print a GGUF file'"'"'s full metadata]' \
                'download[Download a GGUF model from Hugging Face]' \
                'help[Print this message or the help of the given subcommand(s)]'
            ;;
        arg)
            [[ ${line[1]} == show ]] && _orangu_gguf_models
            ;;
    esac
}

_orangu_gguf "$@"
"#;

pub const FISH: &str = r#"# fish completion for orangu-gguf
#
# Quick setup — add to ~/.config/fish/config.fish:
#   orangu-gguf -s | source
#
# Or write once to the fish completions directory:
#   orangu-gguf -s > ~/.config/fish/completions/orangu-gguf.fish

# Completes `show`'s argument with every NR and MODEL from `list`'s output.
function __orangu_gguf_models
    orangu-gguf list 2>/dev/null | awk 'NR>1 {print $1; print $2}'
end

complete -c orangu-gguf -n '__fish_use_subcommand' -a system   -d 'Detect the machine\'s CPU and GPU(s)'
complete -c orangu-gguf -n '__fish_use_subcommand' -a suggest  -d 'Suggest a GGUF model size for this machine\'s hardware'
complete -c orangu-gguf -n '__fish_use_subcommand' -a list     -d 'List every .gguf file under the models directory'
complete -c orangu-gguf -n '__fish_use_subcommand' -a show     -d 'Print a GGUF file\'s full metadata'
complete -c orangu-gguf -n '__fish_use_subcommand' -a download -d 'Download a GGUF model from Hugging Face'
complete -c orangu-gguf -n '__fish_use_subcommand' -a help     -d 'Print this message or the help of the given subcommand(s)'
complete -c orangu-gguf -n '__fish_seen_subcommand_from show' -a '(__orangu_gguf_models)'

complete -c orangu-gguf -s c -l config              -r -d 'Path to the configuration file (orangu-gguf.conf)'
complete -c orangu-gguf -s i -l init                    -d 'Interactively create ~/.orangu/orangu-gguf.conf and exit'
complete -c orangu-gguf -s s -l shell-completions       -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-gguf -s h -l help                    -d 'Print help'
complete -c orangu-gguf -s V -l version                 -d 'Print version'
"#;
