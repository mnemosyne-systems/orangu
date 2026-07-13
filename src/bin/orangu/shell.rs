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

    case "$prev" in
        -c|--config)
            # Configuration file (orangu.conf)
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
        --theme)
            COMPREPLY=( $(compgen -W "classic oranguday tokyonight rosepine-moon auto" -- "$cur") $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "-c --config --theme -w --workspace -r --resume -a --all -l --list -i --init -s --shell-completions -h --help" -- "$cur") )
        return 0
    fi
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
    themes=(classic oranguday tokyonight rosepine-moon auto)
    compadd -a themes
    _files -g '*.theme'
}

_orangu() {
    _arguments -s \
        '(-c --config)'{-c,--config}'[Path to the configuration file (orangu.conf)]:config file:_files' \
        '(--theme)'--theme'[Override the TUI theme with a name or .theme file]:theme:_orangu_themes' \
        '(-w --workspace)'{-w,--workspace}'[Workspace root for local tools]:workspace:_orangu_workspaces' \
        '(-r --resume)'{-r,--resume}'[Resume a session by UUID]:session uuid:_orangu_sessions' \
        '(-a --all)'{-a,--all}'[Reopen the workspace tabs from the previous run]' \
        '(-l --list)'{-l,--list}'[List all stored sessions as a table and exit]' \
        '(-i --init)'{-i,--init}'[Interactively create ~/.orangu/orangu.conf and exit]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]'
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

complete -c orangu -s c -l config           -r                          -d 'Path to the configuration file (orangu.conf)'
complete -c orangu    -l theme              -r -a 'classic oranguday tokyonight rosepine-moon auto' -d 'Override the TUI theme with a name or .theme file'
complete -c orangu    -l theme              -r -a '(__fish_complete_path)' -d 'Theme file'
complete -c orangu -s w -l workspace         -x -a '(__orangu_workspaces)' -d 'Workspace root for local tools'
complete -c orangu -s r -l resume            -x -a '(__orangu_sessions)'   -d 'Resume a session by UUID'
complete -c orangu -s a -l all                                            -d 'Reopen the workspace tabs from the previous run'
complete -c orangu -s l -l list                                           -d 'List all stored sessions as a table and exit'
complete -c orangu -s i -l init                                           -d 'Interactively create ~/.orangu/orangu.conf and exit'
complete -c orangu -s s -l shell-completions                              -d 'Print shell completion script for the detected shell and exit'
complete -c orangu -s h -l help                                           -d 'Print help'
"#;
