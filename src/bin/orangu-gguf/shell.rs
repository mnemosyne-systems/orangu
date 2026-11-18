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

//! Hand-written shell completion scripts, mirroring `orangu`'s and
//! `orangu-server`'s own `-s`/`--shell-completions`. The positional
//! `MANIFEST` completes `.json` files, `-m`/`--model` `.gguf` files, and
//! `-q`/`--quantization` the weight formats — those by shelling back out to
//! `orangu-gguf --list-quantizations` itself and reading its first column,
//! the same trick `orangu-server`'s scripts use for models, so the list
//! never has to be repeated here. `-ts` and `-cs` — the two-letter short
//! options `main.rs`'s `normalize` rewrites — are offered alongside their
//! long forms. No clap-generated completion machinery is involved. The
//! PowerShell script is kept to ASCII: Windows PowerShell decodes a native
//! command's output in the console's code page, which would garble a
//! tooltip's dash.
//!
//! `every_flag_is_offered_by_every_completion_script` (`main.rs`) is what
//! keeps these three scripts in step with the parser.

pub const BASH: &str = r#"# bash completion for orangu-gguf
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-gguf -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-gguf -s > ~/.local/share/bash-completion/completions/orangu-gguf

# Completes -q/--quantization with every weight format from
# `orangu-gguf --list-quantizations`'s output.
_orangu_gguf_quantizations() {
    orangu-gguf --list-quantizations 2>/dev/null | awk '{print $1}'
}

_orangu_gguf() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    case "$prev" in
        -m|--model)
            COMPREPLY=( $(compgen -f -X '!*.gguf' -- "$cur") $(compgen -d -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        -o|--output|--imatrix|--calibration|--imatrix-out|--flamegraph|--dump|--hash|--compare|--merge|--config|--perplexity)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        -q|--quantization)
            COMPREPLY=( $(compgen -W "$(_orangu_gguf_quantizations)" -- "$cur") )
            return 0
            ;;
        --flamegraph-call-graph)
            COMPREPLY=( $(compgen -W "fp dwarf" -- "$cur") )
            return 0
            ;;
        -ts|--training-size|-cs|--context-size|--flamegraph-freq|--split-max-size|--split-max-tensors|--set|--delete|--override-kv)
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "-m --model -ts --training-size -q --quantization -cs --context-size -o --output --imatrix --calibration --imatrix-out --list-quantizations --max-size --max-kl --tensor-type \
             --dump --hash --compare --split-max-size --split-max-tensors --keep-split --merge --set --delete --override-kv --pure --allow-requantize --worker --channel --config --perplexity --flamegraph --flamegraph-freq --flamegraph-call-graph --flamegraph-png -s --shell-completions -h --help -V --version" -- "$cur") )
        return 0
    fi

    COMPREPLY=( $(compgen -f -X '!*.json' -- "$cur") $(compgen -d -- "$cur") )
    compopt -o filenames 2>/dev/null
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

# Completes -q/--quantization with every weight format from
# `orangu-gguf --list-quantizations`'s output.
_orangu_gguf_quantizations() {
    local -a candidates
    candidates=( ${(f)"$(orangu-gguf --list-quantizations 2>/dev/null | awk '{print $1}')"} )
    compadd -a candidates
}

_orangu_gguf() {
    _arguments \
        '(-m --model)'{-m,--model}'[Convert this GGUF file, checkpoint directory or HF repo instead of training]:model:_files -g "*.gguf"' \
        '(-ts --training-size)'{-ts,--training-size}'[Training size for this run, overriding the manifest]:size:' \
        '(-q --quantization)'{-q,--quantization}'[Weight format written, overriding the manifest]:quantization:_orangu_gguf_quantizations' \
        '(-cs --context-size)'{-cs,--context-size}'[Context length the model declares, overriding the manifest]:n:' \
        '(-o --output)'{-o,--output}'[Where the model is written, overriding the manifest]:output:_files' \
        '--imatrix[Fit the quantization to this importance matrix file or HF repo]:imatrix:_files' \
        '--calibration[Measure an importance matrix over this text and fit the quantization to it]:calibration:_files' \
        '--imatrix-out[With --calibration, write the measured importance matrix here and stop]:imatrix-out:_files' \
        '--list-quantizations[List the weight formats a manifest'"'"'s quantization accepts]' \
        '--max-size[With -q auto, the most bytes of weights the file may take]:size:' \
        '--max-kl[With -q auto, the most KL from the source over the calibration text]:nats:' \
        '*--tensor-type[Write tensors whose name matches PATTERN as TYPE (repeatable)]:pattern=type:' \
        '--dump[Print a model file'"'"'s metadata and tensor table]:model:_files -g "*.gguf"' \
        '--hash[Print a content hash of every tensor in a model file]:model:_files -g "*.gguf"' \
        '--compare[Compare two model files'"'"' metadata and tensors]:a:_files -g "*.gguf":b:_files -g "*.gguf"' \
        '--split-max-size[Write the model as shards of at most SIZE each]:size:' \
        '--split-max-tensors[Write the model as shards of at most N tensors each]:n:' \
        '--keep-split[Write a shard set as the source set is cut]' \
        '--merge[Join a shard set, named by its first shard, into one file]:model:_files -g "*.gguf"' \
        '*--set[Set a metadata key, in place when no format is given (repeatable)]:key=type\:value:' \
        '*--delete[Delete a metadata key, in place when no format is given (repeatable)]:key:' \
        '*--override-kv[Write this metadata key into the output (repeatable)]:key=type\:value:' \
        '--pure[Quantize every tensor that may be to the format'"'"'s base type]' \
        '--allow-requantize[Accept a quantized source, rounding its tensors a second time]' \
        '--worker[Join the \[workers\] channel as a node that takes a share of building a model]' \
        '--channel[List the nodes of the \[workers\] channel, servers and gguf nodes]' \
        '--config[Read \[workers\] from this file instead of orangu-gguf.conf]:file:_files' \
        '--perplexity[Print the perplexity of the -m model over this text]:text:_files' \
        '--flamegraph[Record a CPU flamegraph of the run and render it here]:path:_files' \
        '--flamegraph-freq[Sampling frequency in Hz for --flamegraph]:hz:' \
        '--flamegraph-call-graph[Call-graph mode for --flamegraph]:mode:(fp dwarf)' \
        '--flamegraph-png[Also render a PNG beside the flamegraph SVG]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]' \
        '1:manifest:_files -g "*.json"'
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

# Completes -q/--quantization with every weight format from
# `orangu-gguf --list-quantizations`'s output.
function __orangu_gguf_quantizations
    orangu-gguf --list-quantizations 2>/dev/null | awk '{print $1}'
end

# The positional MANIFEST: a JSON file.
complete -c orangu-gguf -n '__fish_is_first_arg' -k -a '(__fish_complete_suffix .json)'
complete -c orangu-gguf -s m -l model              -r -k -a '(__fish_complete_suffix .gguf)' -d 'Convert this GGUF file, checkpoint directory or HF repo instead of training'
complete -c orangu-gguf -o ts -l training-size     -x -d 'Training size for this run, overriding the manifest'
complete -c orangu-gguf -s q -l quantization       -x -a '(__orangu_gguf_quantizations)' -d 'Weight format written, overriding the manifest'
complete -c orangu-gguf -o cs -l context-size      -x -d 'Context length the model declares, overriding the manifest'
complete -c orangu-gguf -s o -l output             -r -d 'Where the model is written, overriding the manifest'
complete -c orangu-gguf      -l imatrix            -r -d 'Fit the quantization to this importance matrix file or HF repo'
complete -c orangu-gguf      -l calibration        -r -d 'Measure an importance matrix over this text and fit the quantization to it'
complete -c orangu-gguf      -l imatrix-out        -r -d 'With --calibration, write the measured importance matrix here and stop'
complete -c orangu-gguf      -l list-quantizations    -d 'List the weight formats a manifest\'s quantization accepts'
complete -c orangu-gguf      -l max-size           -r -d 'With -q auto, the most bytes of weights the file may take'
complete -c orangu-gguf      -l max-kl             -r -d 'With -q auto, the most KL from the source over the calibration text'
complete -c orangu-gguf      -l tensor-type        -r -d 'Write tensors whose name matches PATTERN as TYPE (repeatable)'
complete -c orangu-gguf      -l dump               -r -k -a '(__fish_complete_suffix .gguf)' -d 'Print a model file\'s metadata and tensor table'
complete -c orangu-gguf      -l hash               -r -k -a '(__fish_complete_suffix .gguf)' -d 'Print a content hash of every tensor in a model file'
complete -c orangu-gguf      -l compare            -r -k -a '(__fish_complete_suffix .gguf)' -d 'Compare two model files\' metadata and tensors'
complete -c orangu-gguf      -l split-max-size     -r -d 'Write the model as shards of at most SIZE each'
complete -c orangu-gguf      -l split-max-tensors  -r -d 'Write the model as shards of at most N tensors each'
complete -c orangu-gguf      -l keep-split            -d 'Write a shard set as the source set is cut'
complete -c orangu-gguf      -l merge              -r -k -a '(__fish_complete_suffix .gguf)' -d 'Join a shard set, named by its first shard, into one file'
complete -c orangu-gguf      -l set                -r -d 'Set a metadata key, in place when no format is given (repeatable)'
complete -c orangu-gguf      -l delete             -r -d 'Delete a metadata key, in place when no format is given (repeatable)'
complete -c orangu-gguf      -l override-kv        -r -d 'Write this metadata key into the output (repeatable)'
complete -c orangu-gguf      -l pure                  -d 'Quantize every tensor that may be to the format\'s base type'
complete -c orangu-gguf      -l allow-requantize      -d 'Accept a quantized source, rounding its tensors a second time'
complete -c orangu-gguf      -l worker                -d 'Join the [workers] channel as a node that takes a share of building a model'
complete -c orangu-gguf      -l channel               -d 'List the nodes of the [workers] channel, servers and gguf nodes'
complete -c orangu-gguf      -l config             -r -d 'Read [workers] from this file instead of orangu-gguf.conf'
complete -c orangu-gguf      -l perplexity         -r -d 'Print the perplexity of the -m model over this text'
complete -c orangu-gguf      -l flamegraph         -r -d 'Record a CPU flamegraph of the run and render it here'
complete -c orangu-gguf      -l flamegraph-freq    -x -d 'Sampling frequency in Hz for --flamegraph'
complete -c orangu-gguf      -l flamegraph-call-graph -x -a 'fp dwarf' -d 'Call-graph mode for --flamegraph'
complete -c orangu-gguf      -l flamegraph-png        -d 'Also render a PNG beside the flamegraph SVG'
complete -c orangu-gguf -s s -l shell-completions     -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-gguf -s h -l help                  -d 'Print help'
complete -c orangu-gguf -s V -l version               -d 'Print version'
"#;

pub const POWERSHELL: &str = r#"# PowerShell completion for orangu-gguf
#
# Quick setup - add to your $PROFILE (notepad $PROFILE):
#   orangu-gguf -s | Out-String | Invoke-Expression
#
# Or write once to a file and dot-source it from $PROFILE:
#   orangu-gguf -s > "$HOME\orangu-gguf.ps1"
#   # $PROFILE: . "$HOME\orangu-gguf.ps1"

Register-ArgumentCompleter -Native -CommandName 'orangu-gguf' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    # Every word typed before the one under the cursor; $words[0] is the
    # command itself, so $prev is that when nothing else has been typed.
    $words = @($commandAst.CommandElements | ForEach-Object { $_.Extent.Text })
    if ($wordToComplete -ne '' -and $words.Count -gt 1) {
        $words = $words[0..($words.Count - 2)]
    }
    $prev = $words[-1]

    $flags = @(
        @('-m', '--model', 'Convert this GGUF file, checkpoint directory or HF repo instead of training'),
        @('-ts', '--training-size', 'Training size for this run, overriding the manifest'),
        @('-q', '--quantization', 'Weight format written, overriding the manifest'),
        @('-cs', '--context-size', 'Context length the model declares, overriding the manifest'),
        @('-o', '--output', 'Where the model is written, overriding the manifest'),
        @('--imatrix', 'Fit the quantization to this importance matrix file or HF repo'),
        @('--calibration', 'Measure an importance matrix over this text and fit the quantization to it'),
        @('--imatrix-out', 'With --calibration, write the measured importance matrix here and stop'),
        @('--list-quantizations', 'List the weight formats a manifest''s quantization accepts'),
        @('--max-size', 'With -q auto, the most bytes of weights the file may take'),
        @('--max-kl', 'With -q auto, the most KL from the source over the calibration text'),
        @('--tensor-type', 'Write tensors whose name matches PATTERN as TYPE (repeatable)'),
        @('--dump', 'Print a model file''s metadata and tensor table'),
        @('--hash', 'Print a content hash of every tensor in a model file'),
        @('--compare', 'Compare two model files'' metadata and tensors'),
        @('--split-max-size', 'Write the model as shards of at most SIZE each'),
        @('--split-max-tensors', 'Write the model as shards of at most N tensors each'),
        @('--keep-split', 'Write a shard set as the source set is cut'),
        @('--merge', 'Join a shard set, named by its first shard, into one file'),
        @('--set', 'Set a metadata key, in place when no format is given (repeatable)'),
        @('--delete', 'Delete a metadata key, in place when no format is given (repeatable)'),
        @('--override-kv', 'Write this metadata key into the output (repeatable)'),
        @('--pure', 'Quantize every tensor that may be to the format''s base type'),
        @('--allow-requantize', 'Accept a quantized source, rounding its tensors a second time'),
        @('--worker', 'Join the [workers] channel as a node that takes a share of building a model'),
        @('--channel', 'List the nodes of the [workers] channel, servers and gguf nodes'),
        @('--config', 'Read [workers] from this file instead of orangu-gguf.conf'),
        @('--perplexity', 'Print the perplexity of the -m model over this text'),
        @('--flamegraph', 'Record a CPU flamegraph of the run and render it here'),
        @('--flamegraph-freq', 'Sampling frequency in Hz for --flamegraph'),
        @('--flamegraph-call-graph', 'Call-graph mode for --flamegraph: fp or dwarf'),
        @('--flamegraph-png', 'Also render a PNG beside the flamegraph SVG'),
        @('-s', '--shell-completions', 'Print shell completion script for the detected shell and exit'),
        @('-h', '--help', 'Print help'),
        @('-V', '--version', 'Print version')
    )

    function Offer([string[]]$candidates) {
        foreach ($candidate in $candidates) {
            if ($candidate -like "$wordToComplete*") {
                [System.Management.Automation.CompletionResult]::new($candidate, $candidate, 'ParameterValue', $candidate)
            }
        }
    }

    # -q/--quantization: every weight format from `orangu-gguf --list-quantizations`.
    function Quantizations {
        & orangu-gguf --list-quantizations 2>$null | ForEach-Object { (-split $_)[0] }
    }

    switch ($prev) {
        # A path (the manifest, a model, an output): PowerShell's own completion takes over.
        { $_ -in '-m', '--model', '-o', '--output', '--imatrix', '--calibration', '--imatrix-out', '--flamegraph', '--dump', '--hash', '--compare', '--merge', '--config', '--perplexity' } { return }
        { $_ -in '-q', '--quantization' } { return Offer (Quantizations) }
        '--flamegraph-call-graph' { return Offer @('fp', 'dwarf') }
        { $_ -in '-ts', '--training-size', '-cs', '--context-size', '--flamegraph-freq' } { return }
    }

    if ($wordToComplete.StartsWith('-')) {
        foreach ($flag in $flags) {
            $tooltip = $flag[-1]
            foreach ($name in $flag[0..($flag.Count - 2)]) {
                if ($name -like "$wordToComplete*") {
                    [System.Management.Automation.CompletionResult]::new($name, $name, 'ParameterName', $tooltip)
                }
            }
        }
    }
}
"#;
