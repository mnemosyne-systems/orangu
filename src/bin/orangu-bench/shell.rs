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
//! `orangu-server`'s own `-s`/`--shell-completions`. The bench takes no
//! positional argument and no subcommand — it is all flags — so the scripts
//! offer those, and complete the values that have a finite or filesystem
//! answer: the path-taking flags (`--history`, `--chart`, `--flamegraph`,
//! `--bundle`, `--report`, ...) complete files, `--flamegraph-call-graph`
//! its three modes, and `--host` the usual bind addresses. The rest — token
//! counts, URLs, a model id — are typed. No clap-generated completion
//! machinery is involved. The PowerShell script is kept to ASCII: Windows
//! PowerShell decodes a native command's output in the console's code
//! page, which would garble a tooltip's dash.
//!
//! `every_flag_is_offered_by_every_completion_script` (`main.rs`) is what
//! keeps these three scripts in step with the parser.

pub const BASH: &str = r#"# bash completion for orangu-bench
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-bench -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-bench -s > ~/.local/share/bash-completion/completions/orangu-bench

_orangu_bench() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    case "$prev" in
        --history|--chart|--table|--storage-file|--download-dir|--flamegraph|--image-init|--compare-profiles|--bundle|--read-bundle|--render-profile|--report)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        --flamegraph-call-graph)
            COMPREPLY=( $(compgen -W "auto fp dwarf" -- "$cur") )
            return 0
            ;;
        --host)
            COMPREPLY=( $(compgen -W "all 0.0.0.0 127.0.0.1" -- "$cur") )
            return 0
            ;;
        --url|--depths|--pp|--pp-continue|--pg|--streams|--streams-depth|--interference|--chat|--chat-fixture|--kl|--kl-rows|--kl-top|--kl-tokens|--kl-text|--perplexity|--ppl-ctx|--ppl-chunks|--hellaswag|--winogrande|--score-tasks|--score-rows|--shared-prefix|--shared-prefix-tokens|--prefix-scan|--pp-continue-base|--embed|--image|--image-steps|--image-cfg|--image-prompt|--gen|--curve|--bucket|--reps|--timeout|--model|--label|--storage-probe|--storage-span|--storage-ramp|--download-probe|--download-file|--download-seconds|--cap|--chart-scale|--chart-y-label|--chart-x-label|--chart-panels|--flamegraph-pid|--flamegraph-freq|--flamegraph-layers|--flamegraph-duration|--sweep|--sweep-cmd|--sweep-env|--sweep-start-timeout|--port|--delay|--temperature|--workers)
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "--url --depths --pp --pp-continue --pg --decode-cpu --streams --streams-depth --streams-echo --streams-warm --shared-prefix-cold --interference --chat --chat-fixture --kl --kl-rows --kl-top --kl-tokens --kl-text --perplexity --ppl-ctx --ppl-chunks --hellaswag --winogrande --score-tasks --score-rows --shared-prefix --shared-prefix-tokens \
             --prefix-scan --pp-continue-base --embed --image --image-steps --image-cfg --image-prompt --image-init --gen --curve --bucket --reps --drop-model-cache --no-warmup --fail-on-regression --sweep-as-given --write-profile --per-rep \
             --timeout --model --json --history --label --chart --chart-only --table --storage-probe --storage-file \
             --storage-span --storage-ramp --download-probe --download-file --download-seconds --download-dir --cap --chart-png --chart-scale --chart-y-label --chart-x-label --chart-panels --chart-grid --chart-overlay \
             --flamegraph --flamegraph-pid --flamegraph-freq --flamegraph-call-graph --flamegraph-png --flamegraph-keep-data \
             --flamegraph-layers --flamegraph-duration --flamegraph-watch \
             --compare-profiles --bundle --read-bundle --sweep --sweep-cmd --sweep-env --sweep-start-timeout \
             --render-profile --report --web --host --port --delay --temperature --workers -s --shell-completions -h --help -V --version" -- "$cur") )
        return 0
    fi
}

complete -F _orangu_bench orangu-bench
"#;

pub const ZSH: &str = r#"#compdef orangu-bench
# zsh completion for orangu-bench
#
# Quick setup — add to ~/.zshrc:
#   eval "$(orangu-bench -s)"
#
# Or write once to your fpath directory:
#   orangu-bench -s > ~/.zsh/completions/_orangu-bench
#   # ~/.zshrc: fpath=(~/.zsh/completions $fpath) && autoload -Uz compinit && compinit

_orangu_bench() {
    _arguments \
        '--url[Base URL of the server]:url:' \
        '--depths[Comma-separated context depths to sweep; ranges too: 0-2048+512]:list:' \
        '--pp[Prefill mode: prompt lengths to sweep, reporting prompt-processing rate]:list:' \
        '--pp-continue[Continuation-prefill mode: comma-separated added token counts to sweep]:list:' \
        '--pg[Combined mode: prompt lengths prefilled and generated in one request]:list:' \
        '--decode-cpu[Report the server'"'"'s CPU time per generated token, with prefill excluded]' \
        '--streams[Concurrency mode: comma-separated stream counts; reports AGGREGATE tok/s]:list:' \
        '--streams-depth[Concurrency mode: tokens of context in front of each stream'"'"'s instruction; 0 is none]:n:' \
        '--streams-echo[Concurrency mode: each stream copies a passage out of its prompt, which speculation drafts]' \
        '--streams-warm[Concurrency mode: prefill the prompts before each run, so all streams decode at once]' \
        '--interference[Interference mode: prompt lengths a second request prefills while one stream decodes]:list:' \
        '--chat[Chat mode: replay this many turns of a coding session through /v1/chat/completions]:turns:' \
        '--chat-fixture[The conversation --chat replays (JSON: system, tools, turns)]:file:_files' \
        '--kl[Compare teacher-forced distributions with this reference]:file:_files' \
        '--kl-rows[Positions per --kl pass]:n:' \
        '--kl-top[Candidates --kl keeps per position]:n:' \
        '--kl-tokens[Tokens --kl runs]:n:' \
        '--kl-text[The text --kl runs]:file:_files' \
        '--perplexity[Score this text file]:file:_files' \
        '--ppl-ctx[Tokens in each --perplexity window]:n:' \
        '--ppl-chunks[Windows --perplexity runs]:n:' \
        '--hellaswag[Score this HellaSwag file]:file:_files' \
        '--winogrande[Score this WinoGrande file]:file:_files' \
        '--score-tasks[Tasks the scoring modes run]:n:' \
        '--score-rows[Positions a pass for the scoring modes]:n:' \
        '--shared-prefix[Shared-prefix mode: stream counts that all send the SAME long prefix]:list:' \
        '--shared-prefix-tokens[Length in tokens of the prefix --shared-prefix streams have in common]:n:' \
        '--shared-prefix-cold[Shared-prefix mode: every round sends a prefix the server has not seen, so all streams arrive cold]' \
        '--prefix-scan[Scan-resistance mode: unique prompts to push through between two uses of one hot prefix]:list:' \
        '--pp-continue-base[Prompt length (tokens) to prime the prefix cache with for --pp-continue]:n:' \
        '--embed[Embedding mode: prompt lengths to sweep against /v1/embeddings]:list:' \
        '--image[Image mode: square picture sizes to sweep against /v1/images/generations]:list:' \
        '--image-steps[Denoising steps per picture for --image]:n:' \
        '--image-cfg[Guidance scale for --image; 1 runs the prompt alone]:scale:' \
        '--image-prompt[The prompt every --image picture is drawn from]:text:' \
        '--image-init[A picture attached to every --image request (an edit)]:file:_files' \
        '--gen[Number of tokens to generate per timed run]:n:' \
        '--curve[Curve mode: one generation of this many tokens, bucketed by context; 0 disables]:n:' \
        '--bucket[Bucket width (in context tokens) for --curve]:n:' \
        '--reps[Repetitions per depth; the reported rate is the best run with mean±sd]:n:' \
        '--drop-model-cache[Evict the model from the page cache before every repetition]' \
        '--no-warmup[Skip the initial warmup run]' \
        '--fail-on-regression[After a two-value --sweep, exit 2 if the second value is worse beyond the noise]' \
        '--sweep-as-given[Run a two-value --sweep in the order written instead of as a,b,b,a]' \
        '--write-profile[After a two-value --sweep, keep the winning value in the server tuning profile]' \
        '--per-rep[Also print every repetition'"'"'s rate, in order, after each row]' \
        '--timeout[Per-request timeout in seconds]:seconds:' \
        '--model[Model id to request]:id:' \
        '--json[Emit machine-readable JSON]' \
        '--history[Append each measured point to this tab-separated history file]:path:_files' \
        '--label[Series name recorded in the history file; prefixes each --sweep point]:name:' \
        '--chart[Render the history file to this SVG after measuring]:path:_files' \
        '--chart-only[Only render the chart and table from an existing history file; measure nothing]' \
        '--table[Write the history file as a markdown comparison table to this path (- for stdout)]:path:_files' \
        '--storage-probe[Storage mode: comma-separated read request sizes in KiB to sweep]:list:' \
        '--storage-file[File the storage probe reads (default: the server'"'"'s largest shard)]:path:_files' \
        '--storage-span[MiB to read at each request size, per pass]:mib:' \
        '--storage-ramp[MiB read and discarded before timing starts, at each size]:mib:' \
        '--download-probe[Download mode: comma-separated connection counts to sweep]:list:' \
        '--download-file[Hub file the download probe fetches, as <user>/<repo>:<path>]:spec:' \
        '--download-seconds[Seconds to download for at each connection count, per pass]:secs:' \
        '--download-dir[Directory the download probe writes into]:path:_files -/' \
        '--cap[Run each --sweep server under this memory cap (e.g. 4G)]:size:' \
        '--chart-png[Also render a PNG beside the chart SVG]' \
        '--chart-scale[Pin the chart'"'"'s tok/s axis to MIN:MAX so a pair of charts compare]:min\:max:' \
        '--chart-y-label[Label for the chart'"'"'s y-axis]:text:' \
        '--chart-x-label[Label for the chart'"'"'s x-axis]:text:' \
        '--chart-panels[Draw only these modes'"'"' panels (e.g. pp,tg); default: every mode in the file]:list:' \
        '--chart-grid[Draw one panel per label group and mode, on shared axes]' \
        '--chart-overlay[Draw every label group and mode on one panel: hue per group, shade per series]' \
        '--flamegraph[Record a CPU flamegraph of the server over the measured window]:path:_files' \
        '--flamegraph-pid[Process to profile (default: the server'"'"'s own, else the URL port'"'"'s owner)]:pid:' \
        '--flamegraph-freq[Sampling frequency in Hz for --flamegraph]:hz:' \
        '--flamegraph-call-graph[Call-graph mode for --flamegraph]:mode:(auto fp dwarf)' \
        '--flamegraph-png[Also render a PNG beside the flamegraph SVG]' \
        '--flamegraph-keep-data[Keep the raw perf.data beside the flamegraph instead of deleting it]' \
        '--flamegraph-layers[Profile every running orangu, orangu-coordinator and orangu-server while you drive the workload; one flamegraph per process in DIR]:dir:_files -/' \
        '--flamegraph-duration[Seconds to keep sampling under --flamegraph-layers or --flamegraph-watch]:seconds:' \
        '--flamegraph-watch[With --flamegraph, profile the server while something else drives it]' \
        '--compare-profiles[Compare already-collapsed .folded profiles side by side; measure nothing]:list:_files' \
        '--bundle[Write the whole run — measurements, configuration, host — to one JSON file]:path:_files' \
        '--read-bundle[Read bundles and report them side by side; measure nothing]:list:_files' \
        '--sweep[Sweep one tuning variable: VAR=v1,v2,... (; if a value has a comma); needs --sweep-cmd]:spec:' \
        '--sweep-cmd[Shell command that starts the server, run once per --sweep value]:cmd:' \
        '*--sweep-env[Environment held constant across every --sweep point (repeatable)]:k=v:' \
        '--sweep-start-timeout[Seconds to wait for a swept server to come up]:seconds:' \
        '--render-profile[Re-render an already-collapsed .folded profile to SVG; measure nothing]:path:_files' \
        '--report[Write the run — provenance, measurements, chart, flamegraph — to one PDF]:path:_files' \
        '--web[Serve the web console instead of measuring]' \
        '--host[Address the web console binds: all (or *) for every interface]:host:(all 0.0.0.0 127.0.0.1)' \
        '--port[Port the web console listens on]:port:' \
        '--delay[Seconds to wait between measured points, for a card that heats up]:seconds:' \
        '--workers[A [workers] tree: on, off, or compare both]:mode:(on off compare)' \
        '--temperature[Sampling temperature for the timed decode; 0 is greedy]:t:' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]'
}

_orangu_bench "$@"
"#;

pub const FISH: &str = r#"# fish completion for orangu-bench
#
# Quick setup — add to ~/.config/fish/config.fish:
#   orangu-bench -s | source
#
# Or write once to the fish completions directory:
#   orangu-bench -s > ~/.config/fish/completions/orangu-bench.fish

complete -c orangu-bench -f
complete -c orangu-bench -l url                    -x -d 'Base URL of the server'
complete -c orangu-bench -l depths                 -x -d 'Comma-separated context depths to sweep; ranges too: 0-2048+512'
complete -c orangu-bench -l pp                     -x -d 'Prefill mode: prompt lengths to sweep, reporting prompt-processing rate'
complete -c orangu-bench -l pp-continue            -x -d 'Continuation-prefill mode: comma-separated added token counts to sweep'
complete -c orangu-bench -l pg                     -x -d 'Combined mode: prompt lengths prefilled and generated in one request'
complete -c orangu-bench -l decode-cpu                -d 'Report the server\'s CPU time per generated token, with prefill excluded'
complete -c orangu-bench -l streams                -x -d 'Concurrency mode: comma-separated stream counts; reports AGGREGATE tok/s'
complete -c orangu-bench -l streams-depth          -x -d 'Concurrency mode: tokens of context in front of each stream\'s instruction; 0 is none'
complete -c orangu-bench -l streams-echo              -d 'Concurrency mode: each stream copies a passage out of its prompt, which speculation drafts'
complete -c orangu-bench -l streams-warm              -d 'Concurrency mode: prefill the prompts before each run, so all streams decode at once'
complete -c orangu-bench -l interference           -x -d 'Interference mode: prompt lengths a second request prefills while one stream decodes'
complete -c orangu-bench -l chat                   -x -d 'Chat mode: replay this many turns of a coding session through /v1/chat/completions'
complete -c orangu-bench -l chat-fixture           -r -d 'The conversation --chat replays (JSON: system, tools, turns)'
complete -c orangu-bench -l kl                     -r -d 'Compare teacher-forced distributions with this reference'
complete -c orangu-bench -l kl-rows                -r -d 'Positions per --kl pass'
complete -c orangu-bench -l kl-top                 -r -d 'Candidates --kl keeps per position'
complete -c orangu-bench -l kl-tokens              -r -d 'Tokens --kl runs'
complete -c orangu-bench -l kl-text                -r -d 'The text --kl runs'
complete -c orangu-bench -l perplexity             -r -d 'Score this text file'
complete -c orangu-bench -l ppl-ctx                -r -d 'Tokens in each --perplexity window'
complete -c orangu-bench -l ppl-chunks             -r -d 'Windows --perplexity runs'
complete -c orangu-bench -l hellaswag              -r -d 'Score this HellaSwag file'
complete -c orangu-bench -l winogrande             -r -d 'Score this WinoGrande file'
complete -c orangu-bench -l score-tasks            -r -d 'Tasks the scoring modes run'
complete -c orangu-bench -l score-rows             -r -d 'Positions a pass for the scoring modes'
complete -c orangu-bench -l shared-prefix          -x -d 'Shared-prefix mode: stream counts that all send the SAME long prefix'
complete -c orangu-bench -l shared-prefix-tokens   -x -d 'Length in tokens of the prefix --shared-prefix streams have in common'
complete -c orangu-bench -l shared-prefix-cold        -d 'Shared-prefix mode: every round sends a prefix the server has not seen, so all streams arrive cold'
complete -c orangu-bench -l prefix-scan            -x -d 'Scan-resistance mode: unique prompts to push through between two uses of one hot prefix'
complete -c orangu-bench -l pp-continue-base       -x -d 'Prompt length (tokens) to prime the prefix cache with for --pp-continue'
complete -c orangu-bench -l embed                  -x -d 'Embedding mode: prompt lengths to sweep against /v1/embeddings'
complete -c orangu-bench -l image                  -x -d 'Image mode: square picture sizes to sweep against /v1/images/generations'
complete -c orangu-bench -l image-steps            -x -d 'Denoising steps per picture for --image'
complete -c orangu-bench -l image-cfg              -x -d 'Guidance scale for --image; 1 runs the prompt alone'
complete -c orangu-bench -l image-prompt           -x -d 'The prompt every --image picture is drawn from'
complete -c orangu-bench -l image-init             -r -F -d 'A picture attached to every --image request (an edit)'
complete -c orangu-bench -l gen                    -x -d 'Number of tokens to generate per timed run'
complete -c orangu-bench -l curve                  -x -d 'Curve mode: one generation of this many tokens, bucketed by context; 0 disables'
complete -c orangu-bench -l bucket                 -x -d 'Bucket width (in context tokens) for --curve'
complete -c orangu-bench -l reps                   -x -d 'Repetitions per depth; the reported rate is the best run with mean±sd'
complete -c orangu-bench -l drop-model-cache          -d 'Evict the model from the page cache before every repetition'
complete -c orangu-bench -l no-warmup                 -d 'Skip the initial warmup run'
complete -c orangu-bench -l fail-on-regression        -d 'After a two-value --sweep, exit 2 if the second value is worse beyond the noise'
complete -c orangu-bench -l sweep-as-given            -d 'Run a two-value --sweep in the order written instead of as a,b,b,a'
complete -c orangu-bench -l write-profile             -d 'After a two-value --sweep, keep the winning value in the server tuning profile'
complete -c orangu-bench -l per-rep                   -d 'Also print every repetition\'s rate, in order, after each row'
complete -c orangu-bench -l timeout                -x -d 'Per-request timeout in seconds'
complete -c orangu-bench -l model                  -x -d 'Model id to request'
complete -c orangu-bench -l json                      -d 'Emit machine-readable JSON'
complete -c orangu-bench -l history                -r -d 'Append each measured point to this tab-separated history file'
complete -c orangu-bench -l label                  -x -d 'Series name recorded in the history file; prefixes each --sweep point'
complete -c orangu-bench -l chart                  -r -d 'Render the history file to this SVG after measuring'
complete -c orangu-bench -l chart-only                -d 'Only render the chart and table from an existing history file; measure nothing'
complete -c orangu-bench -l table                  -r -d 'Write the history file as a markdown comparison table to this path (- for stdout)'
complete -c orangu-bench -l storage-probe          -x -d 'Storage mode: comma-separated read request sizes in KiB to sweep'
complete -c orangu-bench -l storage-file           -r -d 'File the storage probe reads (default: the server\'s largest shard)'
complete -c orangu-bench -l storage-span           -x -d 'MiB to read at each request size, per pass'
complete -c orangu-bench -l storage-ramp           -x -d 'MiB read and discarded before timing starts, at each size'
complete -c orangu-bench -l download-probe         -x -d 'Download mode: comma-separated connection counts to sweep'
complete -c orangu-bench -l download-file          -x -d 'Hub file the download probe fetches, as <user>/<repo>:<path>'
complete -c orangu-bench -l download-seconds       -x -d 'Seconds to download for at each connection count, per pass'
complete -c orangu-bench -l download-dir           -r -d 'Directory the download probe writes into'
complete -c orangu-bench -l cap                    -x -d 'Run each --sweep server under this memory cap (e.g. 4G)'
complete -c orangu-bench -l chart-png                 -d 'Also render a PNG beside the chart SVG'
complete -c orangu-bench -l chart-scale            -x -d 'Pin the chart\'s tok/s axis to MIN:MAX so a pair of charts compare'
complete -c orangu-bench -l chart-y-label          -x -d 'Label for the chart\'s y-axis'
complete -c orangu-bench -l chart-x-label          -x -d 'Label for the chart\'s x-axis'
complete -c orangu-bench -l chart-panels           -x -d 'Draw only these modes\' panels (e.g. pp,tg); default: every mode in the file'
complete -c orangu-bench -l chart-grid                 -d 'Draw one panel per label group and mode, on shared axes'
complete -c orangu-bench -l chart-overlay              -d 'Draw every label group and mode on one panel: hue per group, shade per series'
complete -c orangu-bench -l flamegraph             -r -d 'Record a CPU flamegraph of the server over the measured window'
complete -c orangu-bench -l flamegraph-pid         -x -d 'Process to profile (default: the server\'s own, else the URL port\'s owner)'
complete -c orangu-bench -l flamegraph-freq        -x -d 'Sampling frequency in Hz for --flamegraph'
complete -c orangu-bench -l flamegraph-call-graph  -x -a 'auto fp dwarf' -d 'Call-graph mode for --flamegraph'
complete -c orangu-bench -l flamegraph-png            -d 'Also render a PNG beside the flamegraph SVG'
complete -c orangu-bench -l flamegraph-keep-data      -d 'Keep the raw perf.data beside the flamegraph instead of deleting it'
complete -c orangu-bench -l flamegraph-layers      -r -d 'Profile every running orangu, orangu-coordinator and orangu-server while you drive the workload; one flamegraph per process in DIR'
complete -c orangu-bench -l flamegraph-duration    -x -d 'Seconds to keep sampling under --flamegraph-layers or --flamegraph-watch'
complete -c orangu-bench -l flamegraph-watch          -d 'With --flamegraph, profile the server while something else drives it'
complete -c orangu-bench -l compare-profiles       -r -d 'Compare already-collapsed .folded profiles side by side; measure nothing'
complete -c orangu-bench -l bundle                 -r -d 'Write the whole run — measurements, configuration, host — to one JSON file'
complete -c orangu-bench -l read-bundle            -r -d 'Read bundles and report them side by side; measure nothing'
complete -c orangu-bench -l sweep                  -x -d 'Sweep one tuning variable: VAR=v1,v2,... (; if a value has a comma); needs --sweep-cmd'
complete -c orangu-bench -l sweep-cmd              -x -d 'Shell command that starts the server, run once per --sweep value'
complete -c orangu-bench -l sweep-env              -x -d 'Environment held constant across every --sweep point (repeatable)'
complete -c orangu-bench -l sweep-start-timeout    -x -d 'Seconds to wait for a swept server to come up'
complete -c orangu-bench -l render-profile         -r -d 'Re-render an already-collapsed .folded profile to SVG; measure nothing'
complete -c orangu-bench -l report                 -r -d 'Write the run — provenance, measurements, chart, flamegraph — to one PDF'
complete -c orangu-bench -l web                       -d 'Serve the web console instead of measuring'
complete -c orangu-bench -l host                   -x -a 'all 0.0.0.0 127.0.0.1' -d 'Address the web console binds: all (or *) for every interface'
complete -c orangu-bench -l port                   -x -d 'Port the web console listens on'
complete -c orangu-bench -l delay                  -x -d 'Seconds to wait between measured points, for a card that heats up'
complete -c orangu-bench -l workers                -x -a 'on off compare' -d 'A [workers] tree: on, off, or compare both'
complete -c orangu-bench -l temperature            -x -d 'Sampling temperature for the timed decode; 0 is greedy'
complete -c orangu-bench -s s -l shell-completions    -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-bench -s h -l help                 -d 'Print help'
complete -c orangu-bench -s V -l version              -d 'Print version'
"#;

pub const POWERSHELL: &str = r#"# PowerShell completion for orangu-bench
#
# Quick setup - add to your $PROFILE (notepad $PROFILE):
#   orangu-bench -s | Out-String | Invoke-Expression
#
# Or write once to a file and dot-source it from $PROFILE:
#   orangu-bench -s > "$HOME\orangu-bench.ps1"
#   # $PROFILE: . "$HOME\orangu-bench.ps1"

Register-ArgumentCompleter -Native -CommandName 'orangu-bench' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    # Every word typed before the one under the cursor; $words[0] is the
    # command itself, so $prev is that when nothing else has been typed.
    $words = @($commandAst.CommandElements | ForEach-Object { $_.Extent.Text })
    if ($wordToComplete -ne '' -and $words.Count -gt 1) {
        $words = $words[0..($words.Count - 2)]
    }
    $prev = $words[-1]

    $flags = @(
        @('--url', 'Base URL of the server'),
        @('--depths', 'Comma-separated context depths to sweep; ranges too: 0-2048+512'),
        @('--pp', 'Prefill mode: prompt lengths to sweep, reporting prompt-processing rate'),
        @('--pp-continue', 'Continuation-prefill mode: comma-separated added token counts to sweep'),
        @('--pg', 'Combined mode: prompt lengths prefilled and generated in one request'),
        @('--decode-cpu', 'Report the server''s CPU time per generated token, with prefill excluded'),
        @('--streams', 'Concurrency mode: comma-separated stream counts; reports AGGREGATE tok/s'),
        @('--streams-depth', 'Concurrency mode: tokens of context in front of each stream''s instruction; 0 is none'),
        @('--streams-echo', 'Concurrency mode: each stream copies a passage out of its prompt, which speculation drafts'),
        @('--streams-warm', 'Concurrency mode: prefill the prompts before each run, so all streams decode at once'),
        @('--interference', 'Interference mode: prompt lengths a second request prefills while one stream decodes'),
        @('--chat', 'Chat mode: replay this many turns of a coding session through /v1/chat/completions'),
        @('--chat-fixture', 'The conversation --chat replays (JSON: system, tools, turns)'),
        @('--kl', 'Compare teacher-forced distributions with this reference'),
        @('--kl-rows', 'Positions per --kl pass'),
        @('--kl-top', 'Candidates --kl keeps per position'),
        @('--kl-tokens', 'Tokens --kl runs'),
        @('--kl-text', 'The text --kl runs'),
        @('--perplexity', 'Score this text file'),
        @('--ppl-ctx', 'Tokens in each --perplexity window'),
        @('--ppl-chunks', 'Windows --perplexity runs'),
        @('--hellaswag', 'Score this HellaSwag file'),
        @('--winogrande', 'Score this WinoGrande file'),
        @('--score-tasks', 'Tasks the scoring modes run'),
        @('--score-rows', 'Positions a pass for the scoring modes'),
        @('--shared-prefix', 'Shared-prefix mode: stream counts that all send the SAME long prefix'),
        @('--shared-prefix-tokens', 'Length in tokens of the prefix --shared-prefix streams have in common'),
        @('--shared-prefix-cold', 'Shared-prefix mode: every round sends a prefix the server has not seen, so all streams arrive cold'),
        @('--prefix-scan', 'Scan-resistance mode: unique prompts to push through between two uses of one hot prefix'),
        @('--pp-continue-base', 'Prompt length (tokens) to prime the prefix cache with for --pp-continue'),
        @('--embed', 'Embedding mode: prompt lengths to sweep against /v1/embeddings'),
        @('--image', 'Image mode: square picture sizes to sweep against /v1/images/generations'),
        @('--image-steps', 'Denoising steps per picture for --image'),
        @('--image-cfg', 'Guidance scale for --image; 1 runs the prompt alone'),
        @('--image-prompt', 'The prompt every --image picture is drawn from'),
        @('--image-init', 'A picture attached to every --image request (an edit)'),
        @('--gen', 'Number of tokens to generate per timed run'),
        @('--curve', 'Curve mode: one generation of this many tokens, bucketed by context; 0 disables'),
        @('--bucket', 'Bucket width (in context tokens) for --curve'),
        @('--reps', 'Repetitions per depth; the reported rate is the best run with mean+/-sd'),
        @('--drop-model-cache', 'Evict the model from the page cache before every repetition'),
        @('--no-warmup', 'Skip the initial warmup run'),
        @('--fail-on-regression', 'After a two-value --sweep, exit 2 if the second value is worse beyond the noise'),
        @('--sweep-as-given', 'Run a two-value --sweep in the order written instead of as a,b,b,a'),
        @('--write-profile', 'After a two-value --sweep, keep the winning value in the server tuning profile'),
        @('--per-rep', 'Also print every repetition''s rate, in order, after each row'),
        @('--timeout', 'Per-request timeout in seconds'),
        @('--model', 'Model id to request'),
        @('--json', 'Emit machine-readable JSON'),
        @('--history', 'Append each measured point to this tab-separated history file'),
        @('--label', 'Series name recorded in the history file; prefixes each --sweep point'),
        @('--chart', 'Render the history file to this SVG after measuring'),
        @('--chart-only', 'Only render the chart and table from an existing history file; measure nothing'),
        @('--table', 'Write the history file as a markdown comparison table to this path (- for stdout)'),
        @('--storage-probe', 'Storage mode: comma-separated read request sizes in KiB to sweep'),
        @('--storage-file', 'File the storage probe reads (default: the server''s largest shard)'),
        @('--storage-span', 'MiB to read at each request size, per pass'),
        @('--storage-ramp', 'MiB read and discarded before timing starts, at each size'),
        @('--download-probe', 'Download mode: comma-separated connection counts to sweep'),
        @('--download-file', 'Hub file the download probe fetches, as <user>/<repo>:<path>'),
        @('--download-seconds', 'Seconds to download for at each connection count, per pass'),
        @('--download-dir', 'Directory the download probe writes into'),
        @('--cap', 'Run each --sweep server under this memory cap (e.g. 4G)'),
        @('--chart-png', 'Also render a PNG beside the chart SVG'),
        @('--chart-scale', 'Pin the chart''s tok/s axis to MIN:MAX so a pair of charts compare'),
        @('--chart-y-label', 'Label for the chart''s y-axis'),
        @('--chart-x-label', 'Label for the chart''s x-axis'),
        @('--chart-panels', 'Draw only these modes'' panels (e.g. pp,tg); default: every mode in the file'),
        @('--chart-grid', 'Draw one panel per label group and mode, on shared axes'),
        @('--chart-overlay', 'Draw every label group and mode on one panel: hue per group, shade per series'),
        @('--flamegraph', 'Record a CPU flamegraph of the server over the measured window'),
        @('--flamegraph-pid', 'Process to profile (default: the server''s own, else the URL port''s owner)'),
        @('--flamegraph-freq', 'Sampling frequency in Hz for --flamegraph'),
        @('--flamegraph-call-graph', 'Call-graph mode for --flamegraph: fp or dwarf'),
        @('--flamegraph-png', 'Also render a PNG beside the flamegraph SVG'),
        @('--flamegraph-keep-data', 'Keep the raw perf.data beside the flamegraph instead of deleting it'),
        @('--flamegraph-layers', 'Profile every running orangu, orangu-coordinator and orangu-server while you drive the workload; one flamegraph per process in DIR'),
        @('--flamegraph-duration', 'Seconds to keep sampling under --flamegraph-layers or --flamegraph-watch'),
        @('--flamegraph-watch', 'With --flamegraph, profile the server while something else drives it'),
        @('--compare-profiles', 'Compare already-collapsed .folded profiles side by side; measure nothing'),
        @('--bundle', 'Write the whole run - measurements, configuration, host - to one JSON file'),
        @('--read-bundle', 'Read bundles and report them side by side; measure nothing'),
        @('--sweep', 'Sweep one tuning variable: VAR=v1,v2,... (; if a value has a comma); needs --sweep-cmd'),
        @('--sweep-cmd', 'Shell command that starts the server, run once per --sweep value'),
        @('--sweep-env', 'Environment held constant across every --sweep point (repeatable)'),
        @('--sweep-start-timeout', 'Seconds to wait for a swept server to come up'),
        @('--render-profile', 'Re-render an already-collapsed .folded profile to SVG; measure nothing'),
        @('--report', 'Write the run - provenance, measurements, chart, flamegraph - to one PDF'),
        @('--web', 'Serve the web console instead of measuring'),
        @('--host', 'Address the web console binds: all (or *) for every interface'),
        @('--port', 'Port the web console listens on'),
        @('--delay', 'Seconds to wait between measured points, for a card that heats up'),
        @('--workers', 'A [workers] tree: on, off, or compare both'),
        @('--temperature', 'Sampling temperature for the timed decode; 0 is greedy'),
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

    switch ($prev) {
        # A path: PowerShell's own completion takes over.
        { $_ -in '--history', '--chart', '--table', '--storage-file', '--download-dir', '--flamegraph', '--image-init', '--compare-profiles', '--bundle', '--read-bundle', '--render-profile', '--report' } { return }
        '--flamegraph-call-graph' { return Offer @('fp', 'dwarf') }
        '--host' { return Offer @('all', '0.0.0.0', '127.0.0.1') }
        { $_ -in '--url', '--depths', '--pp', '--pp-continue', '--pg', '--streams', '--streams-depth', '--interference', '--chat', '--chat-fixture', '--kl', '--kl-rows', '--kl-top', '--kl-tokens', '--kl-text', '--perplexity', '--ppl-ctx', '--ppl-chunks', '--hellaswag', '--winogrande', '--score-tasks', '--score-rows', '--shared-prefix', '--shared-prefix-tokens', '--prefix-scan', '--pp-continue-base', '--embed', '--gen', '--curve', '--bucket', '--reps', '--timeout', '--model', '--label', '--storage-probe', '--storage-span', '--storage-ramp', '--download-probe', '--download-file', '--download-seconds', '--cap', '--chart-scale', '--chart-y-label', '--chart-x-label', '--chart-panels', '--flamegraph-pid', '--flamegraph-freq', '--flamegraph-layers', '--flamegraph-duration', '--sweep', '--sweep-cmd', '--sweep-env', '--sweep-start-timeout', '--port', '--delay', '--temperature', '--workers' } { return }
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
