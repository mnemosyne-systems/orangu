# Building a Model From Scratch

**Goal:** pretrain an original LLM — no base checkpoint, random weights on
day one — specialized in five languages, **C, Rust, Java, TypeScript, and
Bash**, plus enough general **English** to follow instructions and explain
itself in prose. The target is a **dense ~2B parameter model ("B2")**, or,
hardware permitting, a **sparse Mixture-of-Experts model with ~2B
*active* parameters ("Bx-A2")**, trained entirely from
permissively-licensed GitHub repositories and open English corpora.

This guide gives two independent, complete paths to that same target: a
**Rust toolchain** built on [Burn](https://github.com/tracel-ai/burn), and
a **Python toolchain** built on PyTorch/`transformers`. They aren't meant
to be mixed — pick one. [General](#general) covers everything neither
path changes (data collection, the architecture spec, evaluation, GGUF
export, running the result in orangu); the [Appendix](#appendix) lays out
the trade-offs between the two to help you choose. Every command is
written for **AMD hardware**, but "Choose an NVIDIA card" in General
covers the equivalent hardware tiers and the (mostly mechanical) command
swaps if that's what you have instead.

## General

Everything in this section is shared — the commands and file formats here
don't depend on which toolchain trained the model.

### Reality check before you start

A model in this class is not a laptop weekend project — though B2 is
genuinely small enough that a single high-end consumer AMD card can train
it, just slowly. Rather than committing straight to a B2 run and finding
out three weeks in that something upstream (data mix, tokenizer, packing)
was wrong, this guide treats B0.5 and B1 as real, worth-evaluating
milestones on the way there, not just throwaway smoke tests — both
toolchains reuse the same tokenizer, packed corpus, and training command
for all three, swapping only the size and output directory. Rough orders
of magnitude for each stage, so you can size your plan before committing:

| | Smoke test | B0.5 | B1 | B2 |
|:---|:---|:---|:---|:---|
| Parameters | ~10M | ~0.5B | ~1B | ~2B dense, or ~25B total / 2B active (MoE) |
| Training tokens | 5+M | 50B–100B | 100B–300B | 250B–1T+ |
| Hardware | 1 x Entry-level | 1 x High-end | 1+ x High-end | 2+ High-end (ideally 4+) |
| Wall clock | minutes | a few days | one to two weeks | weeks to a few months |

The token target matters more than it looks: Chinchilla scaling puts the
*bare minimum* at roughly 20 tokens/parameter — ~10B for B0.5, ~20B for
B1, ~40B for B2 — but every recent code model (StarCoder2, DeepSeek-Coder,
Qwen2.5-Coder) trains **far** past that floor because code+text data is
cheap relative to compute and undertrained models are noticeably worse
coders. The token counts in the table already build in that margin; don't
drop below the Chinchilla floor at any stage or the run isn't worth
evaluating.

Exactly how multiple GPUs get used differs a lot between the two
toolchains — see "Set up training devices" under the Rust toolchain and
"Configure distributed pretraining" under the Python toolchain, and the
Appendix for the trade-off in one place. The "1 x Entry-level" smoke-test
row is really a formality — it runs on CPU alone (Burn's `ndarray`
backend, or plain PyTorch with no CUDA/ROCm) and needs no GPU at all; see
the toolchain sections' own smoke-test steps.

Tier names (Entry-level/Mid-range/High-end) match "Choosing an AMD card"
and "Choose an NVIDIA card" below — use whichever vendor's table you need.

Everything in this guide is real, runnable tooling — but scale the
numbers (batch size, step count, hardware) to what you actually have.

### Choosing an AMD card

This guide uses the **Radeon AI PRO R9700** (32GB, RDNA4) as its running
example — a workstation card AMD positions specifically for local AI
workloads, with enough VRAM to comfortably run the whole B0.5→B1→B2 dense
progression as a single replica and room to spare for larger batches or
longer context than a 24GB card allows. As a PRO-tier AI card it's also
more likely to have consistent day-one ROCm support than a gaming-first
card, though always check AMD's ROCm compatibility matrix for your exact
ROCm version before buying — support windows do shift release to release.

It isn't the only reasonable choice. Roughly by tier:

| Tier | Card | VRAM | Architecture | Good for, in this guide |
|:---|:---|:---|:---|:---|
| Entry-level | Radeon RX 9070 XT | 16GB | RDNA4 (consumer) | The cheapest option worth running past the smoke test — comfortable for B0.5, tight once B2's larger batches want more headroom |
| Mid-range | Radeon RX 7900 XTX | 24GB | RDNA3 (consumer) | Solid all-rounder — fits B2 dense fine, well-established ROCm support on Linux |
| High-end | **Radeon AI PRO R9700** | **32GB** | **RDNA4 (workstation)** | **This guide's example — comfortable headroom for B0.5/B1/B2 dense, PRO-tier AI positioning** |
| High-end | Radeon PRO W7900 | 48GB | RDNA3 (workstation) | Extra headroom for bigger batches/longer context, or a leaner slice of experimentation toward the MoE alternative |
| High-end | Instinct MI210 | 64GB (HBM2e) | CDNA2 (datacenter, PCIe) | Easier to rack than OAM-form-factor cards; still short of the ~50GB+ a Bx-A2 MoE replica needs with room for training state |
| High-end | Instinct MI300X | 192GB (HBM3) | CDNA3 (datacenter) | The realistic single-card floor for the Bx-A2 MoE alternative as one replica (see "Set up training devices") |
| High-end | Instinct MI325X | 256GB (HBM3e) | CDNA3 (datacenter) | Same role as the MI300X with more headroom, for a larger MoE or bigger batches |

For the dense B0.5/B1/B2 path, anything from the RX 9070 XT up is
genuinely fine — the R9700 AI PRO hits a practical sweet spot without
needing full datacenter Instinct hardware, not a requirement. The
Instinct rows only start to matter once you're pursuing the MoE
alternative, where single-replica memory (see the FSDP note under "Set
up training devices") is the binding constraint, not compute. If you're
on NVIDIA hardware instead, see "Choose an NVIDIA card" below for the
equivalent tiering.

### Choose an NVIDIA card

Every command in this guide targets AMD/ROCm, but nothing about the
B0.5→B1→B2 (or Bx-A2 MoE) targets is AMD-specific — CUDA is, if anything,
the more mature, better-tested path for both PyTorch and Burn, precisely
because it's what rocBLAS/MIOpen and Burn's own HIP-via-`cubecl` backend
are chasing parity with (see the Performance appendix). If you're already
on NVIDIA hardware, this table uses the same tiers as the AMD one above:

| Tier | Card | VRAM | Architecture | Good for, in this guide |
|:---|:---|:---|:---|:---|
| Entry-level | GeForce RTX 5070 Ti | 16GB (GDDR7) | Blackwell (consumer) | Matches the RX 9070 XT's tier — comfortable for B0.5, tight for B2's larger batches |
| Mid-range | GeForce RTX 4090 | 24GB (GDDR6X) | Ada Lovelace (consumer) | Same VRAM class as the RX 7900 XTX — fits B2 dense fine |
| High-end | GeForce RTX 5090 | 32GB (GDDR7) | Blackwell (consumer) | Matches this guide's R9700 AI PRO on VRAM — comfortable headroom for B0.5/B1/B2 dense |
| High-end | RTX PRO 6000 Blackwell¹ | 96GB (GDDR7) | Blackwell (workstation) | Well past the AMD workstation tier's 48GB — closer to, though likely still short of, comfortably training the Bx-A2 MoE alternative solo |
| High-end | H100 | 80GB (HBM3) | Hopper (datacenter) | The default choice on most existing CUDA clusters, cloud or on-prem |
| High-end | H200 | 141GB (HBM3e) | Hopper (datacenter) | Same role as the H100, with more headroom |
| High-end | B200 | 192GB (HBM3e) | Blackwell (datacenter) | Matches the Instinct MI300X almost exactly — the realistic single-card floor for the Bx-A2 MoE alternative on NVIDIA |

¹ NVIDIA's workstation branding has shifted release to release (Quadro →
RTX A-series → RTX 6000 Ada Generation → RTX PRO Blackwell) — verify the
current name and exact VRAM figure before buying.

Swapping the rest of this guide's commands over to CUDA is mostly
mechanical:

- **Rust toolchain**: build with Burn's `cuda` feature (via `cubecl-cuda`)
  instead of `wgpu` — check `burn`'s current docs for the exact feature
  name, since like the HIP backend mentioned earlier, this has moved
  across releases; it's a more established path than either of Burn's AMD
  backends.
- **Python toolchain**: `pip install torch` from the default PyPI/CUDA
  wheel index instead of the ROCm-specific one, and drop the ROCm caveats
  raised earlier — `bitsandbytes` is fully mature on CUDA. `flash-attn`
  (skipped entirely on the AMD path) is worth installing here — it's the
  original CUDA-only package and a genuine speedup.
- **GGUF conversion (General, above)**: build llama.cpp with
  `-DGGML_CUDA=ON` instead of `-DGGML_VULKAN=ON`/`-DGGML_HIP=ON` — no
  `gfx` target to look up; llama.cpp detects your CUDA compute capability
  automatically.

### Fetch the raw corpus (git clone / curl, one script per source)

Lay out a workspace with one directory per source, populated by one script
per source. Nothing here calls a dataset-hosting library — it's `git clone`
and `curl` all the way down, so every source is independently inspectable,
re-runnable, and deletable. This step is identical regardless of which
toolchain you use later.

```sh
model-build/
  corpus/
    c/            rust/            java/           typescript/     bash/
    english-wikipedia/             english-fineweb/
  scripts/
    discover_repos.sh
    fetch_c.sh    fetch_rust.sh    fetch_java.sh   fetch_typescript.sh
    fetch_bash.sh fetch_english_wikipedia.sh        fetch_english_fineweb.sh
```

#### Shared helper: discover permissively-licensed GitHub repos

`scripts/discover_repos.sh` queries the GitHub Search API for popular repos
in one language under one license and appends `owner/repo` lines to a file.
Restricting to MIT/Apache-2.0/BSD keeps the corpus free of copyleft
obligations; GitHub's search only ANDs multiple `license:` qualifiers
together, so each license is queried separately and the results merged.

```sh
#!/usr/bin/env bash
# scripts/discover_repos.sh <linguist-language> <license> <out-file> [pages]
set -euo pipefail

LANGUAGE="$1"
LICENSE="$2"
OUT="$3"
PAGES="${4:-5}"   # 5 pages x 100 results = up to 500 repos for this license

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token))}"

for page in $(seq 1 "$PAGES"); do
  curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" \
          -H "Accept: application/vnd.github+json" \
          "https://api.github.com/search/repositories?q=language:${LANGUAGE}+license:${LICENSE}+stars:>100&sort=stars&order=desc&per_page=100&page=${page}" \
    | python3 -c "import sys, json; d = json.load(sys.stdin); [print(i['full_name']) for i in d.get('items', [])]" \
    >> "$OUT"
  sleep 2   # stay comfortably under the search API's authenticated rate limit
done
```

#### One fetch script per programming language

`scripts/fetch_c.sh` (the other four languages are the same script with the
language name and linguist tag swapped):

```sh
#!/usr/bin/env bash
# scripts/fetch_c.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=corpus/c
REPO_LIST=corpus/c.repos.txt
mkdir -p "$DEST"
: > "$REPO_LIST"

for license in mit apache-2.0 bsd-3-clause; do
  ./scripts/discover_repos.sh C "$license" "$REPO_LIST"
done
sort -u -o "$REPO_LIST" "$REPO_LIST"

while read -r repo; do
  name="${repo//\//__}"
  [ -d "$DEST/$name" ] && continue
  git clone --depth 1 "https://github.com/${repo}.git" "$DEST/$name" || true
done < "$REPO_LIST"
```

Copy it to the other four, changing only the two lines that name the
language and destination:

| Script | `DEST` | GitHub linguist language |
|:---|:---|:---|
| `fetch_c.sh` | `corpus/c` | `C` |
| `fetch_rust.sh` | `corpus/rust` | `Rust` |
| `fetch_java.sh` | `corpus/java` | `Java` |
| `fetch_typescript.sh` | `corpus/typescript` | `TypeScript` |
| `fetch_bash.sh` | `corpus/bash` | `Shell` (GitHub's linguist tag for `.sh`/`.bash` scripts) |

Run them (each is independent, so they parallelize fine):

```sh
export GITHUB_TOKEN=$(gh auth token)
for lang in c rust java typescript bash; do
  ./scripts/fetch_${lang}.sh &
done
wait
```

For the smoke test, pass fewer pages (edit `PAGES` to `1`, or call
`discover_repos.sh` directly) so you clone dozens of repos, not thousands.

Optional cleanup pass, since TypeScript repos in particular drag in build
output and dependency trees that aren't representative source:

```sh
find corpus/typescript corpus/java -type d \( -name node_modules -o -name dist -o -name build -o -name target \) -prune -exec rm -rf {} +
find corpus -type f -size +2M -delete   # drop generated/minified/vendored blobs
```

#### English: Wikipedia (curl, official dump)

```sh
#!/usr/bin/env bash
# scripts/fetch_english_wikipedia.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=corpus/english-wikipedia
mkdir -p "$DEST"

curl -L --continue-at - -o "$DEST/enwiki-latest-pages-articles.xml.bz2" \
  "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

python3 -m wikiextractor.WikiExtractor \
  "$DEST/enwiki-latest-pages-articles.xml.bz2" \
  -o "$DEST/extracted" --json
```

Clean encyclopedic prose — reliable grammar, low noise, but narrow in tone.
That's why the second English source below matters too. (`wikiextractor`
is a one-off Python tool that only runs here, during corpus prep — it has
nothing to do with either training toolchain and isn't needed again after
this.)

#### English: FineWeb-Edu (curl, quality-filtered web text)

Wikipedia alone (~4B tokens) is nowhere near enough English for a model
this size, and it never sees a casual sentence, a forum answer, or a
tutorial. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
is web text filtered by an educational-quality classifier — broader
register and topic coverage than an encyclopedia, still clean. It's hosted
on the Hugging Face Hub as plain Parquet files, so `curl` works directly
against them; the script below asks the Hub's file-listing API for the
actual shard names rather than guessing them, then curls each one (each
shard is roughly 0.7GB of text):

```sh
#!/usr/bin/env bash
# scripts/fetch_english_fineweb.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=corpus/english-fineweb
mkdir -p "$DEST"
NUM_SHARDS="${1:-10}"   # raise this for more English text

curl -s "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT" \
  | python3 -c "
import sys, json
files = json.load(sys.stdin)
names = sorted(f['path'] for f in files if f['path'].endswith('.parquet'))
for n in names[:$NUM_SHARDS]:
    print(n)
" > "$DEST/shards.txt"

while read -r path; do
  curl -L --continue-at - -o "$DEST/$(basename "$path")" \
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/${path}"
done < "$DEST/shards.txt"
```

Raise `NUM_SHARDS` as your target token budget grows; check the dataset's
"Files" tab on the Hub if you want to hand-pick shards instead.

### Define the target architecture

The actual sizing target is shared; each toolchain section shows how to
instantiate it (a JSON config for the Python toolchain, a Rust `Config`
for the Rust toolchain). Four sizes, sharing one tokenizer and vocabulary,
meant to be trained in sequence per the staged plan above:

| Size | `hidden_size` | `intermediate_size` | `num_hidden_layers` | `num_attention_heads` | `num_key_value_heads` | ~Params |
|:---|:---|:---|:---|:---|:---|:---|
| `Smoke` | 256 | 688 | 4 | 4 | 4 | ~10M |
| `0.5b` | 1280 | 3456 | 24 | 10 | 5 | ~0.5B |
| `1b` | 1536 | 5632 | 28 | 12 | 6 | ~1B |
| `2b` | 2048 | 8192 | 30 | 16 | 8 | ~2B |

All four share `vocab_size: 32768` and `rms_norm_eps: 1e-5`; the three
real sizes share `max_position_embeddings: 8192` and `rope_theta: 1e6`,
while `smoke` uses a `max_position_embeddings` of `512`, since it's only
there to validate the pipeline.

**MoE alternative, Bx-A2 (~2B active):** same attention stack, but each
layer's MLP becomes a bank of 64 small routed experts (each a smaller
`intermediate_size: 2048` MLP) — only 4 of the 64 run per token, so
*active* compute stays near 2B while *total* capacity is much larger
(~25B total, a ~12.5x sparsity ratio, in the same range as
Qwen3-30B-A3B's ~10x). A router load-balancing auxiliary loss
(`router_aux_loss_coef: 0.02`) keeps training from collapsing onto a
handful of favorite experts. **This guide recommends the Python toolchain
specifically for the MoE variant** — see the Appendix for why.

### Evaluate the trained model

This step needs Python regardless of which toolchain trained the model —
there's no Rust equivalent of either evaluation harness below, so `pip
install` them into a throwaway venv just for this step if you used the
Rust toolchain; nothing about either training stack depends on it. Point
both commands at whatever HF-style `config.json` + `.safetensors`
directory your toolchain produced (directly, if you used the Python
toolchain; via its export step, if you used the Rust one).

Code, via [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)
(MultiPL-E covers Rust/Java/TypeScript; HumanEvalPack adds a second Java/JS
check):

```sh
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness
accelerate launch main.py \
  --model <path-to-your-hf-style-checkpoint> \
  --tasks multiple-rs,multiple-java,multiple-ts,humanevalpack \
  --allow_code_execution \
  --temperature 0.2 --n_samples 20 --batch_size 8
```

English understanding, via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```sh
pip install lm-eval
lm_eval --model hf \
  --model_args pretrained=<path-to-your-hf-style-checkpoint> \
  --tasks mmlu,hellaswag,arc_challenge,winogrande \
  --device cuda:0 --batch_size 8
```

### Convert to GGUF and produce the major quantizations

Reimplementing GGUF's exact tensor-naming and metadata conventions by hand
— the kind of detail that's easy to get subtly wrong and hard to notice
until llama.cpp refuses to load the result — would be reckless compared to
reusing the tool that already gets it right, so both toolchains hand off
to llama.cpp's own `convert_hf_to_gguf.py` here, fed by whichever
toolchain's HF-style checkpoint directory.

Build llama.cpp with the **Vulkan** backend — it runs on essentially any
AMD GPU (including older or integrated ones, unlike ROCm's narrower
supported-card list) and is what the rest of orangu's own docs assume for
AMD hardware (see [GGUF.md](GGUF.md)):

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

If your card is one ROCm officially supports (RDNA3/RDNA4/CDNA, e.g. an
R9700 AI PRO or an MI-series Instinct) and you want the extra performance
HIP tends to have over Vulkan on those cards, build against ROCm instead —
swap `gfx1201` below for your card's actual `gfx` target
(`rocminfo | grep gfx` tells you which; don't assume — RDNA4 cards like
the R9700 AI PRO use the newer `gfx120x` family, not RDNA3's `gfx110x`
that a 7900 XTX reports):

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Convert to an unquantized F16 GGUF, then build an importance matrix from a
held-out sample of your own training mix (a few hundred KB of mixed code +
English text) so the low-bit quantizations below lose as little quality as
possible:

```sh
python3 convert_hf_to_gguf.py <path-to-your-hf-style-checkpoint> \
  --outfile ../my-code-model-f16.gguf --outtype f16

cd build
./bin/llama-imatrix -m ../../my-code-model-f16.gguf \
  -f ../../calibration-sample.txt -o ../../my-code-model.imatrix
```

Produce the full standard spread of `Q*` quantizations from that single
F16 + imatrix pair:

```sh
for Q in Q2_K Q3_K_M Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0; do
  ./bin/llama-quantize --imatrix ../../my-code-model.imatrix \
    ../../my-code-model-f16.gguf ../../my-code-model-${Q}.gguf ${Q}
done
```

| Quant | ~bits/weight | Use it when |
|:---|:---|:---|
| `Q2_K` | 2.6 | VRAM is extremely tight; expect a real quality drop |
| `Q3_K_M` | 3.4 | Tight VRAM, better than `Q2_K` |
| `Q4_0` | 4.5 | Legacy/simple 4-bit; `Q4_K_M` is almost always better at the same size |
| `Q4_K_M` | 4.8 | The default sweet spot — `orangu-gguf`'s own preferred quant, see [GGUF.md](GGUF.md) |
| `Q5_K_M` | 5.7 | Noticeably closer to F16 quality, modest size increase |
| `Q6_K` | 6.6 | Near-lossless, for when VRAM allows it |
| `Q8_0` | 8.5 | Effectively lossless; `orangu-gguf`'s second-choice default |

### Run it in orangu

```sh
llama-server -m ./my-code-model-Q4_K_M.gguf --port 8100 --ctx-size 8192 -fa on --jinja
```

```ini
[orangu]
server = main-server
model = my-code-model-Q4_K_M

[main-server]
provider = llama.cpp
endpoint = http://localhost:8100/v1
model = my-code-model-Q4_K_M
```

See [LOCAL_LLM.md](LOCAL_LLM.md) for configuration details and
[GGUF.md](GGUF.md) for `orangu-gguf`'s model-inventory and role-wizard
tooling.

## Rust toolchain

Tokenizer, data pipeline, model, and training loop are all Rust here — no
Python runtime required to train (the two General steps that stay Python
regardless — evaluation and GGUF conversion — are unaffected by this
choice). Burn ships no ready-made Llama/Mixtral training architecture the
way `transformers` does, so the model itself is hand-written below; treat
exact Burn method names as close-but-verify against whatever version you
pin, since the tensor API has moved across releases.

### Prerequisites

- A Linux machine (or several) with an AMD GPU and ROCm installed for
  anything past the smoke test. This guide's example is a single 32GB
  **Radeon AI PRO R9700** — enough headroom to run the whole dense
  B0.5→B1→B2 progression comfortably; datacenter Instinct cards
  (MI300X/MI325X) are the realistic floor once you're pursuing the MoE
  alternative instead. See "Choosing an AMD card" in General for the
  broader lineup.
- Confirm ROCm sees the card(s) before doing anything else:

```sh
rocminfo | grep -i gfx
rocm-smi
```

- The Rust toolchain, and — for the GGUF export step in General — the
  `orangu` build prerequisites from [BUILDING.md](BUILDING.md):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

- A GitHub personal access token for the corpus-discovery step above, if
  you haven't already set one up: `gh auth token`, if you use `gh` with
  orangu's forge integration.

Project layout and `Cargo.toml`:

```
model-build/
  Cargo.toml
  corpus/...      scripts/...        (the corpus fetch step, above)
  src/
    lib.rs
    corpus.rs      model.rs      data.rs      export.rs      safetensors.rs
    bin/
      train_tokenizer.rs   pack_corpus.rs   pretrain.rs
      export_hf.rs         sft.rs
```

```toml
[package]
name = "model-build"
version = "0.1.0"
edition = "2021"

[lib]
name = "model_build"
path = "src/lib.rs"

[dependencies]
burn = { version = "0.17", default-features = false, features = ["train", "autodiff"] }
tokenizers = "0.20"
polars = { version = "0.44", features = ["parquet"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
walkdir = "2"
memmap2 = "0.9"
rand = "0.8"
clap = { version = "4", features = ["derive"] }

[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]
ndarray = ["burn/ndarray"]

[[bin]]
name = "train_tokenizer"
path = "src/bin/train_tokenizer.rs"
[[bin]]
name = "pack_corpus"
path = "src/bin/pack_corpus.rs"
[[bin]]
name = "pretrain"
path = "src/bin/pretrain.rs"
[[bin]]
name = "export_hf"
path = "src/bin/export_hf.rs"
[[bin]]
name = "sft"
path = "src/bin/sft.rs"
```

The `wgpu` feature builds against Vulkan — it runs on essentially any AMD
GPU (matching this project's own AMD/Vulkan preference for llama.cpp, see
[GGUF.md](GGUF.md)) and is the default. The `ndarray` feature is a
pure-CPU backend, used only for the smoke test below. If your Burn version
ships a ROCm/HIP backend directly (via `cubecl-hip` or similar — check
`burn`'s current docs, the crate/feature name has moved around across
releases) and your card is one ROCm officially supports, that's a faster
option for the real runs than Vulkan; this guide sticks to `wgpu` as the
one that's guaranteed to work everywhere.

```sh
cargo build --release
```

### Train a tokenizer on the fetched corpus

`src/corpus.rs` reads text out of every corpus directory regardless of its
format (raw source files, wikiextractor's JSON-lines, or Parquet), so both
the tokenizer trainer and the packer below share the same code. Parquet
reading uses `polars` — its `scan_parquet`/`LazyFrame` API has shifted
shape across releases, so check it against whatever version you pin:

```rust
// src/corpus.rs
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const CODE_EXTENSIONS: &[(&str, &[&str])] = &[
    ("c", &["c", "h"]),
    ("rust", &["rs"]),
    ("java", &["java"]),
    ("typescript", &["ts", "tsx"]),
    ("bash", &["sh", "bash"]),
];

pub fn code_paths(corpus_dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for (lang, extensions) in CODE_EXTENSIONS {
        let lang_dir = corpus_dir.join(lang);
        for entry in WalkDir::new(&lang_dir).into_iter().filter_map(Result::ok) {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext) {
                    paths.push(entry.path().to_path_buf());
                }
            }
        }
    }
    paths
}

#[derive(serde::Deserialize)]
struct WikiDoc {
    text: String,
}

pub fn extract_wikipedia_text(corpus_dir: &Path) -> std::io::Result<PathBuf> {
    let extracted_dir = corpus_dir.join("english-wikipedia").join("extracted");
    let out_path = corpus_dir.join("english-wikipedia").join("extracted.txt");
    let mut out = std::io::BufWriter::new(fs::File::create(&out_path)?);
    for entry in WalkDir::new(&extracted_dir).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let file = fs::File::open(entry.path())?;
        for line in BufReader::new(file).lines().filter_map(Result::ok) {
            if let Ok(doc) = serde_json::from_str::<WikiDoc>(&line) {
                use std::io::Write;
                writeln!(out, "{}", doc.text)?;
            }
        }
    }
    Ok(out_path)
}

pub fn extract_fineweb_text(corpus_dir: &Path) -> polars::prelude::PolarsResult<PathBuf> {
    use polars::prelude::*;
    use std::io::Write;

    let fineweb_dir = corpus_dir.join("english-fineweb");
    let out_path = fineweb_dir.join("extracted.txt");
    let mut out = std::io::BufWriter::new(fs::File::create(&out_path).unwrap());

    let mut shard_paths: Vec<_> = fs::read_dir(&fineweb_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .collect();
    shard_paths.sort();

    for path in shard_paths {
        let df = LazyFrame::scan_parquet(&path, ScanArgsParquet::default())?
            .select([col("text")])
            .collect()?;
        let series = df.column("text")?.str()?;
        for text in series.into_iter().flatten() {
            writeln!(out, "{text}").ok();
        }
    }
    Ok(out_path)
}

pub fn all_text_sources(corpus_dir: &Path) -> Vec<PathBuf> {
    let mut paths = code_paths(corpus_dir);
    paths.push(extract_wikipedia_text(corpus_dir).expect("wikipedia extraction failed"));
    paths.push(extract_fineweb_text(corpus_dir).expect("fineweb extraction failed"));
    paths
}
```

```rust
// src/lib.rs
pub mod corpus;
pub mod data;
pub mod export;
pub mod model;
pub mod safetensors;
```

Tokenizer training uses HuggingFace's `tokenizers` crate directly — it's
native Rust (the Python `tokenizers` package used in the Python toolchain
below is a binding *around* this same crate), so no Python is involved:

```rust
// src/bin/train_tokenizer.rs
use model_build::corpus::all_text_sources;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let special_tokens = [
        "<|endoftext|>",
        "<|pad|>",
        "<fim_prefix>",
        "<fim_middle>",
        "<fim_suffix>",
        "<fim_pad>", // fill-in-the-middle, useful for code completion
        "<|im_start|>",
        "<|im_end|>", // reserved for the chat template below
    ];

    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(32768) // a 5-language + English vocab needs far less than a
        // 100+ language general tokenizer's ~150K entries
        .special_tokens(
            special_tokens
                .iter()
                .map(|t| AddedToken::from(t.to_string(), true))
                .collect(),
        )
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let files: Vec<String> = all_text_sources(std::path::Path::new("corpus"))
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    tokenizer.train_from_files(&mut trainer, files)?;
    tokenizer.get_tokenizer().save("tokenizer.json", true)?;
    Ok(())
}
```

```sh
cargo run --release --bin train_tokenizer
```

### Tokenize and pack the corpus into shards

Concatenate every document's tokens (with an `<|endoftext|>` boundary
between documents), and slice the stream into fixed-length blocks — the
standard packing scheme for causal-LM pretraining, so the model never
wastes a forward pass on padding. Blocks are written as raw little-endian
`u16` values (the 32,768-entry vocabulary fits comfortably) in ~100MB
shards — no NumPy, just flat binary files read back with `memmap2` below:

```rust
// src/bin/pack_corpus.rs
use model_build::corpus::all_text_sources;
use std::fs;
use std::io::Write;
use tokenizers::Tokenizer;

const SEQ_LEN: usize = 8192;
const TOKENS_PER_SHARD: usize = 50_000_000;

fn flush_shard(buffer: &[u16], idx: usize) -> std::io::Result<()> {
    let n_blocks = buffer.len() / SEQ_LEN;
    let mut file = fs::File::create(format!("packed_corpus/shard_{idx:05}.bin"))?;
    let bytes: Vec<u8> = buffer[..n_blocks * SEQ_LEN]
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    file.write_all(&bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    let eos_id = tokenizer.token_to_id("<|endoftext|>").unwrap() as u16;

    fs::create_dir_all("packed_corpus")?;
    let mut buffer: Vec<u16> = Vec::with_capacity(TOKENS_PER_SHARD);
    let mut shard_idx = 0usize;

    for path in all_text_sources(std::path::Path::new("corpus")) {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let encoding = tokenizer.encode(text, false)?;
        buffer.extend(encoding.get_ids().iter().map(|&id| id as u16));
        buffer.push(eos_id);
        if buffer.len() >= TOKENS_PER_SHARD {
            flush_shard(&buffer, shard_idx)?;
            shard_idx += 1;
            buffer.clear();
        }
    }
    if buffer.len() >= SEQ_LEN {
        flush_shard(&buffer, shard_idx)?;
    }
    Ok(())
}
```

```sh
cargo run --release --bin pack_corpus
```

### Implement the architecture in Burn

GQA attention, RoPE, RMSNorm, and the SwiGLU MLP, all as Burn `Module`s.
The shapes follow Burn's own text-generation example patterns
(`Module`/`Config` derives, the built-in `RotaryEncoding` and
`generate_autoregressive_mask`):

```rust
// src/model.rs
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig,
};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[config(default = "1.0e-5")]
    pub rms_norm_eps: f64,
    #[config(default = "1_000_000.0")]
    pub rope_theta: f64,
}

/// Matches the size table under "Define the target architecture" — used by
/// every binary in this section via `--model-size`.
pub fn config_for_size(size: &str) -> ModelConfig {
    let (hidden, intermediate, layers, heads, kv_heads, max_pos) = match size {
        "smoke" => (256, 688, 4, 4, 4, 512),
        "0.5b" => (1280, 3456, 24, 10, 5, 8192),
        "1b" => (1536, 5632, 28, 12, 6, 8192),
        "2b" => (2048, 8192, 30, 16, 8, 8192),
        other => panic!("unknown model size {other}"),
    };
    ModelConfig::new(32768, hidden, intermediate, layers, heads, kv_heads, max_pos)
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let layers = (0..self.num_hidden_layers)
            .map(|_| {
                DecoderBlockConfig::new(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.max_position_embeddings,
                )
                .with_rms_norm_eps(self.rms_norm_eps)
                .with_rope_theta(self.rope_theta)
                .init(device)
            })
            .collect();

        Model {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(self.hidden_size)
                .with_eps(self.rms_norm_eps)
                .init(device),
            lm_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DecoderBlock<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.lm_head.forward(self.norm.forward(hidden))
    }
}

// --- RMSNorm: not shipped in burn::nn, hand-rolled ---

#[derive(Config, Debug)]
pub struct RmsNormConfig {
    pub d_model: usize,
    #[config(default = "1.0e-5")]
    pub eps: f64,
}

impl RmsNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm {
            weight: Param::from_tensor(Tensor::ones([self.d_model], device)),
            eps: self.eps,
        }
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x / (variance + self.eps).sqrt();
        normed * self.weight.val().unsqueeze()
    }
}

// --- SwiGLU MLP ---

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// --- Grouped-query attention with RoPE ---

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_position_embeddings: usize,
    #[config(default = "1_000_000.0")]
    pub rope_theta: f64,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let head_dim = self.hidden_size / self.num_heads;
        Attention {
            q_proj: LinearConfig::new(self.hidden_size, self.num_heads * head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, self.num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, self.num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(self.num_heads * head_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            rotary: RotaryEncodingConfig::new(self.max_position_embeddings, head_dim)
                .with_theta(self.rope_theta as f32)
                .init(device),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    rotary: RotaryEncoding<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .repeat(2, n_rep)
        .reshape([batch, kv_heads * n_rep, seq_len, head_dim])
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();
        let n_rep = self.num_heads / self.num_kv_heads;

        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        let scale = (self.head_dim as f64).sqrt();
        let mask = burn::nn::attention::generate_autoregressive_mask::<B>(batch, seq_len, &device);
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        let probs = softmax(scores, 3);

        let out = probs
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }
}

// --- Decoder block ---

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_position_embeddings: usize,
    #[config(default = "1.0e-5")]
    pub rms_norm_eps: f64,
    #[config(default = "1_000_000.0")]
    pub rope_theta: f64,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock {
            input_norm: RmsNormConfig::new(self.hidden_size)
                .with_eps(self.rms_norm_eps)
                .init(device),
            attention: AttentionConfig::new(
                self.hidden_size,
                self.num_heads,
                self.num_kv_heads,
                self.max_position_embeddings,
            )
            .with_rope_theta(self.rope_theta)
            .init(device),
            post_attention_norm: RmsNormConfig::new(self.hidden_size)
                .with_eps(self.rms_norm_eps)
                .init(device),
            mlp: MlpConfig::new(self.hidden_size, self.intermediate_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub input_norm: RmsNorm<B>,
    pub attention: Attention<B>,
    pub post_attention_norm: RmsNorm<B>,
    pub mlp: Mlp<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.attention.forward(self.input_norm.forward(x)) + residual;
        let residual = x.clone();
        self.mlp.forward(self.post_attention_norm.forward(x)) + residual
    }
}
```

The MoE (Bx-A2) alternative isn't implemented here — see the Appendix for
why this guide points MoE training at the Python toolchain instead.

### Smoke-test the pipeline on one machine

Before committing real hardware-hours, prove the pipeline is wired
correctly end to end on the `smoke` config and a tiny slice of corpus
(re-run the corpus fetch scripts with `PAGES=1` first, if you haven't
already). This is also the one place the pure-CPU `ndarray` backend is
used, so it needs no GPU at all:

```sh
cargo run --release --no-default-features --features ndarray --bin pretrain -- \
  --model-size smoke \
  --output-dir ./checkpoints/smoke \
  --batch-size 2 \
  --grad-accum 1 \
  --max-steps 50 \
  --warmup-steps 5
```
If loss decreases over 50 steps and a checkpoint lands in
`./checkpoints/smoke`, the pipeline is sound — move to the real run.

### Set up training devices

Burn's backend is chosen at **compile time** via the Cargo feature flags
declared above (`wgpu` by default, `ndarray` for the smoke test) — there's
no `--backend` runtime flag, unlike switching devices in PyTorch. Building
with the default `wgpu` feature and running on a machine with one visible
AMD GPU is the primary, fully-supported path for the whole dense
B0.5→B1→B2 progression.

**No FSDP-equivalent exists in Burn today.** PyTorch's Fully Sharded Data
Parallel (and DeepSpeed's ZeRO) let a model's parameters and optimizer
state be *split* across GPUs, so a model bigger than any single card's
memory can still train — the Python toolchain's "Configure distributed
pretraining" section below uses exactly this. Burn has no packaged
equivalent — every training device holds a *complete* copy of the model
and its optimizer state, full stop.

- It's a non-issue for the dense B0.5/B1/B2 path — a ~2B-parameter replica
  is only ~4GB of weights in bf16, comfortably under this guide's 32GB
  R9700 AI PRO example (or even a 24GB card) with plenty of room left for
  optimizer state, so it fits whole on one consumer/workstation card. No
  sharding is needed at this size regardless.
- It's the real constraint on the MoE Bx-A2 alternative — a ~25B-parameter
  MoE model is ~50GB in bf16, too big even for the 32GB R9700 AI PRO to
  hold as a single replica. Without FSDP-style sharding, that model needs
  a single big-memory card (an MI300X's 192GB) or hand-rolled expert
  parallelism Burn doesn't provide — which is why this guide keeps the MoE
  path on the Python toolchain instead of forcing it into Burn.

**If you have more than one GPU**, this is the DIY-est part of the Rust
toolchain: Burn has no `accelerate launch`-style orchestrator either. The
practical workaround, since a B2-class replica fits whole on one card
anyway, is plain independent-replica training with periodic weight
averaging (a crude form of "local SGD"), not true synchronous
data-parallel training with a gradient all-reduce every step:

```sh
HIP_VISIBLE_DEVICES=0 cargo run --release --bin pretrain -- \
  --model-size 2b --output-dir ./checkpoints/base-2b-replica0 &
HIP_VISIBLE_DEVICES=1 cargo run --release --bin pretrain -- \
  --model-size 2b --output-dir ./checkpoints/base-2b-replica1 &
wait
```

Then, every `--save-steps`, average the replicas' checkpoints together
(load each with `CompactRecorder`, arithmetic-mean each corresponding
tensor via `Module::map`, save the result as the next checkpoint for both
replicas to resume from) and let training continue. This is a genuinely
rougher path than FSDP or even ordinary synchronous data-parallel
training — expect more tuning and less predictable convergence than the
single-replica path, and treat multi-GPU as an optimization to reach for
only once the single-card path is proven out, not the default.

### Launch the real pretraining run

A manual training loop, rather than a `Learner`/`TrainStep` setup — Burn's
lower-level `Module`/`Tensor`/`Optimizer` API is more stable across
versions than the higher-level training-loop traits, so this is the more
durable example to copy from:

```rust
// src/data.rs
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use memmap2::Mmap;
use rand::Rng;
use std::fs::File;
use std::path::Path;

const SEQ_LEN: usize = 8192;

pub struct PackedShards {
    mmaps: Vec<Mmap>,
    blocks_per_shard: Vec<usize>,
}

impl PackedShards {
    pub fn open(dir: &Path) -> Self {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("bin"))
            .collect();
        paths.sort();

        let mmaps: Vec<Mmap> = paths
            .iter()
            .map(|p| unsafe { Mmap::map(&File::open(p).unwrap()).unwrap() })
            .collect();
        let blocks_per_shard = mmaps.iter().map(|m| m.len() / (SEQ_LEN * 2)).collect();

        Self {
            mmaps,
            blocks_per_shard,
        }
    }

    pub fn random_batch<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let mut rng = rand::thread_rng();
        let mut batch_ids = Vec::with_capacity(batch_size * SEQ_LEN);

        for _ in 0..batch_size {
            let shard_idx = rng.gen_range(0..self.mmaps.len());
            let block_idx = rng.gen_range(0..self.blocks_per_shard[shard_idx]);
            let start = block_idx * SEQ_LEN * 2;
            let bytes = &self.mmaps[shard_idx][start..start + SEQ_LEN * 2];
            batch_ids.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]) as i32),
            );
        }

        let input_ids: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(batch_ids, [batch_size, SEQ_LEN]),
            device,
        );
        let labels = input_ids.clone();
        (input_ids, labels)
    }
}
```

```rust
// src/bin/pretrain.rs
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use clap::Parser;
use model_build::data::PackedShards;
use model_build::model::{config_for_size, Model};
use std::io::Write;
use std::path::PathBuf;

#[cfg(feature = "ndarray")]
type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray>;
#[cfg(feature = "wgpu")]
type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_size: String,
    #[arg(long, default_value = "./packed_corpus")]
    dataset_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 32)]
    grad_accum: usize,
    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,
    #[arg(long, default_value_t = 500_000)]
    max_steps: usize,
    #[arg(long, default_value_t = 2000)]
    save_steps: usize,
    #[arg(long, default_value_t = 2000)]
    warmup_steps: usize,
    #[arg(long)]
    resume_from: Option<PathBuf>,
}

fn cosine_with_warmup(step: usize, warmup_steps: usize, max_steps: usize, peak_lr: f64) -> f64 {
    if step < warmup_steps {
        return peak_lr * (step as f64 + 1.0) / warmup_steps as f64;
    }
    let progress = (step - warmup_steps) as f64 / (max_steps - warmup_steps).max(1) as f64;
    0.5 * peak_lr * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn main() {
    let args = Args::parse();
    let device = Default::default();

    let config = config_for_size(&args.model_size);
    let mut model: Model<TrainBackend> = config.init(&device);
    if let Some(path) = &args.resume_from {
        let record = CompactRecorder::new()
            .load(path.clone(), &device)
            .expect("checkpoint load failed");
        model = model.load_record(record);
    }

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_beta_2(0.95)
        .init();

    let dataset = PackedShards::open(&args.dataset_dir);
    std::fs::create_dir_all(&args.output_dir).unwrap();
    let mut log = std::fs::File::create(args.output_dir.join("loss.csv")).unwrap();

    for step in 0..args.max_steps {
        let lr = cosine_with_warmup(step, args.warmup_steps, args.max_steps, args.learning_rate);
        let mut step_loss = 0.0f32;

        for _ in 0..args.grad_accum {
            let (input_ids, labels) =
                dataset.random_batch::<TrainBackend>(args.batch_size, &device);
            let logits = model.forward(input_ids);
            let [batch, seq_len, vocab] = logits.dims();
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&device)
                .forward(
                    logits.reshape([batch * seq_len, vocab]),
                    labels.reshape([batch * seq_len]),
                );
            step_loss += loss.clone().into_scalar().elem::<f32>() / args.grad_accum as f32;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            // NOTE: this applies each mini-batch's gradients immediately rather than
            // truly accumulating them first — Burn doesn't (yet) expose a documented
            // GradientsParams add/merge helper. Dividing the learning rate by
            // grad_accum approximates gradient accumulation; check burn::optim's
            // current API for a real accumulation path before relying on this for a
            // long run, since this is an approximation, not the exact PyTorch
            // behavior the Python toolchain below uses.
            model = optimizer.step(lr / args.grad_accum as f64, model, grads);
        }

        writeln!(log, "{step},{step_loss},{lr}").ok();
        if step % args.save_steps == 0 {
            model
                .clone()
                .save_file(
                    args.output_dir.join(format!("checkpoint-{step}")),
                    &CompactRecorder::new(),
                )
                .unwrap();
        }
    }

    model
        .save_file(args.output_dir.join("checkpoint-final"), &CompactRecorder::new())
        .unwrap();
}
```

```sh
cargo run --release --bin pretrain -- \
  --model-size 2b \
  --output-dir ./checkpoints/base-2b \
  --batch-size 8 \
  --grad-accum 32 \
  --learning-rate 3e-4 \
  --max-steps 500000 \
  --save-steps 2000
```
For the B0.5 and B1 stages, run this exact command with only
`--model-size`, `--output-dir`, and `--max-steps` changed, sized to the
smaller token budgets in the reality-check table.

Sizing `--max-steps`: tokens per step = `batch_size × grad_accum × 8192`
(single device, absent the multi-GPU workaround above). The command above
moves ~2.1M tokens/step; to hit a 500B-token target on one card, that's
`5e11 / 2.1e6 ≈ 238,000` steps.

### Monitor, checkpoint, and resume

`pretrain` writes `step,loss,lr` to `<output-dir>/loss.csv` every step —
there's no bundled TensorBoard integration in this manual-loop setup, so
plot it with whatever's convenient (a spreadsheet, `gnuplot`, a Jupyter
cell). Watch for a smooth, monotonic decline — a sudden spike usually
means a bad batch (corrupted shard) or too high a learning rate; a plateau
this early usually means the learning rate is too low or the batch size
too small for this model size. (Burn's higher-level `burn::train::Learner`
API has its own built-in metric dashboard, if you'd rather migrate to that
instead of the manual loop above.)

Resume after a preemption or crash:

```sh
cargo run --release --bin pretrain -- \
  --model-size 2b \
  --output-dir ./checkpoints/base-2b \
  --resume-from ./checkpoints/base-2b/checkpoint-<N>
```

### Export to a Hugging Face–style checkpoint

Both "Evaluate the trained model" and "Convert to GGUF" in General expect
an HF-style `config.json` + `.safetensors` checkpoint directory, not
Burn's native record format — so this small exporter bridges the two, and
gets reused by both.

One detail that matters for correctness: Burn's `Linear` stores its weight
as `[in_features, out_features]` (`forward` computes `input.matmul(weight)`),
while PyTorch's `nn.Linear` — and therefore every tensor GGUF/`transformers`
expect — uses `[out_features, in_features]`. Every weight gets transposed
on the way out, or the converted model is silently wrong:

```rust
// src/safetensors.rs
use std::collections::BTreeMap;
use std::io::Write;

#[derive(serde::Serialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub fn write_safetensors(
    tensors: Vec<(String, Vec<f32>, Vec<usize>)>,
    path: &std::path::Path,
) -> std::io::Result<()> {
    let mut header: BTreeMap<String, TensorInfo> = BTreeMap::new();
    let mut data = Vec::new();
    for (name, values, shape) in &tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            TensorInfo {
                dtype: "F32".to_string(),
                shape: shape.clone(),
                data_offsets: [start, data.len()],
            },
        );
    }
    let header_json = serde_json::to_vec(&header)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&(header_json.len() as u64).to_le_bytes())?;
    file.write_all(&header_json)?;
    file.write_all(&data)?;
    Ok(())
}
```

```rust
// src/export.rs
use crate::model::{Model, ModelConfig};
use crate::safetensors::write_safetensors;
use burn::module::Param;
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::path::Path;

fn tensor_to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> (Vec<f32>, Vec<usize>) {
    let shape = t.dims().to_vec();
    (t.into_data().convert::<f32>().to_vec().unwrap(), shape)
}

fn linear_weight_hf<B: Backend>(linear: &Linear<B>) -> (Vec<f32>, Vec<usize>) {
    // Transpose: Burn's [in, out] -> PyTorch/HF's [out, in].
    tensor_to_vec(linear.weight.val().transpose())
}

fn norm_weight_hf<B: Backend>(weight: &Param<Tensor<B, 1>>) -> (Vec<f32>, Vec<usize>) {
    tensor_to_vec(weight.val())
}

pub fn to_hf_checkpoint<B: Backend>(model: &Model<B>, config: &ModelConfig, out_dir: &Path) {
    let mut tensors = Vec::new();

    let (v, s) = tensor_to_vec(model.embed_tokens.weight.val());
    tensors.push(("model.embed_tokens.weight".to_string(), v, s));

    for (i, layer) in model.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        let (v, s) = linear_weight_hf(&layer.attention.q_proj);
        tensors.push((format!("{p}.self_attn.q_proj.weight"), v, s));
        let (v, s) = linear_weight_hf(&layer.attention.k_proj);
        tensors.push((format!("{p}.self_attn.k_proj.weight"), v, s));
        let (v, s) = linear_weight_hf(&layer.attention.v_proj);
        tensors.push((format!("{p}.self_attn.v_proj.weight"), v, s));
        let (v, s) = linear_weight_hf(&layer.attention.o_proj);
        tensors.push((format!("{p}.self_attn.o_proj.weight"), v, s));
        let (v, s) = norm_weight_hf(&layer.input_norm.weight);
        tensors.push((format!("{p}.input_layernorm.weight"), v, s));
        let (v, s) = norm_weight_hf(&layer.post_attention_norm.weight);
        tensors.push((format!("{p}.post_attention_layernorm.weight"), v, s));
        let (v, s) = linear_weight_hf(&layer.mlp.gate_proj);
        tensors.push((format!("{p}.mlp.gate_proj.weight"), v, s));
        let (v, s) = linear_weight_hf(&layer.mlp.up_proj);
        tensors.push((format!("{p}.mlp.up_proj.weight"), v, s));
        let (v, s) = linear_weight_hf(&layer.mlp.down_proj);
        tensors.push((format!("{p}.mlp.down_proj.weight"), v, s));
    }

    let (v, s) = norm_weight_hf(&model.norm.weight);
    tensors.push(("model.norm.weight".to_string(), v, s));
    let (v, s) = linear_weight_hf(&model.lm_head);
    tensors.push(("lm_head.weight".to_string(), v, s));

    write_safetensors(tensors, &out_dir.join("model.safetensors")).unwrap();

    let config_json = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
    });
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();
}
```

`tokenizer.json`, the file produced above, is already HF- and
llama.cpp-compatible as-is (both `transformers`' `AutoTokenizer` and
llama.cpp's `convert_hf_to_gguf.py` load a bare `tokenizer.json` directly)
— just copy it alongside:

```rust
// src/bin/export_hf.rs
use burn::record::{CompactRecorder, Recorder};
use clap::Parser;
use model_build::model::{config_for_size, Model};
use std::path::PathBuf;

#[cfg(feature = "wgpu")]
type EvalBackend = burn::backend::Wgpu;
#[cfg(feature = "ndarray")]
type EvalBackend = burn::backend::NdArray;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_size: String,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
}

fn main() {
    let args = Args::parse();
    let device = Default::default();
    let config = config_for_size(&args.model_size);

    let record = CompactRecorder::new()
        .load(args.checkpoint, &device)
        .expect("checkpoint load failed");
    let model: Model<EvalBackend> = config.init(&device).load_record(record);

    std::fs::create_dir_all(&args.out_dir).unwrap();
    model_build::export::to_hf_checkpoint(&model, &config, &args.out_dir);
    std::fs::copy("tokenizer.json", args.out_dir.join("tokenizer.json")).unwrap();
}
```

```sh
cargo run --release --bin export_hf -- \
  --model-size 2b \
  --checkpoint ./checkpoints/base-2b/checkpoint-final \
  --out-dir ./hf-export/base-2b
```

Feed `./hf-export/base-2b` (or the SFT/LoRA checkpoint's export, below)
into General's "Evaluate the trained model" and "Convert to GGUF" steps.

### Train the model even more

**Continued pretraining.** The cheapest way to make the model better:
resume "Launch the real pretraining run" with a larger `--max-steps` (or a
fresh corpus pull — re-run the fetch scripts periodically to pick up new
repos and a newer web crawl) and a short LR re-warmup before the cosine
schedule continues, rather than starting over.

**Chat formatting.** The `<|im_start|>`/`<|im_end|>` tokens reserved
earlier just need a plain string template — no `tokenizer.chat_template`
object to set, since that's a `transformers`-specific concept:

```rust
pub fn format_chatml(role: &str, content: &str) -> String {
    format!("<|im_start|>{role}\n{content}<|im_end|>\n")
}
```

**Supervised fine-tuning (SFT).** Turns the base model into something that
follows instructions — a mix of code-instruction data and general-English
chat data. Fetch both the same way the corpus fetch step above did (curl
against the Hub's file-listing API, no `datasets` library involved):

```sh
#!/usr/bin/env bash
# scripts/fetch_sft_data.sh
set -euo pipefail
mkdir -p sft_data/code-instruct sft_data/general-instruct

fetch_dataset() {
  local repo="$1" subdir="$2" dest="$3"
  curl -s "https://huggingface.co/api/datasets/${repo}/tree/main/${subdir}" \
    | python3 -c "import sys,json;print('\n'.join(f['path'] for f in json.load(sys.stdin) if f['path'].endswith('.parquet')))" \
    | while read -r path; do
        curl -L -o "${dest}/$(basename "$path")" \
          "https://huggingface.co/datasets/${repo}/resolve/main/${path}"
      done
}

fetch_dataset bigcode/self-oss-instruct-sc2-exec-filter-50k data sft_data/code-instruct
fetch_dataset HuggingFaceH4/ultrachat_200k data sft_data/general-instruct
```

Reading these back in Rust reuses the Parquet path from `src/corpus.rs`
(same `polars` scan pattern), formatting each example with
`format_chatml`. The one real addition needed over the pretraining packer
above is a **loss mask**: SFT should only backpropagate through the
assistant's tokens, not the prompt. Extend the packing format with a
second `u8` array alongside each shard's token `u16` array — one byte per
token, `1` where it counts toward the loss (the assistant turn) and `0`
elsewhere (computed by tokenizing the prompt-only prefix first to find its
length, then marking everything from there to the matching `<|im_end|>`)
— and have `pretrain`'s loss computation multiply per-token loss by this
mask before averaging, instead of the unmasked `CrossEntropyLossConfig`
call used for pretraining. Point a second `pretrain`-like binary (`sft`) at
`./sft_data`'s packed shards, initialized from
`./checkpoints/base-2b/checkpoint-final` instead of random weights, with a
much lower peak learning rate (`2e-5` instead of `3e-4`) and a handful of
epochs instead of a token-count budget.

**Preference optimization (optional).** If you want responses tuned
toward a preferred style (e.g. terser code, fewer unnecessary comments),
preference optimization (DPO-style: a second loss term comparing a
preferred vs. rejected response) is a refinement on top of the SFT
checkpoint, not a requirement — skip it if the SFT model already behaves
well. There's no packaged Burn equivalent of `trl`'s `DPOTrainer` either,
so pursuing this means writing the paired-comparison loss by hand against
the same manual training loop above — real, bounded work, but roll-your-own
rather than off-the-shelf.

**Lighter-weight continued fine-tuning: LoRA.** A B2 model's weights are
only ~4GB in bf16/f32, so unlike a 4B+ model there's no need for 4-bit
(QLoRA) quantization at all — plain full-precision LoRA already fits
comfortably on a single 24GB-and-up card, including this guide's R9700 AI
PRO example with room to spare. There's no `peft` equivalent in the
Rust ecosystem, so the low-rank adapter is a small, hand-written wrapper
around `Linear`:

```rust
// src/model.rs (additional module)
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    base: Linear<B>, // frozen — see note below on freezing parameters
    lora_a: Param<Tensor<B, 2>>, // [in_features, r]
    lora_b: Param<Tensor<B, 2>>, // [r, out_features]
    scale: f64,
}

impl<B: Backend> LoraLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let base_out = self.base.forward(x.clone());
        let lora_out = x
            .matmul(self.lora_a.val().unsqueeze())
            .matmul(self.lora_b.val().unsqueeze())
            * self.scale;
        base_out + lora_out
    }
}
```

Swap this in for `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/
`down_proj` on a loaded checkpoint, freeze `base` (check Burn's current
docs for the exact parameter-freezing mechanism — `Param` supports marking
a tensor as not requiring gradients, but the precise method name has moved
across versions), and train only `lora_a`/`lora_b` with the same manual
loop above, at a much smaller memory footprint since gradients and
optimizer state are needed only for the low-rank matrices.

## Python toolchain

The mature, battle-tested path: `transformers` ships the Llama and
Mixtral architectures ready to instantiate, `accelerate` gives real FSDP,
and `trl`/`peft` cover SFT/DPO/LoRA — at the cost of a Python environment
and a large dependency stack alongside orangu's own single-binary Rust
build.

### Prerequisites

- A Linux machine (or cluster) with an AMD GPU and ROCm installed for
  anything past the smoke test. This guide's example is a single 32GB
  **Radeon AI PRO R9700** — enough headroom to run the whole dense
  B0.5→B1→B2 progression comfortably; datacenter Instinct cards
  (MI300X/MI325X) are the realistic floor once you're pursuing the MoE
  alternative instead, where `accelerate`'s FSDP support (below) can
  actually shard the model across several. See "Choosing an AMD card" in
  General for the broader lineup.
- Confirm ROCm sees the card(s) before doing anything else:

  ```sh
  rocminfo | grep -i gfx
  rocm-smi
  ```

```sh
python3 -m venv ~/model-build/venv
source ~/model-build/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install "transformers>=4.46" accelerate tokenizers trl peft \
            numpy pyarrow wikiextractor
```
No `flash-attn` install is needed: recent `transformers` already defaults
to PyTorch's built-in `scaled_dot_product_attention`, which runs on ROCm
via MIOpen/composable-kernel without any extra package. ROCm's PyTorch
build also keeps the `torch.cuda`/`cuda:N` device API for compatibility —
every `cuda:0`-style flag later in this section is correct as written on
AMD.

### Train a tokenizer on the fetched corpus

A shared helper reads text out of every corpus directory regardless of its
format (raw source files, wikiextractor's JSON-lines, or Parquet), so both
the tokenizer trainer and the packer below use the same code:

```python
# corpus_iter.py
import json
from pathlib import Path

CODE_EXTENSIONS = {
    "c": {".c", ".h"},
    "rust": {".rs"},
    "java": {".java"},
    "typescript": {".ts", ".tsx"},
    "bash": {".sh", ".bash"},
}

def iter_code_texts(corpus_dir: Path):
    for lang, extensions in CODE_EXTENSIONS.items():
        lang_dir = corpus_dir / lang
        for ext in extensions:
            for path in lang_dir.rglob(f"*{ext}"):
                try:
                    yield path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

def iter_wikipedia_texts(corpus_dir: Path):
    extracted = corpus_dir / "english-wikipedia" / "extracted"
    for path in extracted.rglob("wiki_*"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("text"):
                    yield obj["text"]

def iter_fineweb_texts(corpus_dir: Path):
    import pyarrow.parquet as pq
    for path in sorted((corpus_dir / "english-fineweb").glob("*.parquet")):
        table = pq.read_table(path, columns=["text"])
        for text in table.column("text"):
            yield str(text)

def iter_corpus_texts(corpus_dir: Path):
    yield from iter_code_texts(corpus_dir)
    yield from iter_wikipedia_texts(corpus_dir)
    yield from iter_fineweb_texts(corpus_dir)
```

```python
# train_tokenizer.py
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
from corpus_iter import iter_corpus_texts

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32768,   # a 5-language + English vocab needs far less than a
                         # 100+ language general tokenizer's ~150K entries
    special_tokens=[
        "<|endoftext|>", "<|pad|>",
        "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<fim_pad>",  # fill-in-the-middle, useful for code completion
        "<|im_start|>", "<|im_end|>",                                  # reserved for the chat template below
    ],
)

tokenizer.train_from_iterator(iter_corpus_texts(Path("corpus")), trainer=trainer)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
)
fast_tokenizer.save_pretrained("./tokenizer")
```

```sh
python3 train_tokenizer.py
```

### Tokenize and pack the corpus into shards

Concatenate every document's tokens (with an `<|endoftext|>` boundary
between documents), and slice the stream into fixed-length blocks — the
standard packing scheme for causal-LM pretraining, so the model never
wastes a forward pass on padding. Blocks are written as `uint16` arrays
(the 32,768-entry vocabulary fits comfortably) in ~100MB shards:

```python
# pack_dataset.py
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from corpus_iter import iter_corpus_texts

SEQ_LEN = 8192
TOKENS_PER_SHARD = 50_000_000

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
eos_id = tokenizer.eos_token_id

Path("packed_corpus").mkdir(exist_ok=True)

def flush(tokens, idx):
    arr = np.array(tokens, dtype=np.uint16)
    n_blocks = len(arr) // SEQ_LEN
    arr = arr[: n_blocks * SEQ_LEN].reshape(n_blocks, SEQ_LEN)
    np.save(f"packed_corpus/shard_{idx:05d}.npy", arr)

buffer, shard_idx = [], 0
for text in iter_corpus_texts(Path("corpus")):
    buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    buffer.append(eos_id)
    if len(buffer) >= TOKENS_PER_SHARD:
        flush(buffer, shard_idx)
        shard_idx += 1
        buffer = []

if len(buffer) >= SEQ_LEN:
    flush(buffer, shard_idx)
```

```sh
python3 pack_dataset.py
```

### Define and instantiate the architecture

Matches the size table under "Define the target architecture" in General.
`dense_2b_config.json`, the final dense target:

```json
{
  "architecture": "llama",
  "vocab_size": 32768,
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 30,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "max_position_embeddings": 8192,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-5
}
```

`dense_0.5b_config.json` and `dense_1b_config.json` are the same shape
with the four sizing fields swapped per the General table. Unlike the
Rust toolchain, the Python toolchain **is** the recommended path for the
MoE alternative — `MixtralConfig`/`MixtralForCausalLM` are ready-made in
`transformers`, and `accelerate`'s FSDP support (below) can actually
shard the ~25B-parameter result across multiple cards:

```json
{
  "architecture": "mixtral",
  "vocab_size": 32768,
  "hidden_size": 2048,
  "intermediate_size": 2048,
  "num_hidden_layers": 30,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "num_local_experts": 64,
  "num_experts_per_tok": 4,
  "max_position_embeddings": 8192,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-5,
  "router_aux_loss_coef": 0.02
}
```
Save as `moe_bx_a2_config.json`.

### Smoke-test the pipeline on one machine

Before committing real hardware-hours, prove the pipeline is wired
correctly end to end on a tiny model and a tiny slice of corpus (re-run
the corpus fetch scripts with `PAGES=1` first, if you haven't already):

```json
{
  "architecture": "llama",
  "vocab_size": 32768,
  "hidden_size": 256,
  "intermediate_size": 688,
  "num_hidden_layers": 4,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "max_position_embeddings": 512
}
```
Save as `smoke_config.json`, then use `pretrain.py` below directly (no
`accelerate`, no distributed setup needed for this):

```sh
python3 pretrain.py \
  --model_config smoke_config.json \
  --tokenizer_dir ./tokenizer \
  --dataset_dir ./packed_corpus \
  --output_dir ./checkpoints/smoke \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_steps 50 \
  --warmup_steps 5
```
If loss decreases over 50 steps and a checkpoint lands in
`./checkpoints/smoke`, the pipeline is sound — move to the real run.

### Configure distributed pretraining

`accelerate_fsdp.yaml` — Fully Sharded Data Parallel, so an 80GB card can
hold its shard of a multi-billion-parameter model plus optimizer state.
This is real parameter sharding across GPUs, unlike anything available in
the Rust toolchain today (see the Appendix):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer   # MixtralDecoderLayer for the MoE config
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
mixed_precision: bf16
num_machines: 1
num_processes: 8
main_process_port: 29500
```
For multiple nodes, copy this file to each node and set `num_machines`,
`machine_rank`, and `main_process_ip` accordingly (see the `accelerate`
docs for the full multi-node checklist).

### Launch the real pretraining run

```python
# pretrain.py
import argparse, json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer, LlamaConfig, LlamaForCausalLM,
    MixtralConfig, MixtralForCausalLM,
    Trainer, TrainingArguments, default_data_collator,
)


class PackedShards(torch.utils.data.Dataset):
    def __init__(self, shard_dir):
        self.shards = [np.load(p, mmap_mode="r") for p in sorted(Path(shard_dir).glob("shard_*.npy"))]
        self.lengths = [len(s) for s in self.shards]
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self):
        return int(self.cumulative[-1]) if self.shards else 0

    def __getitem__(self, idx):
        shard_i = int(np.searchsorted(self.cumulative, idx, side="right"))
        local_i = idx - (self.cumulative[shard_i - 1] if shard_i > 0 else 0)
        input_ids = torch.tensor(self.shards[shard_i][local_i].astype(np.int64))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def build_model(config_path, tokenizer):
    cfg = json.loads(Path(config_path).read_text())
    arch = cfg.pop("architecture", "llama")
    if arch == "mixtral":
        model = MixtralForCausalLM(MixtralConfig(**cfg))
    else:
        model = LlamaForCausalLM(LlamaConfig(**cfg))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = build_model(args.model_config, tokenizer)
    dataset = PackedShards(args.dataset_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=20,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused",
        weight_decay=0.1,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
```

```sh
accelerate launch --config_file accelerate_fsdp.yaml pretrain.py \
  --model_config dense_2b_config.json \
  --tokenizer_dir ./tokenizer \
  --dataset_dir ./packed_corpus \
  --output_dir ./checkpoints/base-2b \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --learning_rate 3e-4 \
  --max_steps 500000 \
  --save_steps 2000 \
  --bf16 --gradient_checkpointing
```
A 2B model's weights and optimizer state are small enough that this batch
size is comfortable even before FSDP shards it further — push it higher if
your cards have room. For the B0.5 and B1 stages, run this exact command
with only `--model_config`, `--output_dir`, and `--max_steps` changed
(e.g. `dense_0.5b_config.json` → `./checkpoints/base-0.5b`, sized to the
smaller token budgets in the General reality check) — everything else
about the pipeline is identical.

Sizing `--max_steps`: global tokens per step = `per_device_batch_size ×
grad_accum × num_gpus × 8192`. The command above moves ~16.8M tokens/step;
to hit a 500B-token target, that's `5e11 / 16.8e6 ≈ 29,800` steps — adjust
`--max_steps` (and batch/accumulation, if you have more or fewer GPUs) to
your own token budget.

### Monitor, checkpoint, and resume

```sh
tensorboard --logdir ./checkpoints/base-2b/runs
```
Watch the training loss for a smooth, monotonic decline — a sudden spike
usually means a bad batch (corrupted shard) or too high a learning rate; a
plateau this early usually means the learning rate is too low or the batch
size too small for this model size.

Resume after a preemption or crash:

```sh
accelerate launch --config_file accelerate_fsdp.yaml pretrain.py \
  --model_config dense_2b_config.json \
  --tokenizer_dir ./tokenizer --dataset_dir ./packed_corpus \
  --output_dir ./checkpoints/base-2b \
  --resume_from_checkpoint ./checkpoints/base-2b/checkpoint-<N> \
  --bf16 --gradient_checkpointing
```

### Train the model even more

**Continued pretraining.** The cheapest way to make the model better:
resume "Launch the real pretraining run" with a larger `--max_steps` (or a
fresh corpus pull — re-run the fetch scripts periodically to pick up new
repos and a newer web crawl) and a short LR re-warmup before the cosine
schedule continues, rather than starting over.

**Chat template.** The `<|im_start|>`/`<|im_end|>` tokens reserved earlier
need a template before instruction tuning can use them:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/base-2b")
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
tokenizer.save_pretrained("./checkpoints/base-2b")
```

**Supervised fine-tuning (SFT).** Turns the base model into something
that follows instructions — a mix of code-instruction data and
general-English chat data, via TRL:

```python
# sft.py
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_path = "./checkpoints/base-2b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

code_instruct = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train")
general_instruct = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

def to_chatml(example):
    messages = example.get("messages") or [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = concatenate_datasets([
    code_instruct.map(to_chatml, remove_columns=code_instruct.column_names),
    general_instruct.map(to_chatml, remove_columns=general_instruct.column_names),
])

config = SFTConfig(
    output_dir="./checkpoints/base-2b-sft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=8192,
    dataset_text_field="text",
    packing=True,
)

trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
trainer.save_model("./checkpoints/base-2b-sft")
```

```sh
accelerate launch --config_file accelerate_fsdp.yaml sft.py
```

The result, `./checkpoints/base-2b-sft`, is already an HF-style
`config.json` + `.safetensors` directory — feed it straight into
General's "Evaluate the trained model" and "Convert to GGUF" steps, no
separate export needed (unlike the Rust toolchain).

**Preference optimization (optional).** If you want responses tuned
toward a preferred style (e.g. terser code, fewer unnecessary comments),
run `trl`'s `DPOTrainer` on a preference-pairs dataset against the SFT
checkpoint. This is a refinement, not a requirement — skip it if the SFT
model already behaves well.

**Lighter-weight continued fine-tuning: LoRA.** A B2 model's weights are
only ~4GB in bf16, so unlike a 4B+ model there's no need to reach for
4-bit (QLoRA) quantization here at all — plain bf16 LoRA already fits
comfortably on a single 24GB-and-up card, including this guide's R9700 AI
PRO example, which conveniently sidesteps `bitsandbytes`'s ROCm support
being less mature than its CUDA support:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/base-2b-sft", torch_dtype="bfloat16", device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# ... train with the same SFTTrainer as above, using this `model` ...
model = model.merge_and_unload()
model.save_pretrained("./checkpoints/base-2b-sft-lora-merged")
```

## Appendix

### Rust

**Pros**
- Matches orangu's own "single Rust binary, no runtime" philosophy — no
  Python environment needed to train (General's evaluation step is the
  one unavoidable exception, regardless of toolchain).
- The `tokenizers` crate is the same native-Rust library the Python
  `tokenizers` package binds to — no fidelity lost switching to it.
- Backend flexibility via Burn: the same model code runs on Vulkan (`wgpu`)
  or CPU (`ndarray`) by swapping a Cargo feature — useful for the
  CPU-only smoke test, and for staying on hardware ROCm doesn't officially
  support.
- Full control: every part of the model (attention, RoPE, norm, MLP) is
  explicit, hand-written code, nothing hidden inside a framework.

**Cons**
- No FSDP/ZeRO-style parameter sharding — every device holds a full model
  replica; multi-GPU means independent replicas with manual weight
  averaging, not true synchronous data-parallel training.
- No packaged Llama/Mixtral training architecture — GQA, RoPE, RMSNorm,
  and SwiGLU are all hand-rolled, and Burn's tensor API has shifted across
  releases, so exact method names may need adjusting for your pinned
  version.
- No MoE routing/batched-expert-execution primitives, and the FSDP gap
  above makes the ~25B-parameter MoE alternative impractical without a
  single big-memory (MI300X-class) card — this guide recommends the
  Python toolchain for the MoE variant instead.
- No `peft`/`trl` equivalents — LoRA, SFT loss-masking, and any DPO-style
  preference loss are all hand-written rather than a maintained library
  call.
- A trained checkpoint isn't directly usable by the (Python-only)
  evaluation harnesses or `convert_hf_to_gguf.py` — needs an extra
  export-to-safetensors step first.
- Ecosystem is younger and smaller — fewer worked examples, more version
  churn, less community troubleshooting to lean on than PyTorch's.

### Python

**Pros**
- Mature, battle-tested ecosystem: `transformers` ships the Llama and
  Mixtral architectures ready to instantiate, no hand-written
  attention/MLP code.
- Real FSDP support via `accelerate` — parameters and optimizer state
  genuinely shard across GPUs, so the ~25B-parameter MoE alternative is
  practical on ordinary multi-GPU hardware, not just a single huge card.
- `trl`/`peft` provide maintained SFT/DPO/LoRA implementations instead of
  hand-rolled ones.
- The trained checkpoint is already in the exact HF-style format the
  evaluation harnesses and `convert_hf_to_gguf.py` expect — no extra
  export step.
- A much larger body of documentation, examples, and community
  troubleshooting for any issue you hit.

**Cons**
- Requires a Python environment and a large dependency stack (`torch`,
  `transformers`, `accelerate`, `trl`, `peft`, ...) alongside — or instead
  of — orangu's own single-binary Rust build.
- Less aligned with this project's own "100% Rust, no runtime"
  philosophy — if that matters to you as a design goal for your own
  tooling around the model, it's a real mismatch.
- Python/pip dependency management (version pins, ROCm-specific wheel
  indices) is its own recurring maintenance surface.

### Performance

No specific published benchmark compares this exact matchup — Burn
training a ~2B-parameter Llama-style model against `transformers` doing
the same, on AMD ROCm. Burn's GPU backends are new enough that this
comparison doesn't appear to exist publicly yet, and it isn't something
this guide benchmarks itself either: doing so honestly would need real
ROCm hardware with both full stacks built, which is out of scope here.
What follows is reasoned from well-established facts about each
ecosystem's maturity, not a measured number — a starting point for your
own thinking, not a citation.

**Where Rust plausibly wins:**
- **Process/startup overhead.** A `cargo run --release` binary starts in
  milliseconds; `python3 pretrain.py` pays the cost of importing `torch`
  and `transformers` (multiple seconds) on every invocation. Negligible
  across one six-week run, but adds up across many short smoke-test
  iterations while debugging.
- **CPU-bound corpus/data-pipeline work.** No GIL, no per-object Python
  overhead, for the corpus-fetching, tokenizing, and packing steps —
  though the actual tokenization in both toolchains dispatches into the
  *same* Rust `tokenizers` crate either way, so the difference is in the
  surrounding glue code, not the tokenizer itself.
- **The training loop's batch-fetching hot path.** Rust's memmap-based
  random-access reads have essentially zero framework overhead, versus
  Python's `DataLoader`/multiprocessing machinery. This matters
  proportionally more at smaller model sizes (B0.5), where each step's
  GPU compute is fast enough that data-loading overhead is a bigger share
  of total step time; less at B2, where GPU compute dominates every step
  regardless.

**Where Python almost certainly wins — the part that actually decides
total wall-clock time:**
- **GPU kernel throughput.** Training cost is almost entirely GPU matmul
  and attention kernels, not host-language dispatch. PyTorch calls into
  rocBLAS/MIOpen/hipBLASLt — libraries AMD has tuned for years
  specifically for these workloads. Burn's `wgpu` backend compiles
  through a much younger WGSL/Vulkan code-generation pipeline, and its
  `cubecl`-based backends are newer still, without the same
  battle-tested ROCm/HIP path PyTorch has (see "Set up training devices"
  above). A gap of even 20–50% between a decade-tuned vendor kernel and a
  newer generic one would be unsurprising, and would swing a training run
  from six weeks to nine.

**The practical upshot:** if what you care about is *time to a trained B2
model*, expect the Python toolchain to be faster today, not the Rust one
— precisely because the expensive part of training is GPU kernels calling
into mature vendor libraries, and Rust's advantages here live around the
edges (corpus prep, tokenizer training, process startup), not in the
number that actually determines how many weeks the run takes.

If you need real numbers instead of this reasoning, the only trustworthy
source is a benchmark on your own target hardware: run a fixed number of
steps of the `smoke` or `0.5b` config in both toolchains back-to-back and
compare tokens/sec — nothing published, and nothing this guide could run
without real ROCm hardware to test on, substitutes for that.

### Licenses

Restricting the corpus-discovery step to `MIT`/`Apache-2.0`/`BSD-3-Clause`
repos keeps the corpus free of copyleft obligations, but license metadata
on GitHub is self-reported and occasionally wrong or missing at the file
level (vendored code, dual-licensed subdirectories). Review the
`*.repos.txt` lists that step produces before a large run.
