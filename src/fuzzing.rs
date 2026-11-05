//! A small mutation fuzzer, for the parsers that take input from outside.
//!
//! It runs as ordinary tests, on the stable toolchain, with no dependencies. A
//! plain `cargo test` gives each target a few hundred inputs, and a longer
//! campaign is an environment variable away:
//!
//! ```sh
//! ORANGU_FUZZ_SECONDS=300 ORANGU_FUZZ_OUT=/tmp/fuzz \
//!     cargo test --bin orangu-server fuzz_ -- --test-threads=1
//! ```
//!
//! Debug builds are the ones to run, as their overflow checks turn arithmetic
//! on untrusted numbers into findings. Runs repeat exactly for one
//! `ORANGU_FUZZ_SEED`.
//!
//! It is not coverage-guided: it mutates its seeds blindly, so it finds what
//! good seeds and a dictionary lead it to, and misses bugs that need a path
//! found step by step.
//!
//! A panic fails the test once a campaign ends, with the input saved. What ends
//! the process instead cannot be caught, so an input that hangs past
//! `ORANGU_FUZZ_INPUT_SECONDS` aborts it from a watchdog, and under
//! `ORANGU_FUZZ_OUT` each input is written to `<target>.current` before it runs,
//! which is what is left after a stack overflow or `ulimit -v` running out.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

/// Inputs a target gets in an ordinary test run.
const QUICK_INPUTS: u64 = 300;

/// Numbers that make length fields, counts and offsets misbehave.
const INTERESTING: &[u64] = &[
    0,
    1,
    2,
    7,
    8,
    16,
    31,
    32,
    64,
    255,
    256,
    4096,
    65_535,
    65_536,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x1_0000_0000,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    u64::MAX - 1,
    u64::MAX,
];

/// A deterministic generator: SplitMix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `n`, which must not be zero.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// One random change to `input`.
fn mutate_once(input: &mut Vec<u8>, seeds: &[Vec<u8>], dictionary: &[&[u8]], rng: &mut Rng) {
    let len = input.len();
    match rng.below(10) {
        0 if len > 0 => {
            let at = rng.below(len);
            input[at] ^= 1 << rng.below(8);
        }
        1 if len > 0 => {
            let at = rng.below(len);
            input[at] = rng.next() as u8;
        }
        // A number over whatever is there, in each width a format stores one.
        2 | 3 if len > 0 => {
            let value = INTERESTING[rng.below(INTERESTING.len())];
            let width = [1usize, 2, 4, 8][rng.below(4)];
            let at = rng.below(len);
            let bytes = value.to_le_bytes();
            let end = (at + width).min(len);
            input[at..end].copy_from_slice(&bytes[..end - at]);
        }
        4 | 5 if !dictionary.is_empty() => {
            let word = dictionary[rng.below(dictionary.len())];
            let at = rng.below(len + 1);
            input.splice(at..at, word.iter().copied());
        }
        6 if len > 0 => {
            let start = rng.below(len);
            let end = (start + 1 + rng.below(64)).min(len);
            input.drain(start..end);
        }
        // A stretch repeated many times, for what grows with nesting or length.
        7 if len > 0 => {
            let start = rng.below(len);
            let end = (start + 1 + rng.below(32)).min(len);
            let stretch = input[start..end].to_vec();
            let most = if rng.below(16) == 0 { 2000 } else { 8 };
            let times = 1 + rng.below(most);
            let repeated: Vec<u8> = std::iter::repeat_n(stretch, times).flatten().collect();
            input.splice(start..start, repeated);
        }
        8 if !seeds.is_empty() => {
            let other = &seeds[rng.below(seeds.len())];
            if !other.is_empty() {
                let from = rng.below(other.len());
                let to = from + rng.below(other.len() - from + 1);
                let at = rng.below(len + 1);
                input.splice(at..at, other[from..to].iter().copied());
            }
        }
        9 if len > 0 => input.truncate(rng.below(len)),
        _ => input.push(rng.next() as u8),
    }
}

thread_local! {
    /// Set while a fuzz target runs on this thread: where its panic came from.
    static PANIC_AT: std::cell::RefCell<Option<Option<String>>> = const {
        std::cell::RefCell::new(None)
    };
}

fn install_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let caught = PANIC_AT.with(|slot| {
                let mut slot = slot.borrow_mut();
                let Some(at) = slot.as_mut() else {
                    return false;
                };
                *at = Some(info.location().map_or_else(
                    || "unknown".to_string(),
                    |l| format!("{}:{}", l.file(), l.line()),
                ));
                true
            });
            if !caught {
                previous(info);
            }
        }));
    });
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

/// A panic a campaign caught: where it was, how often, and the shortest input.
#[derive(Default)]
struct Finding {
    count: u64,
    shortest: Vec<u8>,
}

/// Mutates `seeds` and runs `target` over each result, until the campaign's
/// budget runs out, then panics if anything did.
pub fn fuzz(
    name: &str,
    seeds: Vec<Vec<u8>>,
    dictionary: &[&[u8]],
    max_len: usize,
    target: impl Fn(&[u8]),
) {
    install_panic_hook();
    assert!(!seeds.is_empty(), "{name} has no seeds");
    let seconds = env_u64("ORANGU_FUZZ_SECONDS");
    let deadline = seconds.map(|s| Instant::now() + Duration::from_secs(s));
    let out = std::env::var_os("ORANGU_FUZZ_OUT").map(PathBuf::from);
    if let Some(dir) = &out {
        let _ = std::fs::create_dir_all(dir);
    }
    let current_path = out.as_ref().map(|dir| dir.join(format!("{name}.current")));
    let per_input = Duration::from_secs(env_u64("ORANGU_FUZZ_INPUT_SECONDS").unwrap_or(20));
    let mut rng = Rng::new(env_u64("ORANGU_FUZZ_SEED").unwrap_or(1) ^ hash(name));

    // The watchdog: the one way out of a target that never returns.
    let started_at = Arc::new(AtomicU64::new(0));
    let current = Arc::new(Mutex::new(Vec::<u8>::new()));
    let finished = Arc::new(AtomicBool::new(false));
    let origin = Instant::now();
    let watchdog = {
        let (started_at, current, finished) =
            (started_at.clone(), current.clone(), finished.clone());
        let (name, out) = (name.to_string(), out.clone());
        std::thread::spawn(move || {
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(100));
                let started = started_at.load(Ordering::Relaxed);
                if started == 0 {
                    continue;
                }
                let elapsed = origin.elapsed().as_millis() as u64 - started;
                if elapsed > per_input.as_millis() as u64 {
                    let input = current.lock().map(|c| c.clone()).unwrap_or_default();
                    let dir = out.unwrap_or_else(std::env::temp_dir);
                    let path = dir.join(format!("{name}.hang"));
                    let _ = std::fs::write(&path, &input);
                    eprintln!(
                        "fuzz {name}: an input of {} bytes ran past {elapsed} ms: {}",
                        input.len(),
                        path.display()
                    );
                    std::process::abort();
                }
            }
        })
    };

    let mut findings: BTreeMap<String, Finding> = BTreeMap::new();
    let mut inputs = 0u64;
    loop {
        match deadline {
            Some(deadline) if Instant::now() >= deadline => break,
            None if inputs >= QUICK_INPUTS => break,
            _ => {}
        }
        // Every seed runs as it is first: they have to work to be worth mutating.
        let mut input = seeds[rng.below(seeds.len())].clone();
        if inputs >= seeds.len() as u64 {
            for _ in 0..1 + rng.below(4) {
                mutate_once(&mut input, &seeds, dictionary, &mut rng);
            }
            input.truncate(max_len);
        } else {
            input.clone_from(&seeds[inputs as usize]);
        }
        inputs += 1;
        if let Some(path) = &current_path {
            let _ = std::fs::write(path, &input);
        }
        if let Ok(mut slot) = current.lock() {
            slot.clone_from(&input);
        }
        PANIC_AT.with(|slot| *slot.borrow_mut() = Some(None));
        started_at.store(
            (origin.elapsed().as_millis() as u64).max(1),
            Ordering::Relaxed,
        );
        let result = catch_unwind(AssertUnwindSafe(|| target(&input)));
        started_at.store(0, Ordering::Relaxed);
        let at = PANIC_AT.with(|slot| slot.borrow_mut().take().flatten());
        if result.is_err() {
            let finding = findings.entry(at.unwrap_or_default()).or_default();
            finding.count += 1;
            if finding.count == 1 || input.len() < finding.shortest.len() {
                finding.shortest = input;
            }
        }
    }
    finished.store(true, Ordering::Relaxed);
    let _ = watchdog.join();
    if let Some(path) = &current_path {
        let _ = std::fs::remove_file(path);
    }
    if deadline.is_some() {
        println!("fuzz {name}: {inputs} inputs, {} findings", findings.len());
    }
    if findings.is_empty() {
        return;
    }
    let dir = out.unwrap_or_else(std::env::temp_dir);
    let mut report = format!(
        "fuzz {name}: {} kinds of panic in {inputs} inputs",
        findings.len()
    );
    for (index, (at, finding)) in findings.iter().enumerate() {
        let path = dir.join(format!("{name}.panic{index}"));
        let _ = std::fs::write(&path, &finding.shortest);
        report.push_str(&format!(
            "\n  {at}: {} times, shortest {} bytes: {}",
            finding.count,
            finding.shortest.len(),
            path.display()
        ));
    }
    panic!("{report}");
}

fn hash(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_3325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Writes GGUF structures, for a target's seeds.
pub struct GgufBytes(pub Vec<u8>);

impl GgufBytes {
    pub fn new(tensors: u64, metadata: u64) -> Self {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tensors.to_le_bytes());
        bytes.extend_from_slice(&metadata.to_le_bytes());
        Self(bytes)
    }

    pub fn string(&mut self, text: &str) -> &mut Self {
        self.0.extend_from_slice(&(text.len() as u64).to_le_bytes());
        self.0.extend_from_slice(text.as_bytes());
        self
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// A metadata pair holding a string.
    pub fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.string(key).u32(8).string(value)
    }

    /// A metadata pair holding a 32-bit number.
    pub fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.string(key).u32(4).u32(value)
    }

    /// A metadata pair holding an array of strings.
    pub fn kv_strings(&mut self, key: &str, values: &[String]) -> &mut Self {
        self.string(key).u32(9).u32(8).u64(values.len() as u64);
        for value in values {
            self.string(value);
        }
        self
    }

    /// A tensor-info entry.
    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> &mut Self {
        self.string(name).u32(dims.len() as u32);
        for &dim in dims {
            self.u64(dim);
        }
        self.u32(ggml_type).u64(offset)
    }
}
