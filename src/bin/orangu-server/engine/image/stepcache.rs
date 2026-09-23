//! Fewer transformer passes: EasyCache (Zhou et al., 2025), the
//! transformer's input-to-output residual reused across steps.
//!
//! Over the middle of a flow-matching schedule a step's velocity changes
//! about as fast as its input does, at a rate that itself changes slowly.
//! So the rate `k = ‖v − v′‖ / ‖x − x′‖` measured between the last two
//! passes predicts the next output's relative change from the input's
//! alone, `k · ‖x_t − x_{t−1}‖ / ‖v‖`. While those predictions, summed over
//! the steps since the last pass, stay under a threshold, the step takes
//! `v = x_t + r` — the residual `v − x` of the last two passes extrapolated
//! to this step — and runs no transformer at all. The first and last stretch of the
//! schedule, where the picture's layout and its fine detail are decided,
//! always pass.

use anyhow::{Result, bail};

/// Where the cache may skip, as fractions of the steps run: before
/// `START` the rate is still being learned and the layout is being
/// decided, from `END` on the detail is.
const START: f64 = 0.15;
const END: f64 = 0.95;

/// `[orangu-server].image_cache`: how many passes a picture may reuse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageCache {
    /// Every step runs the transformer.
    Off,
    /// EasyCache under this accumulated relative-change threshold — the
    /// default, at [`ImageCache::EASY`].
    Easy(f32),
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::Easy(Self::EASY)
    }
}

impl ImageCache {
    /// The default threshold of `easy`: on Qwen-Image 2.1 at 40 steps it
    /// reuses 25 of the steps at ~31 dB against the uncached picture
    /// (`doc/PERF-IMAGE.md`, task 4); 0.1 starts to ghost the lettering.
    pub const EASY: f32 = 0.08;

    /// `off`, `easy`, or `easy:<threshold>`.
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim().to_ascii_lowercase();
        match value.split_once(':') {
            None if value == "off" => Ok(Self::Off),
            None if value == "easy" => Ok(Self::Easy(Self::EASY)),
            Some(("easy", threshold)) => match threshold.trim().parse::<f32>() {
                Ok(t) if t.is_finite() && t > 0.0 => Ok(Self::Easy(t)),
                _ => bail!("the easy threshold must be a positive number, got '{threshold}'"),
            },
            _ => bail!("expected off, easy or easy:<threshold>, got '{value}'"),
        }
    }
}

static CHOICE: std::sync::Mutex<ImageCache> =
    std::sync::Mutex::new(ImageCache::Easy(ImageCache::EASY));

/// The configured [`ImageCache`], set once by `main`.
pub fn set(choice: ImageCache) {
    *CHOICE.lock().unwrap_or_else(|p| p.into_inner()) = choice;
}

/// The configured choice, `ORANGU_IMAGE_CACHE` over the configuration.
pub fn configured() -> ImageCache {
    std::env::var("ORANGU_IMAGE_CACHE")
        .ok()
        .and_then(|v| ImageCache::parse(&v).ok())
        .unwrap_or_else(|| *CHOICE.lock().unwrap_or_else(|p| p.into_inner()))
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// One picture's cache: decides per step whether the transformer runs.
pub struct StepCache {
    threshold: f64,
    first: usize,
    last: usize,
    /// The previous step's input, passed or not.
    previous_input: Option<Vec<f32>>,
    /// The last pass's input and output.
    passed: Option<(Vec<f32>, Vec<f32>)>,
    /// The residuals `v − x` of the last two passes, with their steps,
    /// newest last — for extrapolating one to the current step.
    residuals: Vec<(usize, Vec<f32>)>,
    /// `‖v − v′‖ / ‖x − x′‖` between the last two passes.
    rate: Option<f64>,
    /// The predicted relative output change since the last pass.
    accumulated: f64,
    /// Steps that reused rather than passed.
    pub skipped: usize,
}

impl StepCache {
    /// A cache for a picture of `steps` steps, or `None` when off.
    pub fn new(choice: ImageCache, steps: usize) -> Option<Self> {
        let ImageCache::Easy(threshold) = choice else {
            return None;
        };
        Some(Self {
            threshold: threshold as f64,
            first: (steps as f64 * START).round() as usize,
            last: (steps as f64 * END).round() as usize,
            previous_input: None,
            passed: None,
            residuals: Vec::new(),
            rate: None,
            accumulated: 0.0,
            skipped: 0,
        })
    }

    /// Step `done` (counted from `0`) of this picture on input `x`: the
    /// reused velocity, or `None` when the transformer must run (and
    /// [`StepCache::record`] be given its output).
    pub fn reuse(&mut self, done: usize, x: &[f32]) -> Option<Vec<f32>> {
        let previous = self.previous_input.replace(x.to_vec());
        if done < self.first || done >= self.last {
            return None;
        }
        let (rate, previous, (passed_x, passed_v)) = match (self.rate, previous, &self.passed) {
            (Some(rate), Some(previous), Some(passed)) => (rate, previous, passed),
            _ => return None,
        };
        let v_norm = norm(passed_v);
        if v_norm == 0.0 {
            return None;
        }
        self.accumulated += rate * distance(x, &previous) / v_norm;
        if self.accumulated >= self.threshold {
            return None;
        }
        self.skipped += 1;
        // The residual extrapolated along the last two passes' (a first
        // order TaylorSeer): at the same threshold 36.5 against 30.1 dB
        // for the last residual as it was.
        if let [(s0, r0), (s1, r1)] = &self.residuals[..] {
            let f = (done - s1) as f32 / (s1 - s0) as f32;
            return Some(
                x.iter()
                    .zip(r1.iter().zip(r0))
                    .map(|(&x, (&r1, &r0))| x + r1 + (r1 - r0) * f)
                    .collect(),
            );
        }
        Some(
            x.iter()
                .zip(passed_x.iter().zip(passed_v))
                .map(|(&x, (&px, &pv))| x + (pv - px))
                .collect(),
        )
    }

    /// The transformer ran on `x` at step `done` and gave `v`.
    pub fn record(&mut self, done: usize, x: &[f32], v: &[f32]) {
        if self.residuals.len() == 2 {
            self.residuals.remove(0);
        }
        self.residuals
            .push((done, v.iter().zip(x).map(|(v, x)| v - x).collect()));
        if let Some((px, pv)) = &self.passed {
            let dx = distance(x, px);
            if dx > 0.0 {
                self.rate = Some(distance(v, pv) / dx);
            }
        }
        self.passed = Some((x.to_vec(), v.to_vec()));
        self.accumulated = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_choice_parses() {
        assert_eq!(ImageCache::parse("off").unwrap(), ImageCache::Off);
        assert_eq!(
            ImageCache::parse("Easy").unwrap(),
            ImageCache::Easy(ImageCache::EASY)
        );
        assert_eq!(
            ImageCache::parse("easy:0.1").unwrap(),
            ImageCache::Easy(0.1)
        );
        assert!(ImageCache::parse("easy:0").is_err());
        assert!(ImageCache::parse("teacache").is_err());
    }

    /// A velocity that is an affine function of the input is reused
    /// closely, and only inside the window.
    #[test]
    fn a_steady_rate_is_reused_inside_the_window() {
        let mut cache = StepCache::new(ImageCache::Easy(10.0), 20).unwrap();
        let velocity = |x: &[f32]| x.iter().map(|&x| 2.0 * x + 1.0).collect::<Vec<f32>>();
        let mut passes = 0;
        for done in 0..20 {
            let x: Vec<f32> = (0..8).map(|i| i as f32 + 0.01 * done as f32).collect();
            match cache.reuse(done, &x) {
                Some(v) => {
                    assert!((3..19).contains(&done), "step {done} reused");
                    // The residual `v − x = x + 1` of the last pass, on
                    // this x: off by the input's drift since, 0.01 a step.
                    let exact = velocity(&x);
                    for (a, b) in v.iter().zip(&exact) {
                        assert!((a - b).abs() < 0.01 * 17.0, "{a} vs {b}");
                    }
                }
                None => {
                    passes += 1;
                    cache.record(done, &x, &velocity(&x));
                }
            }
        }
        assert_eq!(passes + cache.skipped, 20);
        assert!(cache.skipped > 10, "only {} reused", cache.skipped);
    }

    #[test]
    fn off_is_no_cache() {
        assert!(StepCache::new(ImageCache::Off, 40).is_none());
    }
}
