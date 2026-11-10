//! `--perplexity`, `--hellaswag`, `--winogrande`: how good a model's
//! predictions are, as opposed to how fast it makes them.
//!
//! All three run teacher-forced through `/logits`, which answers the
//! log-probability of each token given the ones before it. Perplexity is
//! `e` to the mean negative log-probability over text. The two task scores ask
//! which of several continuations of a sentence the model finds most likely.
//!
//! Perplexity takes the text in windows of `--ppl-ctx` tokens and scores only
//! each window's second half, so that every scored token has context behind
//! it. Figures are comparable only between runs with the same window.

use std::path::Path;

use crate::{Args, history};

/// One token's log-probability sum: what a chunk adds to a perplexity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    pub nll: f64,
    pub nll2: f64,
    pub count: usize,
}

impl Tally {
    /// Adds the scores of positions `first..` of a window, where `scores[i]`
    /// is the log-probability of the token after position `i`.
    pub fn add_window(&mut self, scores: &[f32], first: usize) {
        for &score in &scores[first.min(scores.len())..] {
            let nll = -f64::from(score);
            self.nll += nll;
            self.nll2 += nll * nll;
            self.count += 1;
        }
    }

    /// The perplexity and its standard error.
    pub fn perplexity(&self) -> (f64, f64) {
        let count = self.count.max(1) as f64;
        let mean = self.nll / count;
        let perplexity = mean.exp();
        let variance = self.nll2 / count - mean * mean;
        let error = if variance > 0.0 && self.count > 1 {
            (variance / (count - 1.0)).sqrt() * perplexity
        } else {
            0.0
        };
        (perplexity, error)
    }
}

/// The windows of `tokens`: as many whole ones as there are, or `limit`.
/// `bos`, where the model starts a sequence with one, takes the place of each
/// window's first token.
pub fn windows(
    tokens: &[u32],
    n_ctx: usize,
    limit: Option<usize>,
    bos: Option<u32>,
) -> Vec<Vec<u32>> {
    let whole = tokens.len() / n_ctx.max(1);
    tokens
        .chunks_exact(n_ctx.max(1))
        .take(limit.map_or(whole, |limit| limit.min(whole)))
        .map(|window| {
            let mut window = window.to_vec();
            if let Some(bos) = bos {
                window[0] = bos;
            }
            window
        })
        .collect()
}

/// One HellaSwag task: a context and four endings, one of them the real one.
#[derive(Clone, Debug, PartialEq)]
pub struct SwagTask {
    pub context: String,
    pub gold: usize,
    pub endings: [String; 4],
}

/// HellaSwag's text form: six lines a task, the context, the index of the gold
/// ending, and the four endings.
pub fn parse_hellaswag(text: &str) -> anyhow::Result<Vec<SwagTask>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || !lines.len().is_multiple_of(6) {
        anyhow::bail!(
            "{} lines is not a whole number of six-line tasks",
            lines.len()
        );
    }
    lines
        .chunks(6)
        .enumerate()
        .map(|(i, task)| {
            let gold: usize = task[1]
                .trim()
                .parse()
                .ok()
                .filter(|gold| *gold < 4)
                .ok_or_else(|| {
                    anyhow::anyhow!("task {i}: `{}` is not an ending's index", task[1])
                })?;
            Ok(SwagTask {
                context: task[0].to_string(),
                gold,
                endings: [
                    task[2].to_string(),
                    task[3].to_string(),
                    task[4].to_string(),
                    task[5].to_string(),
                ],
            })
        })
        .collect()
}

/// How many leading tokens every sequence shares.
pub fn common_prefix(sequences: &[Vec<u32>]) -> usize {
    let Some((first, rest)) = sequences.split_first() else {
        return 0;
    };
    (0..first.len())
        .take_while(|&k| rest.iter().all(|s| s.get(k) == first.get(k)))
        .count()
}

/// The mean log-probability of tokens `from..` of a sequence, given `scores`,
/// where `scores[i]` is that of the token after position `i`.
fn mean_from(scores: &[f32], from: usize, to: usize) -> f64 {
    let from = from.max(1);
    if to <= from {
        return f64::NAN;
    }
    let sum: f64 = scores[from - 1..to - 1].iter().map(|&s| f64::from(s)).sum();
    sum / (to - from) as f64
}

/// The ending whose tokens the model finds most likely on average. The earlier
/// one wins a tie.
pub fn best_ending(scored: &[(usize, &[f32])], common: usize) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (i, (len, scores)) in scored.iter().enumerate() {
        let mean = mean_from(scores, common, *len);
        if mean > best.1 {
            best = (i, mean);
        }
    }
    best.0
}

/// One WinoGrande task: a sentence with a gap, two ways to fill it, and which
/// is right (1 or 2).
#[derive(Clone, Debug, PartialEq)]
pub struct WinoTask {
    pub first: String,
    pub second: String,
    pub choices: [String; 2],
    pub answer: usize,
}

/// The dataset's CSV: index, sentence with a `_`, the two choices, the answer.
/// Lines that do not read as a task are counted, not fatal, since the tools
/// that make these files leave a header or a stray line in.
pub fn parse_winogrande(text: &str) -> (Vec<WinoTask>, usize) {
    let mut skipped = 0;
    let tasks = text
        .lines()
        .filter_map(|line| {
            let task = wino_line(line);
            skipped += usize::from(task.is_none());
            task
        })
        .collect();
    (tasks, skipped)
}

fn wino_line(line: &str) -> Option<WinoTask> {
    let mut commas = [0usize; 4];
    let mut found = 0;
    let mut quoted = false;
    for (at, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                commas[found] = at;
                found += 1;
                if found == 4 {
                    break;
                }
            }
            _ => {}
        }
    }
    if found != 4 {
        return None;
    }
    let sentence = line[commas[0] + 1..commas[1]]
        .strip_prefix('"')
        .map_or(&line[commas[0] + 1..commas[1]], |s| {
            s.strip_suffix('"').unwrap_or(s)
        });
    let (first, second) = sentence.split_once('_')?;
    let answer: usize = line[commas[3] + 1..].trim().parse().ok()?;
    if !(1..=2).contains(&answer) {
        return None;
    }
    Some(WinoTask {
        first: first.to_string(),
        second: second.to_string(),
        choices: [
            line[commas[1] + 1..commas[2]].to_string(),
            line[commas[2] + 1..commas[3]].to_string(),
        ],
        answer,
    })
}

/// A sentence with both choices in the gap, tokenized.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WinoTokens {
    pub whole: [Vec<u32>; 2],
    /// Tokens up to and including each choice.
    pub base: [usize; 2],
}

/// Fewer tokens than this after the choice, and the sentence's end says too
/// little to judge by, so the choice's own tokens count too.
const MIN_TRAILING: usize = 3;

/// Which choice the model prefers, 1 or 2: the one under which the rest of the
/// sentence reads as more likely, leaving out its last token, which is
/// punctuation far more often than not.
pub fn wino_choice(tokens: &WinoTokens, scores: [&[f32]; 2]) -> usize {
    let common = common_prefix(&tokens.whole);
    let lens = [tokens.whole[0].len(), tokens.whole[1].len()];
    let skip_choice = lens.iter().all(|len| len - common > MIN_TRAILING);
    let mean = |i: usize| {
        let from = if skip_choice { tokens.base[i] } else { common };
        let last = usize::from(lens[i] - from > 1);
        mean_from(scores[i], from, lens[i] - last)
    };
    if mean(0) > mean(1) { 1 } else { 2 }
}

/// A 95% Wilson score interval of `correct` successes in `n`, as fractions.
pub fn wilson(correct: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let n = n as f64;
    let p = correct as f64 / n;
    let za = 1.959_963_984_54_f64;
    let z = za * za / n;
    let spread = z * (n * (4.0 * p * (1.0 - p) + z)).sqrt() / (za + za);
    (
        (p + z * 0.5 - spread) / (1.0 + z),
        (p + z * 0.5 + spread) / (1.0 + z),
    )
}

/// `n` of `total` indexes, the same ones for the same arguments, in an order
/// that does not follow the file, since these files keep related tasks
/// together and the score of a part should stand for the whole.
pub fn chosen(total: usize, n: Option<usize>) -> Vec<usize> {
    let mut indexes: Vec<usize> = (0..total).collect();
    let Some(n) = n.filter(|&n| n < total) else {
        return indexes;
    };
    // SplitMix64: all that is needed is the same order on every machine.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in 0..n {
        let j = i + (next() % (total - i) as u64) as usize;
        indexes.swap(i, j);
    }
    indexes.truncate(n);
    indexes
}

/// What `/logits` answers when asked to score.
#[derive(serde::Deserialize)]
struct Scored {
    scores: Vec<f32>,
}

fn scores_of(
    client: &reqwest::blocking::Client,
    args: &Args,
    tokens: &[u32],
) -> anyhow::Result<Vec<f32>> {
    let body = serde_json::json!({
        "tokens": tokens,
        "rows": args.score_rows,
        "top": 1,
        "score": true,
    });
    let resp = crate::post_with_one_retry(client, &format!("{}/logits", args.url), &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    let scored: Scored = resp.json()?;
    if scored.scores.len() + 1 != tokens.len() {
        anyhow::bail!(
            "/logits scored {} of {} tokens: an older server?",
            scored.scores.len(),
            tokens.len()
        );
    }
    Ok(scored.scores)
}

fn tokenize(
    client: &reqwest::blocking::Client,
    args: &Args,
    text: &str,
) -> anyhow::Result<Vec<u32>> {
    let body = serde_json::json!({ "content": text, "add_special": true });
    let resp = crate::post_with_one_retry(client, &format!("{}/tokenize", args.url), &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    let value: serde_json::Value = resp.json()?;
    Ok(serde_json::from_value(value["tokens"].clone())?)
}

fn read(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

fn record(label: &str, mode: &str, n: usize, value: f64, sd: f64) -> history::Record {
    history::Record {
        date: history::today(),
        label: label.to_string(),
        mode: mode.to_string(),
        n: n as u32,
        best: value,
        mean: value,
        sd,
        sd_sample: None,
        device: None,
    }
}

/// `--perplexity`.
pub fn run_perplexity(
    client: &reqwest::blocking::Client,
    args: &Args,
    path: &Path,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let n_ctx = args.ppl_ctx;
    if n_ctx < 4 {
        anyhow::bail!("--ppl-ctx must be at least 4");
    }
    let tokens = tokenize(client, args, &read(path)?)?;
    if tokens.len() < 2 * n_ctx {
        anyhow::bail!(
            "{} tokenizes to {} tokens, and a window of {n_ctx} needs at least {}",
            path.display(),
            tokens.len(),
            2 * n_ctx
        );
    }
    // Tokenizing nothing gives the BOS alone, where this model has one.
    let bos = tokenize(client, args, "")?.first().copied();
    let windows = windows(&tokens, n_ctx, args.ppl_chunks, bos);
    let started = std::time::Instant::now();
    let mut tally = Tally::default();
    for (i, window) in windows.iter().enumerate() {
        tally.add_window(&scores_of(client, args, window)?, n_ctx / 2);
        if !args.json {
            let (perplexity, _) = tally.perplexity();
            eprint!(
                "\rperplexity {perplexity:.4} after {} of {} windows",
                i + 1,
                windows.len()
            );
        }
    }
    let (perplexity, error) = tally.perplexity();
    let seconds = started.elapsed().as_secs_f64();
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "perplexity": perplexity,
                "error": error,
                "windows": windows.len(),
                "window": n_ctx,
                "scored_tokens": tally.count,
                "rows": args.score_rows,
                "seconds": seconds,
            })
        );
    } else {
        eprintln!();
        println!(
            "perplexity {perplexity:.4} +/- {error:.5}, {} tokens scored in {} windows of {n_ctx}, {seconds:.1} s",
            tally.count,
            windows.len(),
        );
    }
    if let Some(baseline) = args.ppl_baseline {
        let rise = 100.0 * (perplexity - baseline) / baseline;
        let over = rise > args.ppl_max_increase;
        if over {
            crate::kl::note_over_threshold();
        }
        if !args.json {
            println!(
                "threshold  {rise:+.2}% against {baseline} (at most +{}%): {}",
                args.ppl_max_increase,
                if over { "OVER" } else { "within" }
            );
        }
    }
    Ok(vec![record(label, "perplexity", n_ctx, perplexity, error)])
}

/// `--hellaswag`.
pub fn run_hellaswag(
    client: &reqwest::blocking::Client,
    args: &Args,
    path: &Path,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let tasks =
        parse_hellaswag(&read(path)?).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let indexes = chosen(tasks.len(), args.score_tasks);
    let mut correct = 0;
    for (done, &index) in indexes.iter().enumerate() {
        let task = &tasks[index];
        let mut sequences = Vec::with_capacity(4);
        for ending in &task.endings {
            sequences.push(tokenize(
                client,
                args,
                &format!("{} {ending}", task.context),
            )?);
        }
        let common = common_prefix(&sequences);
        let mut scores = Vec::with_capacity(4);
        for sequence in &sequences {
            scores.push(scores_of(client, args, sequence)?);
        }
        let scored: Vec<(usize, &[f32])> = sequences
            .iter()
            .zip(&scores)
            .map(|(sequence, scores)| (sequence.len(), scores.as_slice()))
            .collect();
        correct += usize::from(best_ending(&scored, common) == task.gold);
        progress(args, "hellaswag", correct, done + 1, indexes.len());
    }
    report(args, label, "hellaswag", correct, indexes.len(), 0)
}

/// `--winogrande`.
pub fn run_winogrande(
    client: &reqwest::blocking::Client,
    args: &Args,
    path: &Path,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let (tasks, skipped) = parse_winogrande(&read(path)?);
    if tasks.is_empty() {
        anyhow::bail!("{}: no tasks", path.display());
    }
    let indexes = chosen(tasks.len(), args.score_tasks);
    let mut correct = 0;
    for (done, &index) in indexes.iter().enumerate() {
        let task = &tasks[index];
        let mut tokens = WinoTokens::default();
        for i in 0..2 {
            let upto = format!("{}{}", task.first, task.choices[i]);
            tokens.base[i] = tokenize(client, args, &upto)?.len();
            tokens.whole[i] = tokenize(client, args, &format!("{upto}{}", task.second))?;
        }
        let first = scores_of(client, args, &tokens.whole[0])?;
        let second = scores_of(client, args, &tokens.whole[1])?;
        correct += usize::from(wino_choice(&tokens, [&first, &second]) == task.answer);
        progress(args, "winogrande", correct, done + 1, indexes.len());
    }
    report(args, label, "winogrande", correct, indexes.len(), skipped)
}

fn progress(args: &Args, name: &str, correct: usize, done: usize, total: usize) {
    if !args.json {
        eprint!(
            "\r{name} {:.2}% after {done} of {total} tasks",
            100.0 * correct as f64 / done as f64
        );
    }
}

fn report(
    args: &Args,
    label: &str,
    name: &str,
    correct: usize,
    n: usize,
    skipped: usize,
) -> anyhow::Result<Vec<history::Record>> {
    let score = 100.0 * correct as f64 / n.max(1) as f64;
    let (low, high) = wilson(correct, n);
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                name: score,
                "correct": correct,
                "tasks": n,
                "skipped_lines": skipped,
                "interval_95": [low * 100.0, high * 100.0],
                "rows": args.score_rows,
            })
        );
    } else {
        eprintln!();
        println!(
            "{name} {score:.2}% ({correct} of {n} tasks), 95% interval [{:.2}%, {:.2}%]",
            low * 100.0,
            high * 100.0
        );
        if skipped > 0 {
            println!("{skipped} lines did not read as a task");
        }
    }
    Ok(vec![record(label, name, n, score, (high - low) * 25.0)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_is_e_to_the_mean_negative_log_probability() {
        let mut tally = Tally::default();
        tally.add_window(&[-9.0, -9.0, -1.0, -3.0], 2);
        assert_eq!(tally.count, 2);
        let (perplexity, error) = tally.perplexity();
        assert!((perplexity - 2f64.exp()).abs() < 1e-9);
        // Variance 1 over one degree of freedom, carried through the exponent.
        assert!((error - 2f64.exp()).abs() < 1e-9, "{error}");
    }

    #[test]
    fn a_window_of_n_scores_the_tokens_of_its_second_half() {
        // 8 tokens give 7 scores; a window's second half starts at position 4
        // and its last token predicts nothing, so 3 are scored.
        let mut tally = Tally::default();
        tally.add_window(&[-1.0; 7], 8 / 2);
        assert_eq!(tally.count, 3);
    }

    #[test]
    fn windows_are_whole_and_begin_with_the_bos() {
        let tokens: Vec<u32> = (1..=11).collect();
        assert_eq!(
            windows(&tokens, 4, None, Some(99)),
            vec![vec![99, 2, 3, 4], vec![99, 6, 7, 8]]
        );
        assert_eq!(
            windows(&tokens, 4, None, None),
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
        );
        assert_eq!(windows(&tokens, 4, Some(1), None).len(), 1);
        assert_eq!(windows(&tokens, 4, Some(9), None).len(), 2);
    }

    #[test]
    fn hellaswag_reads_six_lines_a_task_and_refuses_anything_else() {
        let task = parse_hellaswag("Ctx: a b\n3\ne0\ne1\ne2\ne3\n").unwrap();
        assert_eq!(task.len(), 1);
        assert_eq!(task[0].context, "Ctx: a b");
        assert_eq!(task[0].gold, 3);
        assert_eq!(task[0].endings[1], "e1");
        assert!(parse_hellaswag("a\n3\nb\n").is_err());
        assert!(parse_hellaswag("").is_err());
        assert!(parse_hellaswag("a\n4\ne\ne\ne\ne\n").is_err());
        assert!(parse_hellaswag("a\nx\ne\ne\ne\ne\n").is_err());
    }

    /// An ending is judged by its mean, not its sum: a long ending is not
    /// penalized for having more tokens to get wrong.
    #[test]
    fn the_likeliest_ending_on_average_wins_however_long_it_is() {
        // Two shared tokens, then each ending. scores[1] is the first token
        // after the shared ones.
        let short: &[f32] = &[-9.0, -2.0];
        let long: &[f32] = &[-9.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!(best_ending(&[(3, short), (6, long)], 2), 1);
        let unlikely: &[f32] = &[-9.0, -3.0, -3.0];
        let likely: &[f32] = &[-9.0, -2.0, -2.0];
        assert_eq!(best_ending(&[(4, unlikely), (4, likely)], 2), 1);
        // The earlier one wins a tie.
        assert_eq!(best_ending(&[(4, likely), (4, likely)], 2), 0);
        // What the sequences share does not count.
        let a: &[f32] = &[-50.0, -1.0];
        let b: &[f32] = &[-0.1, -2.0];
        assert_eq!(best_ending(&[(3, a), (3, b)], 2), 0);
    }

    #[test]
    fn a_common_prefix_stops_at_the_first_difference() {
        assert_eq!(common_prefix(&[vec![1, 2, 3], vec![1, 2, 4]]), 2);
        assert_eq!(common_prefix(&[vec![1, 2], vec![1, 2, 3]]), 2);
        assert_eq!(common_prefix(&[vec![9], vec![1]]), 0);
        assert_eq!(common_prefix(&[]), 0);
    }

    #[test]
    fn winogrande_reads_its_csv_quotes_and_all() {
        let (tasks, skipped) = parse_winogrande(
            "0,Sarah beat Maria so _ won.,Sarah,Maria,1\n\
             1,\"Well, _ lost.\",Sarah,Maria,2\n\
             sentence,option1,option2,answer\n\
             3,no gap here,a,b,1\n\
             4,_ odd,a,b,7\n",
        );
        assert_eq!(skipped, 3);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].first, "Sarah beat Maria so ");
        assert_eq!(tasks[0].second, " won.");
        assert_eq!(tasks[0].choices, ["Sarah".to_string(), "Maria".to_string()]);
        assert_eq!(tasks[0].answer, 1);
        assert_eq!(tasks[1].first, "Well, ");
        assert_eq!(tasks[1].second, " lost.");
        assert_eq!(tasks[1].answer, 2);
    }

    /// With enough sentence after the gap, only what follows the choice
    /// counts, whatever the model thought of the choice itself, and the last
    /// token is left out.
    #[test]
    fn a_long_trailing_sentence_is_judged_without_the_choice() {
        let tokens = WinoTokens {
            whole: [vec![1, 2, 10, 5, 6, 7, 8], vec![1, 2, 20, 5, 6, 7, 8]],
            base: [3, 3],
        };
        // Choice 1's own token is unlikely, but the rest reads better after it.
        let first: &[f32] = &[-1.0, -30.0, -1.0, -1.0, -1.0, -99.0];
        let second: &[f32] = &[-1.0, -0.1, -2.0, -2.0, -2.0, -0.1];
        assert_eq!(wino_choice(&tokens, [first, second]), 1);
    }

    /// With little after the gap, the choice's own tokens count.
    #[test]
    fn a_short_trailing_sentence_is_judged_with_the_choice() {
        let tokens = WinoTokens {
            whole: [vec![1, 2, 10, 5], vec![1, 2, 20, 5]],
            base: [3, 3],
        };
        // The scored tokens are the choice alone, the last being left out.
        let first: &[f32] = &[-1.0, -5.0, -0.1];
        let second: &[f32] = &[-1.0, -0.5, -9.0];
        assert_eq!(wino_choice(&tokens, [first, second]), 2);
    }

    #[test]
    fn the_wilson_interval_holds_the_score_and_stays_in_range() {
        let (low, high) = wilson(50, 100);
        assert!((low - 0.4038).abs() < 1e-3, "{low}");
        assert!((high - 0.5962).abs() < 1e-3, "{high}");
        let (low, high) = wilson(0, 10);
        assert!(
            low.abs() < 1e-9 && high > 0.2 && high < 0.35,
            "{low} {high}"
        );
        let (low, high) = wilson(10, 10);
        assert!(low > 0.65 && (high - 1.0).abs() < 1e-9, "{low} {high}");
        assert_eq!(wilson(0, 0), (0.0, 0.0));
    }

    #[test]
    fn a_part_of_the_tasks_is_the_same_part_every_time_and_has_no_repeats() {
        let part = chosen(1000, Some(50));
        assert_eq!(part, chosen(1000, Some(50)));
        assert_eq!(part.len(), 50);
        let mut unique = part.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 50);
        // Not simply the start of the file.
        assert!(part.iter().any(|&i| i >= 50));
        assert_eq!(chosen(3, Some(50)), vec![0, 1, 2]);
        assert_eq!(chosen(3, None), vec![0, 1, 2]);
    }
}
