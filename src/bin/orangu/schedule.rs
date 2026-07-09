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

//! Cron-style scheduled commands behind `/schedule`.
//!
//! Jobs live in `~/.orangu/schedule`, one per line in classic crontab form —
//! five time fields, then the command to run:
//!
//! ```text
//! 0 * * * * /export pr
//! 30 6 * * 1-5 /statistics
//! ```
//!
//! The run loop checks the schedule as it goes around (see
//! [`enqueue_due_jobs`]); a due job's command is pushed onto the same pending
//! queue that holds commands typed while a request is in flight, so it runs
//! exactly like one the user entered — in the active workspace tab, but
//! unattended: `/auto_review` skips its interactive pre-start and browse
//! phases when scheduled, so a run completes without a human present. `&&`
//! chains commands — `auto review immediate && export auto review` reviews,
//! then exports the finished report — running each part in order and dropping
//! the rest of the chain when a part fails. Every minute boundary since the
//! last check is considered, so a job isn't skipped when the loop was busy
//! (or idle) across its minute; nothing fires for minutes that passed before
//! orangu started. The file is re-read at each minute boundary, so edits
//! apply without a restart.
//!
//! Times are UTC — orangu has no timezone database to resolve local time
//! against. Fields are numeric (no `JAN`/`MON` names) and support `*`, lists
//! (`1,15`), ranges (`1-5`), and steps (`*/10`, `8-18/2`), with `0` or `7`
//! for Sunday. When both day-of-month and day-of-week are restricted the job
//! runs when either matches, like classic cron.

use std::collections::VecDeque;
use std::path::PathBuf;

/// One parsed cron time field: the set of values it allows.
#[derive(Debug, Clone, PartialEq)]
struct CronField {
    allowed: Vec<bool>,
    /// Whether the field was `*` (or `*/1`) — needed for cron's special
    /// day-of-month/day-of-week rule, where a lone `*` means "unrestricted"
    /// rather than "every value".
    any: bool,
}

impl CronField {
    fn contains(&self, value: u64) -> bool {
        self.allowed.get(value as usize).copied().unwrap_or(false)
    }
}

/// A parsed five-field cron expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CronSchedule {
    minutes: CronField,
    hours: CronField,
    days_of_month: CronField,
    months: CronField,
    days_of_week: CronField,
}

/// One line of `~/.orangu/schedule`: its cron expression and the command
/// chain it runs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScheduledJob {
    pub(crate) schedule: CronSchedule,
    /// The line's commands, split on `&&`, in order. Usually one; a chain
    /// like `auto review immediate && export auto review` runs its parts
    /// sequentially and stops if one fails.
    pub(crate) commands: Vec<String>,
}

/// A crontab line orangu couldn't parse, kept so `/schedule` can point at it
/// instead of silently ignoring it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InvalidLine {
    pub(crate) line: String,
    pub(crate) error: String,
}

/// The parsed schedule file: runnable jobs plus any lines that didn't parse.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Crontab {
    pub(crate) jobs: Vec<ScheduledJob>,
    pub(crate) invalid: Vec<InvalidLine>,
}

/// Parse one cron field over `min..=max` into its allowed-value set.
fn parse_field(field: &str, min: u64, max: u64) -> Result<CronField, String> {
    let mut allowed = vec![false; (max + 1) as usize];
    let mut any = true;
    for part in field.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((range, step)) => {
                let step: u64 = step.parse().map_err(|_| format!("bad step in {part:?}"))?;
                if step == 0 {
                    return Err(format!("step of 0 in {part:?}"));
                }
                (range, step)
            }
            None => (part, 1),
        };
        let (start, end) = if range == "*" {
            (min, max)
        } else if let Some((a, b)) = range.split_once('-') {
            let a: u64 = a.parse().map_err(|_| format!("bad range in {part:?}"))?;
            let b: u64 = b.parse().map_err(|_| format!("bad range in {part:?}"))?;
            (a, b)
        } else {
            let v: u64 = range
                .parse()
                .map_err(|_| format!("bad value in {part:?}"))?;
            (v, v)
        };
        if start < min || end > max || start > end {
            return Err(format!("{part:?} outside {min}-{max}"));
        }
        if range != "*" || step != 1 {
            any = false;
        }
        let mut v = start;
        while v <= end {
            allowed[v as usize] = true;
            v += step;
        }
    }
    Ok(CronField { allowed, any })
}

/// Parse a five-field cron expression (`minute hour day-of-month month
/// day-of-week`). `7` is accepted as Sunday alongside `0`.
pub(crate) fn parse_cron(expression: &str) -> Result<CronSchedule, String> {
    let fields: Vec<&str> = expression.split_whitespace().collect();
    if fields.len() != 5 {
        return Err(format!("expected 5 time fields, found {}", fields.len()));
    }
    let mut days_of_week = parse_field(fields[4], 0, 7)?;
    // Cron treats 7 as another spelling of Sunday.
    if days_of_week.contains(7) {
        days_of_week.allowed[0] = true;
    }
    Ok(CronSchedule {
        minutes: parse_field(fields[0], 0, 59)?,
        hours: parse_field(fields[1], 0, 23)?,
        days_of_month: parse_field(fields[2], 1, 31)?,
        months: parse_field(fields[3], 1, 12)?,
        days_of_week,
    })
}

impl CronSchedule {
    /// Whether the schedule fires at the given minute since the Unix epoch
    /// (UTC). Classic cron day rule: when both day-of-month and day-of-week
    /// are restricted, either matching is enough; otherwise the restricted
    /// one (if any) must match.
    pub(crate) fn matches_minute(&self, unix_minute: u64) -> bool {
        let secs = unix_minute * 60;
        let minute = unix_minute % 60;
        let hour = (secs / 3600) % 24;
        let day = secs / 86_400;
        let (_, month, day_of_month) = crate::export::civil_from_days(day as i64);
        // `weekday_index` is Monday-first (0 = Monday); cron counts from
        // Sunday (0 = Sunday).
        let day_of_week = (crate::activity_log::weekday_index(day) as u64 + 1) % 7;

        if !self.minutes.contains(minute)
            || !self.hours.contains(hour)
            || !self.months.contains(month as u64)
        {
            return false;
        }
        let dom = self.days_of_month.contains(day_of_month as u64);
        let dow = self.days_of_week.contains(day_of_week);
        match (self.days_of_month.any, self.days_of_week.any) {
            (false, false) => dom || dow,
            (false, true) => dom,
            (true, false) => dow,
            (true, true) => true,
        }
    }

    /// The next minute (since the Unix epoch, UTC) at or after `from` the
    /// schedule fires, scanning up to four years ahead — far enough for any
    /// satisfiable expression, `None` for one that never fires (e.g. Feb 30).
    pub(crate) fn next_run(&self, from: u64) -> Option<u64> {
        const FOUR_YEARS_OF_MINUTES: u64 = 4 * 366 * 24 * 60;
        (from..from + FOUR_YEARS_OF_MINUTES).find(|&minute| self.matches_minute(minute))
    }
}

/// Parse the whole schedule file: one job per line, `#` comments and blank
/// lines skipped, unparseable lines collected rather than dropped.
pub(crate) fn parse_crontab(content: &str) -> Crontab {
    let mut crontab = Crontab::default();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // The command starts at the sixth whitespace-separated token.
        let mut fields = trimmed.splitn(6, char::is_whitespace);
        let expression: Vec<&str> = (&mut fields).take(5).collect();
        let command = fields.next().map(str::trim).unwrap_or_default();
        if expression.len() < 5 || command.is_empty() {
            crontab.invalid.push(InvalidLine {
                line: trimmed.to_string(),
                error: "expected 5 time fields and a command".to_string(),
            });
            continue;
        }
        // `&&` chains the commands: each runs after the previous, and a
        // failure drops the rest of the chain.
        let commands: Vec<String> = command
            .split("&&")
            .map(|part| part.trim().to_string())
            .collect();
        if commands.iter().any(String::is_empty) {
            crontab.invalid.push(InvalidLine {
                line: trimmed.to_string(),
                error: "empty command in && chain".to_string(),
            });
            continue;
        }
        match parse_cron(&expression.join(" ")) {
            Ok(schedule) => crontab.jobs.push(ScheduledJob { schedule, commands }),
            Err(error) => crontab.invalid.push(InvalidLine {
                line: trimmed.to_string(),
                error,
            }),
        }
    }
    crontab
}

/// The schedule file's path, `~/.orangu/schedule`. `None` when no home
/// directory resolves.
pub(crate) fn schedule_file_path() -> Option<PathBuf> {
    Some(home::home_dir()?.join(".orangu").join("schedule"))
}

/// Read and parse `~/.orangu/schedule`. A missing file is an empty schedule.
pub(crate) fn load_crontab() -> Crontab {
    let Some(path) = schedule_file_path() else {
        return Crontab::default();
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Crontab::default();
    };
    parse_crontab(&content)
}

/// Push every job due since the last check onto the pending-commands queue.
/// Called each time around the run loop; does nothing until the minute
/// advances past `last_minute` (which it updates), then checks every minute
/// boundary crossed since, so jobs aren't skipped when the loop was busy
/// across their minute. Initialise `last_minute` to the current minute at
/// startup so past minutes never fire.
///
/// Every enqueued command carries its firing's chain id, marking it as a
/// scheduled, unattended run; a job's `&&` chain shares one id so a failed
/// link can drop the rest of the chain.
pub(crate) fn enqueue_due_jobs(
    last_minute: &mut u64,
    pending: &mut VecDeque<crate::wait::PendingCommand>,
) {
    let now = crate::session_store::current_unix_timestamp() / 60;
    if now <= *last_minute {
        return;
    }
    let crontab = load_crontab();
    for minute in (*last_minute + 1)..=now {
        for (index, job) in crontab.jobs.iter().enumerate() {
            if job.schedule.matches_minute(minute) {
                // Unique per job firing: no two firings of the same minute
                // share an id, and neither do two minutes of the same job.
                let chain = minute.wrapping_mul(10_000).wrapping_add(index as u64);
                for command in &job.commands {
                    pending.push_back(crate::wait::PendingCommand {
                        text: command.clone(),
                        chain: Some(chain),
                    });
                }
            }
        }
    }
    *last_minute = now;
}

/// The `/schedule` listing: each job with its next run time (UTC), then any
/// lines that didn't parse. With no schedule file (or an empty one), says how
/// to create it.
pub(crate) fn format_schedule_list() -> String {
    let path = schedule_file_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "~/.orangu/schedule".to_string());
    let crontab = load_crontab();
    if crontab.jobs.is_empty() && crontab.invalid.is_empty() {
        return format!(
            "No scheduled jobs. Add crontab-style lines (UTC) to {path}, e.g.\n  0 * * * * /export pr"
        );
    }

    let now = crate::session_store::current_unix_timestamp() / 60;
    let mut out = format!("Scheduled jobs ({path}, times UTC):\n");
    for job in &crontab.jobs {
        let next = match job.schedule.next_run(now + 1) {
            Some(minute) => crate::session_store::format_unix_timestamp_human(minute * 60),
            None => "never".to_string(),
        };
        out.push_str(&format!("{:<40} next: {next}\n", job.commands.join(" && ")));
    }
    for invalid in &crontab.invalid {
        out.push_str(&format!("invalid ({}): {}\n", invalid.error, invalid.line));
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cron_accepts_wildcards_lists_ranges_and_steps() {
        let schedule = parse_cron("*/15 8-18/2 1,15 * 1-5").expect("parse");
        assert!(schedule.minutes.contains(0));
        assert!(schedule.minutes.contains(45));
        assert!(!schedule.minutes.contains(20));
        assert!(schedule.hours.contains(8));
        assert!(schedule.hours.contains(10));
        assert!(!schedule.hours.contains(9));
        assert!(schedule.days_of_month.contains(1));
        assert!(schedule.days_of_month.contains(15));
        assert!(!schedule.days_of_month.contains(2));
        assert!(schedule.months.any);
        assert!(schedule.days_of_week.contains(1));
        assert!(!schedule.days_of_week.contains(0));
    }

    #[test]
    fn parse_cron_treats_seven_as_sunday() {
        let schedule = parse_cron("0 0 * * 7").expect("parse");
        assert!(schedule.days_of_week.contains(0));
    }

    #[test]
    fn parse_cron_rejects_bad_input() {
        assert!(parse_cron("0 0 * *").is_err()); // Four fields.
        assert!(parse_cron("60 * * * *").is_err()); // Minute out of range.
        assert!(parse_cron("* 24 * * *").is_err()); // Hour out of range.
        assert!(parse_cron("* * 0 * *").is_err()); // Day of month from 1.
        assert!(parse_cron("* * * 13 *").is_err()); // Month out of range.
        assert!(parse_cron("*/0 * * * *").is_err()); // Zero step.
        assert!(parse_cron("5-1 * * * *").is_err()); // Backwards range.
        assert!(parse_cron("a * * * *").is_err()); // Not a number.
    }

    #[test]
    fn matches_minute_checks_every_field() {
        // 2021-01-01 (day 18628) was a Friday; 12:30 UTC.
        let minute = 18628 * 24 * 60 + 12 * 60 + 30;
        assert!(
            parse_cron("30 12 1 1 *")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            parse_cron("30 12 * * 5")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            !parse_cron("31 12 1 1 *")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            !parse_cron("30 13 1 1 *")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            !parse_cron("30 12 2 1 *")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            !parse_cron("30 12 1 2 *")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            !parse_cron("30 12 * * 4")
                .expect("parse")
                .matches_minute(minute)
        );
    }

    #[test]
    fn matches_minute_runs_on_either_day_field_when_both_are_restricted() {
        // 2021-01-01 was a Friday (cron day-of-week 5), the 1st of the month.
        let minute = 18628 * 24 * 60;
        // Day-of-month matches even though day-of-week (Monday) doesn't.
        assert!(
            parse_cron("0 0 1 * 1")
                .expect("parse")
                .matches_minute(minute)
        );
        // Day-of-week matches even though day-of-month (the 2nd) doesn't.
        assert!(
            parse_cron("0 0 2 * 5")
                .expect("parse")
                .matches_minute(minute)
        );
        // Neither matches.
        assert!(
            !parse_cron("0 0 2 * 1")
                .expect("parse")
                .matches_minute(minute)
        );
        // Only one field restricted: it alone decides.
        assert!(
            !parse_cron("0 0 2 * *")
                .expect("parse")
                .matches_minute(minute)
        );
        assert!(
            !parse_cron("0 0 * * 1")
                .expect("parse")
                .matches_minute(minute)
        );
    }

    #[test]
    fn next_run_finds_the_following_match() {
        // From 2021-01-01 00:00, the next "30 12 on the 15th" is Jan 15 12:30.
        let from = 18628 * 24 * 60;
        let schedule = parse_cron("30 12 15 * *").expect("parse");
        let next = schedule.next_run(from).expect("next");
        assert_eq!(next, (18628 + 14) * 24 * 60 + 12 * 60 + 30);
        // February 30th never comes.
        assert_eq!(
            parse_cron("0 0 30 2 *").expect("parse").next_run(from),
            None
        );
    }

    #[test]
    fn parse_crontab_splits_jobs_comments_and_invalid_lines() {
        let content = "\
# hourly pull request report
0 * * * * /export pr

30 6 * * 1-5 /statistics
not a cron line
61 * * * * /diff
";
        let crontab = parse_crontab(content);
        assert_eq!(crontab.jobs.len(), 2);
        assert_eq!(crontab.jobs[0].commands, vec!["/export pr"]);
        assert_eq!(crontab.jobs[1].commands, vec!["/statistics"]);
        assert_eq!(crontab.invalid.len(), 2);
        assert!(crontab.invalid[0].line.starts_with("not a cron"));
        assert!(crontab.invalid[1].line.starts_with("61"));
    }

    #[test]
    fn parse_crontab_keeps_multi_word_commands_whole() {
        let crontab = parse_crontab("*/5 * * * * export auto review");
        assert_eq!(crontab.jobs.len(), 1);
        assert_eq!(crontab.jobs[0].commands, vec!["export auto review"]);
    }

    #[test]
    fn parse_crontab_splits_and_chains_on_double_ampersand() {
        let crontab = parse_crontab("0 6 * * * auto review immediate && export auto review");
        assert_eq!(crontab.jobs.len(), 1);
        assert_eq!(
            crontab.jobs[0].commands,
            vec!["auto review immediate", "export auto review"]
        );
        // An empty link in the chain is a parse error, not a silent skip.
        let crontab = parse_crontab("0 6 * * * auto review && && export auto review");
        assert!(crontab.jobs.is_empty());
        assert_eq!(crontab.invalid.len(), 1);
        assert_eq!(crontab.invalid[0].error, "empty command in && chain");
    }

    #[test]
    fn enqueue_due_jobs_only_reacts_to_minute_boundaries() {
        // Same minute: nothing happens, the marker doesn't move.
        let now = crate::session_store::current_unix_timestamp() / 60;
        let mut last = now;
        let mut pending = VecDeque::new();
        enqueue_due_jobs(&mut last, &mut pending);
        assert_eq!(last, now);
        assert!(pending.is_empty());
    }
}
