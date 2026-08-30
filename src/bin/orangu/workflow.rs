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

//! Sequential execution of a fully validated YAML workflow.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use anyhow::{Context, Result, anyhow};
use orangu::{config::ClientAppConfiguration, skills::SkillRegistry};

use crate::{
    r#loop,
    oneshot::{OneshotContext, OneshotSession, validate_workflow_input},
};

/// Check every explicit command before any job starts. Natural-language steps
/// remain model prompts; slash commands must name either a built-in command or
/// a skill visible from that job's workspace, and must be runnable headlessly.
pub(crate) fn validate(plan: &WorkflowPlan) -> Result<()> {
    let mut failures = Vec::new();

    for job in &plan.jobs {
        let skills = SkillRegistry::discover(&job.workspace);
        let mut commands = Vec::new();
        for (name, body) in &job.functions {
            collect_commands(body, &format!("function '{name}'"), &mut commands);
        }
        collect_commands(&job.steps, "", &mut commands);
        for (location, command) in commands {
            if let Err(error) = validate_workflow_input(command, &skills) {
                failures.push(format!("job '{}' {location}: {error}", job.name));
            }
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "workflow command validation failed:\n- {}",
            failures.join("\n- ")
        ))
    }
}

/// Every runnable command in a step tree: plain commands plus the `if` and
/// `while` conditions that execute at runtime. Locations read like
/// `step 2 then 1` so failures point at the offending nested step.
fn collect_commands<'a>(
    steps: &'a [WorkflowStep],
    scope: &str,
    output: &mut Vec<(String, &'a String)>,
) {
    for (index, step) in steps.iter().enumerate() {
        let here = if scope.is_empty() {
            format!("step {}", index + 1)
        } else {
            format!("{scope} step {}", index + 1)
        };
        match step {
            WorkflowStep::Command(command) => output.push((here, command)),
            WorkflowStep::If {
                condition,
                then_branch,
                else_branch,
            } => {
                output.push((format!("{here} condition"), condition));
                collect_commands(then_branch, &format!("{here} then"), output);
                collect_commands(else_branch, &format!("{here} else"), output);
            }
            WorkflowStep::For { body, .. } => {
                collect_commands(body, &format!("{here} for"), output);
            }
            WorkflowStep::While {
                condition, body, ..
            } => {
                output.push((format!("{here} condition"), condition));
                collect_commands(body, &format!("{here} while"), output);
            }
            WorkflowStep::Call(_)
            | WorkflowStep::Approved(_)
            | WorkflowStep::Loop(_)
            | WorkflowStep::Break
            | WorkflowStep::Return
            | WorkflowStep::Label(_)
            | WorkflowStep::Goto(_) => {}
        }
    }
}

/// Run jobs and their steps in declaration order. A job owns one session,
/// so all model prompts in that job share conversation history. Control-flow
/// steps interpret at execution time: a condition is an ordinary workflow
/// command whose success means true, `for` iterates its finite items, and
/// `while` re-checks its condition up to its validated `max_turns`.
pub(crate) async fn run(
    plan: WorkflowPlan,
    config: ClientAppConfiguration,
    config_path: PathBuf,
    quiet: bool,
) -> Result<()> {
    for job in plan.jobs {
        if !quiet {
            eprintln!(
                "workflow: job '{}' (role {}, workspace {})",
                job.name,
                job.role,
                job.workspace.display()
            );
        }
        let mut runtime = JobRuntime {
            job_name: job.name.clone(),
            workspace: job.workspace.clone(),
            role: job.role.clone(),
            functions: job.functions,
            config: config.clone(),
            config_path: config_path.clone(),
            quiet,
            session: None,
        };
        let steps = job.steps;
        match run_block(&steps, &mut runtime).await? {
            Signal::Continue | Signal::Return => {}
            Signal::Break => {
                return Err(anyhow!(
                    "workflow job '{}' has break outside a for or while loop",
                    runtime.job_name
                ));
            }
        }
    }

    Ok(())
}

/// What a step list evaluates to: keep going, or unwind to a loop (`Break`),
/// a caller (`Return`), or a same-list label (`Goto`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Signal {
    Continue,
    Break,
    Return,
}

struct JobRuntime {
    job_name: String,
    workspace: PathBuf,
    role: String,
    functions: std::collections::BTreeMap<String, Vec<WorkflowStep>>,
    config: ClientAppConfiguration,
    config_path: PathBuf,
    quiet: bool,
    session: Option<OneshotSession>,
}

impl JobRuntime {
    async fn session(&mut self) -> Result<&mut OneshotSession> {
        if self.session.is_none() {
            self.session = Some(
                OneshotSession::new(OneshotContext {
                    config: self.config.clone(),
                    config_path: self.config_path.clone(),
                    workspace: self.workspace.clone(),
                    role: Some(self.role.clone()),
                    quiet: self.quiet,
                })
                .await
                .with_context(|| format!("workflow job '{}' could not start", self.job_name))?,
            );
        }
        Ok(self
            .session
            .as_mut()
            .expect("workflow session was initialized"))
    }
}

/// Interpret one step list. Labels only address this list: a `goto` jumps to
/// a label from the same steps, so control never implicitly enters a nested
/// block. `Break` and `Return` propagate to their loop or caller.
fn run_block<'a>(
    steps: &'a [WorkflowStep],
    runtime: &'a mut JobRuntime,
) -> Pin<Box<dyn Future<Output = Result<Signal>> + 'a>> {
    Box::pin(run_block_inner(steps, runtime))
}

async fn run_block_inner(steps: &[WorkflowStep], runtime: &mut JobRuntime) -> Result<Signal> {
    let mut labels = std::collections::HashMap::new();
    for (index, step) in steps.iter().enumerate() {
        if let WorkflowStep::Label(name) = step {
            labels.insert(name.clone(), index);
        }
    }
    let mut pc = 0;
    while pc < steps.len() {
        let step = &steps[pc];
        match step {
            WorkflowStep::Approved(path) => {
                if !runtime.quiet {
                    eprintln!("workflow: approved {}", path.display());
                }
            }
            WorkflowStep::Label(_) => {}
            WorkflowStep::Goto(label) => {
                pc = *labels.get(label).with_context(|| {
                    format!(
                        "workflow job '{}' jumps to unknown label '{label}'",
                        runtime.job_name
                    )
                })?;
                continue;
            }
            WorkflowStep::Break => return Ok(Signal::Break),
            WorkflowStep::Return => return Ok(Signal::Return),
            WorkflowStep::Command(command) => {
                if !runtime.quiet {
                    eprintln!("workflow: job '{}': {command}", runtime.job_name);
                }
                let job = runtime.job_name.clone();
                runtime
                    .session()
                    .await?
                    .run(command)
                    .await
                    .with_context(|| format!("workflow job '{job}' failed at: {command}"))?;
            }
            WorkflowStep::Call(name) => {
                if !runtime.quiet {
                    eprintln!("workflow: job '{}': call {name}", runtime.job_name);
                }
                let body = runtime.functions.get(name).cloned().with_context(|| {
                    format!(
                        "workflow job '{}' calls unknown function '{name}'",
                        runtime.job_name
                    )
                })?;
                match run_block(&body, runtime).await? {
                    Signal::Continue | Signal::Return => {}
                    Signal::Break => return Ok(Signal::Break),
                }
            }
            WorkflowStep::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let branch = if eval_condition(condition, runtime).await? {
                    then_branch
                } else {
                    else_branch
                };
                match run_block(branch, runtime).await? {
                    Signal::Continue => {}
                    Signal::Break => return Ok(Signal::Break),
                    Signal::Return => return Ok(Signal::Return),
                }
            }
            WorkflowStep::For { var, items, body } => {
                for item in items {
                    let expanded = body
                        .iter()
                        .map(|step| substitute(step, var, item))
                        .collect::<Vec<_>>();
                    match run_block(&expanded, runtime).await? {
                        Signal::Continue => {}
                        Signal::Break => break,
                        Signal::Return => return Ok(Signal::Return),
                    }
                }
            }
            WorkflowStep::While {
                condition,
                max_turns,
                body,
            } => {
                for _ in 0..*max_turns {
                    if !eval_condition(condition, runtime).await? {
                        break;
                    }
                    match run_block(body, runtime).await? {
                        Signal::Continue => {}
                        Signal::Break => break,
                        Signal::Return => return Ok(Signal::Return),
                    }
                }
            }
            WorkflowStep::Loop(spec) => {
                if !runtime.quiet {
                    eprintln!("workflow: job '{}': code-and-review loop", runtime.job_name);
                }
                let job = runtime.job_name.clone();
                r#loop::run_workflow(
                    spec.clone(),
                    r#loop::LoopRunContext {
                        config: runtime.config.clone(),
                        workspace: runtime.workspace.clone(),
                        role: Some(runtime.role.clone()),
                        quiet: runtime.quiet,
                    },
                )
                .await
                .with_context(|| format!("workflow job '{job}' loop failed"))?;
            }
        }
        pc += 1;
    }
    Ok(Signal::Continue)
}

/// Run a condition command in the job session. Success means true; failure
/// means false and selects the other branch instead of failing the job.
async fn eval_condition(condition: &str, runtime: &mut JobRuntime) -> Result<bool> {
    if !runtime.quiet {
        eprintln!("workflow: job '{}': if {condition}", runtime.job_name);
    }
    let outcome = runtime.session().await?.run(condition).await;
    Ok(outcome.is_ok())
}

/// Substitute one `for` iteration value into a compiled step. Only command
/// text, conditions, and loop parameters can carry `${var}`: approvals
/// resolve at validation time and never contain it.
fn substitute(step: &WorkflowStep, var: &str, value: &str) -> WorkflowStep {
    let placeholder = format!("${{{var}}}");
    let fill = |text: &String| text.replace(&placeholder, value);
    match step {
        WorkflowStep::Command(command) => WorkflowStep::Command(fill(command)),
        WorkflowStep::Approved(path) => WorkflowStep::Approved(path.clone()),
        WorkflowStep::Loop(spec) => WorkflowStep::Loop(WorkflowLoop {
            objective: fill(&spec.objective),
            stop: match &spec.stop {
                WorkflowLoopStop::Turns(count) => WorkflowLoopStop::Turns(*count),
                WorkflowLoopStop::Time(duration) => WorkflowLoopStop::Time(*duration),
                WorkflowLoopStop::Goal(condition) => WorkflowLoopStop::Goal(fill(condition)),
            },
            checks: spec.checks.iter().map(fill).collect(),
            rubric: spec.rubric.iter().map(fill).collect(),
        }),
        WorkflowStep::Call(name) => WorkflowStep::Call(name.clone()),
        WorkflowStep::If {
            condition,
            then_branch,
            else_branch,
        } => WorkflowStep::If {
            condition: fill(condition),
            then_branch: then_branch
                .iter()
                .map(|step| substitute(step, var, value))
                .collect(),
            else_branch: else_branch
                .iter()
                .map(|step| substitute(step, var, value))
                .collect(),
        },
        WorkflowStep::For {
            var: inner,
            items,
            body,
        } => WorkflowStep::For {
            var: inner.clone(),
            items: items.iter().map(fill).collect(),
            body: body
                .iter()
                .map(|step| substitute(step, var, value))
                .collect(),
        },
        WorkflowStep::While {
            condition,
            max_turns,
            body,
        } => WorkflowStep::While {
            condition: fill(condition),
            max_turns: *max_turns,
            body: body
                .iter()
                .map(|step| substitute(step, var, value))
                .collect(),
        },
        WorkflowStep::Break => WorkflowStep::Break,
        WorkflowStep::Return => WorkflowStep::Return,
        WorkflowStep::Label(name) => WorkflowStep::Label(name.clone()),
        WorkflowStep::Goto(label) => WorkflowStep::Goto(label.clone()),
    }
}

/// Apply a loop lifecycle action to every job described by the workflow.
pub(crate) async fn manage(
    plan: WorkflowPlan,
    config: ClientAppConfiguration,
    quiet: bool,
    action: &str,
) -> Result<()> {
    for job in plan.jobs {
        if !quiet {
            eprintln!(
                "workflow: {} job '{}' (workspace {})",
                action,
                job.name,
                job.workspace.display()
            );
        }
        r#loop::workflow_action(
            action,
            r#loop::LoopRunContext {
                config: config.clone(),
                workspace: job.workspace,
                role: Some(job.role),
                quiet,
            },
        )
        .await
        .with_context(|| format!("workflow job '{}' {} failed", job.name, action))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };
    use tempfile::NamedTempFile;

    fn serve_non_coordinator(
        listener: TcpListener,
        requests: usize,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            for _ in 0..requests {
                let (mut stream, _) = listener.accept().expect("accept coordinator probe");
                let mut request = Vec::new();
                let mut buffer = [0u8; 1024];
                loop {
                    let read = stream.read(&mut buffer).expect("read coordinator probe");
                    assert!(read > 0, "coordinator probe ended before its headers");
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                assert!(
                    String::from_utf8_lossy(&request).contains("/v1/coordinator"),
                    "unexpected request: {}",
                    String::from_utf8_lossy(&request)
                );
                stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    )
                    .expect("write coordinator response");
            }
        })
    }

    #[tokio::test]
    async fn complete_multi_job_workflow_runs_functions_in_order() {
        let root = tempfile::tempdir().expect("root");
        let output = root.path().join("output");
        std::fs::create_dir(&output).expect("output directory");
        for workspace in ["one", "two"] {
            std::fs::create_dir(root.path().join(workspace)).expect("workspace");
        }

        let yaml = format!(
            r#"orangu:
  version: 1
  variables:
    upstream: >-
      {}
  jobs:
    - job: one
      workspace: one
    - job: two
      workspace: two
  functions:
    create_result:
      - command: /create_file ${{job}}.txt containing created-${{job}}
    move_result:
      - command: /shell mv "${{job}}.txt" "${{upstream}}/${{job}}.txt"
  main:
    - approved: ${{upstream}}
    - call: create_result
    - call: move_result
"#,
            output.display()
        );
        let plan = compile(&yaml, root.path()).expect("compile workflow");
        validate(&plan).expect("preflight workflow");

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake server");
        let endpoint = format!("http://{}/v1", listener.local_addr().expect("address"));
        let server = serve_non_coordinator(listener, plan.jobs.len());
        let config_file = NamedTempFile::new().expect("config");
        std::fs::write(
            config_file.path(),
            format!(
                "[orangu]\nserver = test\nmodel = test-model\ntimeout = 5\n\n[test]\nrole = all\nendpoint = {endpoint}\nmodel = test-model\n"
            ),
        )
        .expect("write config");
        let config =
            orangu::config::load_client_configuration(config_file.path()).expect("load config");

        run(plan, config, config_file.path().to_path_buf(), true)
            .await
            .expect("run workflow");
        server.join().expect("fake server thread");

        for job in ["one", "two"] {
            let result = output.join(format!("{job}.txt"));
            assert!(result.is_file(), "{} was not collected", result.display());
            let content = std::fs::read_to_string(&result).expect("read result");
            assert!(content.contains(&format!("created-{job}")), "{content}");
            assert!(
                !root.path().join(job).join(format!("{job}.txt")).exists(),
                "source should have been moved"
            );
        }
    }

    #[tokio::test]
    async fn control_flow_for_if_while_break_and_goto_run_in_order() {
        let root = tempfile::tempdir().expect("root");
        std::fs::create_dir(root.path().join("repo")).expect("workspace");

        let yaml = r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: repo
  functions:
    make_pair:
      - for:
          var: item
          items: [alpha, beta]
          do:
            - command: /create_file ${item}.txt containing ${item}
      - return: true
  main:
    - call: make_pair
    - for:
        var: name
        items: [first, second]
        do:
          - command: /create_file ${name}.txt containing ${name}
          - break: true
    - if:
        condition: /shell test -f alpha.txt
        then:
          - command: /create_file picked.txt containing then-branch
        else:
          - command: /create_file picked.txt containing else-branch
    - if:
        condition: /shell test -f missing.txt
        then:
          - command: /create_file other.txt containing then-branch
        else:
          - command: /create_file other.txt containing else-branch
    - while:
        condition: /shell test ! -f stop.txt
        max_turns: 5
        do:
          - command: /shell touch stop.txt
    - goto: skip
    - command: /create_file skipped.txt containing skipped
    - label: skip
    - command: /create_file reached.txt containing reached
"#;
        let plan = compile(yaml, root.path()).expect("compile workflow");
        validate(&plan).expect("preflight workflow");

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake server");
        let endpoint = format!("http://{}/v1", listener.local_addr().expect("address"));
        let server = serve_non_coordinator(listener, plan.jobs.len());
        let config_file = NamedTempFile::new().expect("config");
        std::fs::write(
            config_file.path(),
            format!(
                "[orangu]\nserver = test\nmodel = test-model\ntimeout = 5\n\n[test]\nrole = all\nendpoint = {endpoint}\nmodel = test-model\n"
            ),
        )
        .expect("write config");
        let config =
            orangu::config::load_client_configuration(config_file.path()).expect("load config");

        run(plan, config, config_file.path().to_path_buf(), true)
            .await
            .expect("run workflow");
        server.join().expect("fake server thread");

        let repo = root.path().join("repo");
        for (file, content) in [
            ("alpha.txt", "alpha"),
            ("beta.txt", "beta"),
            ("first.txt", "first"),
            ("picked.txt", "then-branch"),
            ("other.txt", "else-branch"),
            ("reached.txt", "reached"),
        ] {
            let text = std::fs::read_to_string(repo.join(file)).expect(file);
            assert!(text.contains(content), "{file}: {text}");
        }
        assert!(repo.join("stop.txt").is_file(), "while body ran once");
        assert!(!repo.join("second.txt").exists(), "break stopped the loop");
        assert!(!repo.join("skipped.txt").exists(), "goto skipped the step");
    }
}

mod language {
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

    //! Parser and whole-file validator for Orangu YAML workflows.
    //!
    //! A workflow is fully parsed, its variables and function calls are expanded
    //! once per job, and every workspace and approved path is checked before the
    //! runner can start the first (potentially long-running) command.

    use serde::Deserialize;
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::error::Error;
    use std::fmt;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    const KNOWN_ROLES: &[&str] = &["all", "code", "review", "explorer", "embeddings"];

    /// A validated workflow, with function calls and variables expanded for each
    /// job. The runner consumes this type rather than the raw YAML.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct WorkflowPlan {
        pub version: u32,
        pub jobs: Vec<WorkflowJob>,
    }

    /// One independent job in a validated workflow.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct WorkflowJob {
        pub name: String,
        pub workspace: PathBuf,
        pub role: String,
        /// The job's `main` steps. Function `call`s resolve against
        /// `functions` at execution time so `return` and `break` behave.
        pub steps: Vec<WorkflowStep>,
        /// Every function compiled for this job, with that job's variables
        /// (and `${job}`) already interpolated.
        pub functions: BTreeMap<String, Vec<WorkflowStep>>,
    }

    /// One compiled workflow operation. Control-flow steps keep their nested
    /// bodies: the runner interprets them at execution time, after every step
    /// (including every branch and condition) has been validated.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum WorkflowStep {
        /// Input for Orangu's existing command/prompt dispatcher.
        Command(String),
        /// Explicit workflow-language permission to pass an existing path outside
        /// the workspace through a variable. It does not widen file-tool roots or
        /// turn the platform shell into an operating-system sandbox.
        Approved(PathBuf),
        /// A bounded tool-enabled work and read-only review cycle.
        Loop(WorkflowLoop),
        /// Run a named function. Recursion is rejected at validation time.
        Call(String),
        /// Run `then_branch` when `condition` succeeds, else `else_branch`.
        /// The condition is an ordinary workflow command: success is true.
        If {
            condition: String,
            then_branch: Vec<WorkflowStep>,
            else_branch: Vec<WorkflowStep>,
        },
        /// Run `body` once per item with `${var}` set to that item.
        For {
            var: String,
            items: Vec<String>,
            body: Vec<WorkflowStep>,
        },
        /// Re-evaluate `condition` before each turn and run `body` while it
        /// succeeds, up to `max_turns` turns.
        While {
            condition: String,
            max_turns: u32,
            body: Vec<WorkflowStep>,
        },
        /// Stop the innermost enclosing `for` or `while` body.
        Break,
        /// Stop the current function and continue after its `call`.
        Return,
        /// A jump target for `goto`. Labels live in the step list that
        /// declares them; a `goto` must name a label from the same list.
        Label(String),
        /// Continue execution at the same-list label with this name.
        Goto(String),
    }

    /// A validated code-and-review loop embedded in a workflow.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct WorkflowLoop {
        pub objective: String,
        pub stop: WorkflowLoopStop,
        pub checks: Vec<String>,
        pub rubric: Vec<String>,
    }

    /// The single condition that bounds a workflow loop.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum WorkflowLoopStop {
        Turns(u32),
        Time(Duration),
        Goal(String),
    }

    /// A syntax or semantic failure found before workflow execution.
    #[derive(Debug)]
    pub enum WorkflowError {
        Yaml(serde_yaml::Error),
        Validation(Vec<String>),
    }

    impl fmt::Display for WorkflowError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Yaml(error) => write!(f, "invalid workflow YAML: {error}"),
                Self::Validation(errors) => {
                    writeln!(f, "workflow validation failed:")?;
                    for error in errors {
                        writeln!(f, "- {error}")?;
                    }
                    Ok(())
                }
            }
        }
    }

    impl Error for WorkflowError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self {
                Self::Yaml(error) => Some(error),
                Self::Validation(_) => None,
            }
        }
    }

    /// Parse and validate a workflow without executing it.
    ///
    /// Relative workspace paths are resolved against `base_dir`. Relative
    /// approved paths are resolved against the job workspace. Every referenced
    /// workspace and approved path must already exist, making this function both
    /// the structural validator and the execution preflight.
    pub fn compile(source: &str, base_dir: &Path) -> Result<WorkflowPlan, WorkflowError> {
        let document: RawDocument = serde_yaml::from_str(source).map_err(WorkflowError::Yaml)?;
        compile_document(document.orangu, base_dir)
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawDocument {
        orangu: RawWorkflow,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawWorkflow {
        version: u32,
        #[serde(default)]
        role: Option<String>,
        #[serde(default)]
        variables: BTreeMap<String, Scalar>,
        #[serde(default)]
        jobs: Vec<RawJob>,
        #[serde(default)]
        functions: BTreeMap<String, Vec<RawStep>>,
        #[serde(default)]
        main: Vec<RawStep>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawJob {
        job: String,
        workspace: String,
        #[serde(default)]
        role: Option<String>,
        #[serde(default)]
        variables: BTreeMap<String, Scalar>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawStep {
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        call: Option<String>,
        #[serde(default)]
        approved: Option<OneOrMany>,
        #[serde(default)]
        r#loop: Option<RawLoop>,
        #[serde(default, rename = "if")]
        r#if: Option<RawIf>,
        #[serde(default, rename = "for")]
        r#for: Option<RawFor>,
        #[serde(default, rename = "while")]
        r#while: Option<RawWhile>,
        #[serde(default, rename = "break")]
        r#break: Option<bool>,
        #[serde(default, rename = "return")]
        r#return: Option<bool>,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        goto: Option<String>,
    }

    /// `if` runs its `then` steps when `condition` — an ordinary workflow
    /// command — succeeds, and its `else` steps otherwise.
    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawIf {
        condition: String,
        then: Vec<RawStep>,
        #[serde(default)]
        r#else: Option<Vec<RawStep>>,
    }

    /// `for` repeats its `do` steps once per item (or once per value of a
    /// finite `range` such as `1..3`), with `${var}` set to that value.
    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawFor {
        var: String,
        #[serde(default)]
        items: Option<Vec<String>>,
        #[serde(default)]
        range: Option<String>,
        #[serde(rename = "do")]
        body: Vec<RawStep>,
    }

    /// `while` re-runs `condition` before every turn and executes `do` while
    /// it succeeds, stopping after `max_turns` turns at the latest.
    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawWhile {
        condition: String,
        max_turns: u32,
        #[serde(rename = "do")]
        body: Vec<RawStep>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawLoop {
        objective: String,
        stop: RawLoopStop,
        #[serde(default)]
        review: RawLoopReview,
    }

    #[derive(Debug, Default, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawLoopReview {
        #[serde(default)]
        checks: Vec<String>,
        #[serde(default)]
        rubric: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
    enum RawLoopStop {
        Turns { count: u32 },
        Time { duration: String },
        Goal { condition: String },
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }

    impl OneOrMany {
        fn values(&self) -> Box<dyn Iterator<Item = &str> + '_> {
            match self {
                Self::One(value) => Box::new(std::iter::once(value.as_str())),
                Self::Many(values) => Box::new(values.iter().map(String::as_str)),
            }
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(untagged)]
    enum Scalar {
        String(String),
        Integer(i64),
        Float(f64),
        Boolean(bool),
    }

    impl Scalar {
        fn text(&self) -> String {
            match self {
                Self::String(value) => value.clone(),
                Self::Integer(value) => value.to_string(),
                Self::Float(value) => value.to_string(),
                Self::Boolean(value) => value.to_string(),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum StepKind<'a> {
        Command(&'a str),
        Call(&'a str),
        Approved(&'a OneOrMany),
        Loop(&'a RawLoop),
        If(&'a RawIf),
        For(&'a RawFor),
        While(&'a RawWhile),
        Break,
        Return,
        Label(&'a str),
        Goto(&'a str),
    }

    /// Upper bounds that keep every loop finite before execution starts.
    const MAX_FOR_ITEMS: usize = 100;
    const MAX_FOR_RANGE_SPAN: u64 = 1000;
    const MAX_WHILE_TURNS: u32 = 1000;

    fn compile_document(raw: RawWorkflow, base_dir: &Path) -> Result<WorkflowPlan, WorkflowError> {
        let mut errors = Vec::new();

        if raw.version != 1 {
            errors.push(format!(
                "unsupported workflow version {}; expected 1",
                raw.version
            ));
        }
        if raw.jobs.is_empty() {
            errors.push("at least one job is required".to_string());
        }
        if raw.main.is_empty() {
            errors.push("main must contain at least one step".to_string());
        }

        validate_role(
            raw.role.as_deref().unwrap_or("all"),
            "global role",
            &mut errors,
        );
        validate_variable_names(&raw.variables, "global variables", &mut errors);
        validate_functions(&raw.functions, &raw.main, &mut errors);

        let mut names = HashSet::new();
        for job in &raw.jobs {
            if job.job.trim().is_empty() {
                errors.push("job names cannot be empty".to_string());
            } else if !names.insert(job.job.as_str()) {
                errors.push(format!("duplicate job name '{}'", job.job));
            }
            validate_variable_names(
                &job.variables,
                &format!("variables for job '{}'", job.job),
                &mut errors,
            );
            if let Some(role) = &job.role {
                validate_role(role, &format!("role for job '{}'", job.job), &mut errors);
            }
        }

        if !errors.is_empty() {
            return Err(WorkflowError::Validation(errors));
        }

        let global_role = raw.role.as_deref().unwrap_or("all");
        let mut jobs = Vec::with_capacity(raw.jobs.len());
        for job in &raw.jobs {
            match compile_job(&raw, job, global_role, base_dir) {
                Ok(compiled) => jobs.push(compiled),
                Err(mut job_errors) => errors.append(&mut job_errors),
            }
        }

        if errors.is_empty() {
            Ok(WorkflowPlan {
                version: raw.version,
                jobs,
            })
        } else {
            Err(WorkflowError::Validation(errors))
        }
    }

    fn compile_job(
        workflow: &RawWorkflow,
        job: &RawJob,
        global_role: &str,
        base_dir: &Path,
    ) -> Result<WorkflowJob, Vec<String>> {
        let mut errors = Vec::new();
        let mut raw_variables = workflow.variables.clone();
        raw_variables.extend(job.variables.clone());

        let variables = match resolve_variables(&raw_variables, &job.job) {
            Ok(variables) => variables,
            Err(mut variable_errors) => {
                for error in &mut variable_errors {
                    *error = format!("job '{}': {error}", job.job);
                }
                errors.append(&mut variable_errors);
                HashMap::new()
            }
        };

        let workspace = match interpolate(&job.workspace, &variables) {
            Ok(path) => match existing_path(&path, base_dir) {
                Ok(path) if path.is_dir() => Some(path),
                Ok(path) => {
                    errors.push(format!(
                        "job '{}': workspace '{}' is not a directory",
                        job.job,
                        path.display()
                    ));
                    None
                }
                Err(error) => {
                    errors.push(format!("job '{}': workspace {error}", job.job));
                    None
                }
            },
            Err(error) => {
                errors.push(format!("job '{}': workspace {error}", job.job));
                None
            }
        };

        // Functions keep their bodies: `call` runs them at execution time so
        // `return` and `break` behave across call boundaries. Approvals apply
        // to later steps in the same function or `main` list, so each scope
        // compiles with its own approval set: a function either approves its
        // own external paths or its caller approves before the `call`.
        let scope = workspace.as_deref().unwrap_or(base_dir);
        let mut functions = BTreeMap::new();
        for (name, body) in &workflow.functions {
            let mut bound = Vec::new();
            match compile_block(body, &variables, scope, &mut bound) {
                Ok(steps) => {
                    functions.insert(name.clone(), steps);
                }
                Err(error) => errors.push(format!("job '{}': function '{name}' {error}", job.job)),
            }
        }
        let mut steps = Vec::new();
        if !variables.is_empty() {
            let mut bound = Vec::new();
            match compile_block(&workflow.main, &variables, scope, &mut bound) {
                Ok(main) => steps = main,
                Err(error) => errors.push(format!("job '{}': {error}", job.job)),
            }
        }

        if errors.is_empty() {
            let scope = workspace
                .as_deref()
                .expect("workspace exists when no errors were recorded");
            let mut approved = HashSet::new();
            let mut bound = Vec::new();
            let mut call_stack = Vec::new();
            if let Err(error) = check_raw_approvals(
                &workflow.main,
                &workflow.functions,
                &variables,
                scope,
                &mut approved,
                &mut bound,
                &mut call_stack,
            ) {
                errors.push(format!("job '{}': {error}", job.job));
            }
        }

        if errors.is_empty() {
            Ok(WorkflowJob {
                name: job.job.clone(),
                workspace: workspace.expect("workspace exists when no errors were recorded"),
                role: job.role.as_deref().unwrap_or(global_role).to_string(),
                steps,
                functions,
            })
        } else {
            Err(errors)
        }
    }

    fn validate_role(role: &str, context: &str, errors: &mut Vec<String>) {
        if !KNOWN_ROLES.contains(&role) {
            errors.push(format!(
                "{context} '{role}' is unknown; expected one of {}",
                KNOWN_ROLES.join(", ")
            ));
        }
    }

    fn validate_variable_names(
        variables: &BTreeMap<String, Scalar>,
        context: &str,
        errors: &mut Vec<String>,
    ) {
        for name in variables.keys() {
            if name == "job" {
                errors.push(format!(
                    "{context}: 'job' is reserved for the current job name"
                ));
            } else if !valid_identifier(name) {
                errors.push(format!(
                "{context}: variable '{name}' must start with a letter or underscore and contain only letters, digits, or underscores"
            ));
            }
        }
    }

    fn valid_identifier(name: &str) -> bool {
        let mut chars = name.chars();
        let Some(first) = chars.next() else {
            return false;
        };
        (first == '_' || first.is_ascii_alphabetic())
            && chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
    }

    fn validate_functions(
        functions: &BTreeMap<String, Vec<RawStep>>,
        main: &[RawStep],
        errors: &mut Vec<String>,
    ) {
        for (name, steps) in functions {
            if !valid_identifier(name) {
                errors.push(format!(
                "function name '{name}' must start with a letter or underscore and contain only letters, digits, or underscores"
            ));
            }
            if steps.is_empty() {
                errors.push(format!("function '{name}' must contain at least one step"));
            }
            validate_block(
                steps,
                &format!("function '{name}'"),
                false,
                true,
                &[],
                errors,
            );
        }
        validate_block(main, "main", false, false, &[], errors);

        let mut visiting = Vec::new();
        let mut visited = HashSet::new();
        for name in functions.keys() {
            visit_function(name, functions, &mut visiting, &mut visited, errors);
        }
        for called in nested_calls(main) {
            if !functions.contains_key(called) {
                errors.push(format!("main calls unknown function '{called}'"));
            }
        }
    }

    /// Every `call` target in a step list, including inside `if` branches and
    /// `for`/`while` bodies. Recursion and unknown calls are rejected before
    /// execution no matter how deeply a call is nested.
    fn nested_calls(steps: &[RawStep]) -> Vec<&str> {
        let mut calls = Vec::new();
        for step in steps {
            match step.kind() {
                Ok(StepKind::Call(name)) => calls.push(name),
                Ok(StepKind::If(body)) => {
                    calls.extend(nested_calls(&body.then));
                    if let Some(other) = &body.r#else {
                        calls.extend(nested_calls(other));
                    }
                }
                Ok(StepKind::For(body)) => calls.extend(nested_calls(&body.body)),
                Ok(StepKind::While(body)) => calls.extend(nested_calls(&body.body)),
                _ => {}
            }
        }
        calls
    }

    /// Validate one step list. `in_loop` tracks lexical nesting inside
    /// `for`/`while` bodies so `break` is accepted only there; `in_function`
    /// tracks function bodies so `return` is rejected in `main`. `bound`
    /// holds the active `for` variables for shadowing checks. Labels and
    /// `goto`s pair up inside the list that declares them: a jump never
    /// crosses a function boundary or enters a nested block implicitly.
    fn validate_block(
        steps: &[RawStep],
        context: &str,
        in_loop: bool,
        in_function: bool,
        bound: &[String],
        errors: &mut Vec<String>,
    ) {
        let mut labels = HashSet::new();
        for (index, step) in steps.iter().enumerate() {
            if let Some(label) = &step.label {
                if label.trim().is_empty() || !valid_identifier(label) {
                    errors.push(format!(
                        "{context} step {} has an invalid label; labels must start with a letter or underscore and contain only letters, digits, or underscores",
                        index + 1
                    ));
                } else if !labels.insert(label.as_str()) {
                    errors.push(format!(
                        "{context} step {} redefines label '{label}'",
                        index + 1
                    ));
                }
            }
        }
        for (index, step) in steps.iter().enumerate() {
            let step_no = index + 1;
            match step.kind() {
                Ok(StepKind::Command(command)) if command.trim().is_empty() => {
                    errors.push(format!("{context} step {step_no} has an empty command"));
                }
                Ok(StepKind::Call(name)) if name.trim().is_empty() => {
                    errors.push(format!("{context} step {step_no} has an empty call"));
                }
                Ok(StepKind::Approved(values))
                    if values.values().next().is_none()
                        || values.values().any(|path| path.trim().is_empty()) =>
                {
                    errors.push(format!(
                        "{context} step {step_no} has an empty approved path"
                    ));
                }
                Ok(StepKind::Loop(spec)) => validate_loop_shape(spec, context, index, errors),
                Ok(StepKind::If(body)) => {
                    if body.condition.trim().is_empty() {
                        errors.push(format!(
                            "{context} step {step_no} has an empty if condition"
                        ));
                    }
                    if body.then.is_empty() {
                        errors.push(format!("{context} step {step_no} has an empty then branch"));
                    } else {
                        validate_block(
                            &body.then,
                            &format!("{context} step {step_no} then"),
                            in_loop,
                            in_function,
                            bound,
                            errors,
                        );
                    }
                    if let Some(other) = &body.r#else {
                        if other.is_empty() {
                            errors
                                .push(format!("{context} step {step_no} has an empty else branch"));
                        } else {
                            validate_block(
                                other,
                                &format!("{context} step {step_no} else"),
                                in_loop,
                                in_function,
                                bound,
                                errors,
                            );
                        }
                    }
                }
                Ok(StepKind::For(body)) => {
                    if body.var == "job" {
                        errors.push(format!(
                            "{context} step {step_no}: 'job' is reserved for the current job name"
                        ));
                    } else if !valid_identifier(&body.var) {
                        errors.push(format!(
                            "{context} step {step_no} has an invalid for variable '{}'; variables must start with a letter or underscore and contain only letters, digits, or underscores",
                            body.var
                        ));
                    } else if bound.iter().any(|name| name == &body.var) {
                        errors.push(format!(
                            "{context} step {step_no} shadows the outer for variable '{}'",
                            body.var
                        ));
                    }
                    match (&body.items, &body.range) {
                        (Some(_), Some(_)) => errors.push(format!(
                            "{context} step {step_no} for accepts either items or range, not both"
                        )),
                        (None, None) => errors
                            .push(format!("{context} step {step_no} for needs items or range")),
                        (Some(items), None) => {
                            if items.is_empty() {
                                errors.push(format!(
                                    "{context} step {step_no} for needs at least one item"
                                ));
                            } else if items.len() > MAX_FOR_ITEMS {
                                errors.push(format!(
                                    "{context} step {step_no} for has {} items; at most {MAX_FOR_ITEMS} are allowed",
                                    items.len()
                                ));
                            } else if items.iter().any(|item| item.trim().is_empty()) {
                                errors.push(format!(
                                    "{context} step {step_no} for has an empty item"
                                ));
                            }
                        }
                        (None, Some(range)) => {
                            if expand_range(range).is_err() {
                                errors.push(format!(
                                    "{context} step {step_no} has an invalid for range '{range}'; use Start..End with at most {MAX_FOR_RANGE_SPAN} values"
                                ));
                            }
                        }
                    }
                    if body.body.is_empty() {
                        errors.push(format!("{context} step {step_no} has an empty for body"));
                    } else {
                        let mut inner = bound.to_vec();
                        inner.push(body.var.clone());
                        validate_block(
                            &body.body,
                            &format!("{context} step {step_no} for"),
                            true,
                            in_function,
                            &inner,
                            errors,
                        );
                    }
                }
                Ok(StepKind::While(body)) => {
                    if body.condition.trim().is_empty() {
                        errors.push(format!(
                            "{context} step {step_no} has an empty while condition"
                        ));
                    }
                    if body.max_turns == 0 || body.max_turns > MAX_WHILE_TURNS {
                        errors.push(format!(
                            "{context} step {step_no} while max_turns must be between 1 and {MAX_WHILE_TURNS}"
                        ));
                    }
                    if body.body.is_empty() {
                        errors.push(format!("{context} step {step_no} has an empty while body"));
                    } else {
                        validate_block(
                            &body.body,
                            &format!("{context} step {step_no} while"),
                            true,
                            in_function,
                            bound,
                            errors,
                        );
                    }
                }
                Ok(StepKind::Break) => {
                    if !in_loop {
                        errors.push(format!(
                            "{context} step {step_no} has break outside a for or while loop"
                        ));
                    }
                }
                Ok(StepKind::Return) => {
                    if !in_function {
                        errors.push(format!(
                            "{context} step {step_no} has return outside a function"
                        ));
                    }
                }
                Ok(StepKind::Label(_)) => {}
                Ok(StepKind::Goto(target)) => {
                    if target.trim().is_empty() || !valid_identifier(target) {
                        errors.push(format!(
                            "{context} step {step_no} has an invalid goto label"
                        ));
                    } else if !labels.contains(target) {
                        errors.push(format!(
                            "{context} step {step_no} jumps to unknown label '{target}' in the same step list"
                        ));
                    }
                }
                Ok(_) => {}
                Err(error) => errors.push(format!("{context} step {step_no} {error}")),
            }
        }
    }

    /// Expand a finite `Start..End` range into its values. Bounds are checked
    /// by the validator; this is the shared expansion used at compile time.
    fn expand_range(range: &str) -> Result<Vec<String>, ()> {
        let (start, end) = range.split_once("..").ok_or(())?;
        let start: u64 = start.trim().parse().map_err(|_| ())?;
        let end: u64 = end.trim().parse().map_err(|_| ())?;
        if start > end || end - start + 1 > MAX_FOR_RANGE_SPAN {
            return Err(());
        }
        Ok((start..=end).map(|value| value.to_string()).collect())
    }

    fn validate_loop_shape(spec: &RawLoop, context: &str, index: usize, errors: &mut Vec<String>) {
        let step = index + 1;
        if spec.objective.trim().is_empty() {
            errors.push(format!("{context} step {step} has an empty loop objective"));
        }
        match &spec.stop {
            RawLoopStop::Turns { count: 0 } => errors.push(format!(
                "{context} step {step} loop turn count must be greater than zero"
            )),
            RawLoopStop::Time { duration } if duration.trim().is_empty() => {
                errors.push(format!("{context} step {step} has an empty loop duration"))
            }
            RawLoopStop::Goal { condition } if condition.trim().is_empty() => errors.push(format!(
                "{context} step {step} has an empty loop goal condition"
            )),
            _ => {}
        }
        if spec
            .review
            .checks
            .iter()
            .any(|check| check.trim().is_empty())
        {
            errors.push(format!(
                "{context} step {step} has an empty loop validation command"
            ));
        }
        if spec
            .review
            .rubric
            .iter()
            .any(|criterion| criterion.trim().is_empty())
        {
            errors.push(format!(
                "{context} step {step} has an empty loop review criterion"
            ));
        }
    }

    fn visit_function<'a>(
        name: &'a str,
        functions: &'a BTreeMap<String, Vec<RawStep>>,
        visiting: &mut Vec<&'a str>,
        visited: &mut HashSet<&'a str>,
        errors: &mut Vec<String>,
    ) {
        if visited.contains(name) {
            return;
        }
        if let Some(position) = visiting.iter().position(|candidate| *candidate == name) {
            let mut cycle = visiting[position..].to_vec();
            cycle.push(name);
            errors.push(format!("recursive function call: {}", cycle.join(" -> ")));
            return;
        }
        let Some(steps) = functions.get(name) else {
            return;
        };
        visiting.push(name);
        for called in nested_calls(steps) {
            if functions.contains_key(called) {
                visit_function(called, functions, visiting, visited, errors);
            } else {
                errors.push(format!(
                    "function '{name}' calls unknown function '{called}'"
                ));
            }
        }
        visiting.pop();
        visited.insert(name);
    }

    impl RawStep {
        fn kind(&self) -> Result<StepKind<'_>, &'static str> {
            const OPTIONS: &str = "must contain exactly one of 'command', 'call', 'approved', 'loop', 'if', 'for', 'while', 'break', 'return', 'label', or 'goto'";
            let count = usize::from(self.command.is_some())
                + usize::from(self.call.is_some())
                + usize::from(self.approved.is_some())
                + usize::from(self.r#loop.is_some())
                + usize::from(self.r#if.is_some())
                + usize::from(self.r#for.is_some())
                + usize::from(self.r#while.is_some())
                + usize::from(self.r#break.is_some())
                + usize::from(self.r#return.is_some())
                + usize::from(self.label.is_some())
                + usize::from(self.goto.is_some());
            if count != 1 {
                return Err(OPTIONS);
            }
            if let Some(command) = &self.command {
                Ok(StepKind::Command(command))
            } else if let Some(call) = &self.call {
                Ok(StepKind::Call(call))
            } else if let Some(approved) = &self.approved {
                Ok(StepKind::Approved(approved))
            } else if let Some(r#loop) = &self.r#loop {
                Ok(StepKind::Loop(r#loop))
            } else if let Some(r#if) = &self.r#if {
                Ok(StepKind::If(r#if))
            } else if let Some(r#for) = &self.r#for {
                Ok(StepKind::For(r#for))
            } else if let Some(r#while) = &self.r#while {
                Ok(StepKind::While(r#while))
            } else if let Some(r#break) = &self.r#break {
                if *r#break {
                    Ok(StepKind::Break)
                } else {
                    Err("break takes no value; write '- break: true'")
                }
            } else if let Some(r#return) = &self.r#return {
                if *r#return {
                    Ok(StepKind::Return)
                } else {
                    Err("return takes no value; write '- return: true'")
                }
            } else if let Some(label) = &self.label {
                Ok(StepKind::Label(label))
            } else {
                Ok(StepKind::Goto(
                    self.goto.as_ref().expect("action count checked"),
                ))
            }
        }
    }

    fn resolve_variables(
        raw: &BTreeMap<String, Scalar>,
        job: &str,
    ) -> Result<HashMap<String, String>, Vec<String>> {
        let mut resolved = HashMap::new();
        resolved.insert("job".to_string(), job.to_string());
        let mut errors = Vec::new();
        for name in raw.keys() {
            let mut stack = Vec::new();
            if let Err(error) = resolve_variable(name, raw, &mut resolved, &mut stack) {
                errors.push(error);
            }
        }
        if errors.is_empty() {
            Ok(resolved)
        } else {
            errors.sort();
            errors.dedup();
            Err(errors)
        }
    }

    fn resolve_variable(
        name: &str,
        raw: &BTreeMap<String, Scalar>,
        resolved: &mut HashMap<String, String>,
        stack: &mut Vec<String>,
    ) -> Result<String, String> {
        if let Some(value) = resolved.get(name) {
            return Ok(value.clone());
        }
        if let Some(position) = stack.iter().position(|candidate| candidate == name) {
            let mut cycle = stack[position..].to_vec();
            cycle.push(name.to_string());
            return Err(format!(
                "recursive variable reference: {}",
                cycle.join(" -> ")
            ));
        }
        let value = raw
            .get(name)
            .ok_or_else(|| format!("undefined variable '{name}'"))?;
        stack.push(name.to_string());
        let text = interpolate_with(&value.text(), |referenced| {
            resolve_variable(referenced, raw, resolved, stack)
        })?;
        stack.pop();
        resolved.insert(name.to_string(), text.clone());
        Ok(text)
    }

    fn interpolate(input: &str, variables: &HashMap<String, String>) -> Result<String, String> {
        interpolate_with(input, |name| {
            variables
                .get(name)
                .cloned()
                .ok_or_else(|| format!("references undefined variable '{name}'"))
        })
    }

    fn interpolate_with(
        input: &str,
        mut resolve: impl FnMut(&str) -> Result<String, String>,
    ) -> Result<String, String> {
        let mut output = String::with_capacity(input.len());
        let mut rest = input;
        while let Some(start) = rest.find("${") {
            output.push_str(&rest[..start]);
            let expression = &rest[start + 2..];
            let Some(end) = expression.find('}') else {
                return Err(format!(
                    "contains an unterminated variable reference: '{input}'"
                ));
            };
            let name = &expression[..end];
            if !valid_identifier(name) {
                return Err(format!("contains invalid variable reference '${{{name}}}'"));
            }
            output.push_str(&resolve(name)?);
            rest = &expression[end + 1..];
        }
        output.push_str(rest);
        Ok(output)
    }

    /// Compile one step list into its executable tree. `bound` holds the
    /// active `for` variables, whose `${var}` references are left for
    /// execution time to substitute. External-path approvals are checked
    /// separately by [`check_approvals`], which walks the compiled tree in
    /// execution order.
    fn compile_block(
        raw_steps: &[RawStep],
        variables: &HashMap<String, String>,
        workspace: &Path,
        bound: &mut Vec<String>,
    ) -> Result<Vec<WorkflowStep>, String> {
        let mut output = Vec::with_capacity(raw_steps.len());
        for step in raw_steps {
            match step.kind().map_err(str::to_string)? {
                StepKind::Command(command) => {
                    let command = interpolate_scoped(command, variables, bound)?;
                    output.push(WorkflowStep::Command(command));
                }
                StepKind::Approved(paths) => {
                    for path in paths.values() {
                        for name in variable_references(path)? {
                            if bound.iter().any(|active| active == name) {
                                return Err(format!(
                                    "approved paths cannot use the for-loop variable '${{{name}}}'; approve a concrete path before the loop"
                                ));
                            }
                        }
                        let path = interpolate_scoped(path, variables, bound)?;
                        let path = existing_path(&path, workspace)
                            .map_err(|error| format!("approved path {error}"))?;
                        output.push(WorkflowStep::Approved(path));
                    }
                }
                StepKind::Loop(spec) => output.push(WorkflowStep::Loop(compile_loop(
                    spec, variables, workspace, bound,
                )?)),
                StepKind::Call(name) => output.push(WorkflowStep::Call(name.to_string())),
                StepKind::If(body) => {
                    let condition = interpolate_scoped(&body.condition, variables, bound)?;
                    if condition.trim().is_empty() {
                        return Err("has an empty if condition".to_string());
                    }
                    let then_branch = compile_block(&body.then, variables, workspace, bound)?;
                    let else_branch = match &body.r#else {
                        Some(other) => compile_block(other, variables, workspace, bound)?,
                        None => Vec::new(),
                    };
                    output.push(WorkflowStep::If {
                        condition,
                        then_branch,
                        else_branch,
                    });
                }
                StepKind::For(body) => {
                    let items = match (&body.items, &body.range) {
                        (Some(items), None) => items
                            .iter()
                            .map(|item| interpolate_scoped(item, variables, bound))
                            .collect::<Result<Vec<_>, _>>()?,
                        (None, Some(range)) => {
                            let range = interpolate_scoped(range, variables, bound)?;
                            expand_range(&range)
                                .map_err(|_| format!("has an invalid for range '{range}'"))?
                        }
                        _ => return Err("for needs items or range".to_string()),
                    };
                    if items.iter().any(|item| item.trim().is_empty()) {
                        return Err("for has an empty item".to_string());
                    }
                    bound.push(body.var.clone());
                    let compiled = compile_block(&body.body, variables, workspace, bound)?;
                    bound.pop();
                    output.push(WorkflowStep::For {
                        var: body.var.clone(),
                        items,
                        body: compiled,
                    });
                }
                StepKind::While(body) => {
                    let condition = interpolate_scoped(&body.condition, variables, bound)?;
                    if condition.trim().is_empty() {
                        return Err("has an empty while condition".to_string());
                    }
                    let compiled = compile_block(&body.body, variables, workspace, bound)?;
                    output.push(WorkflowStep::While {
                        condition,
                        max_turns: body.max_turns,
                        body: compiled,
                    });
                }
                StepKind::Break => output.push(WorkflowStep::Break),
                StepKind::Return => output.push(WorkflowStep::Return),
                StepKind::Label(name) => output.push(WorkflowStep::Label(name.to_string())),
                StepKind::Goto(target) => output.push(WorkflowStep::Goto(target.to_string())),
            }
        }
        Ok(output)
    }

    /// Check external variable paths in execution order on the raw steps,
    /// descending into function calls and both `if` branches. An `approved`
    /// step authorizes later uses wherever the run can reach them — including
    /// inside a called function or after it returns — exactly as if the calls
    /// were inlined. Branch and loop bodies share the scope's approval set:
    /// validation is conservative and accepts an approval that precedes the
    /// branch or loop. Loop checks obey the same ordering: an approval must
    /// precede the `loop` step whose checks use the path. Raw text is checked
    /// before interpolation so `${var}` references are still visible.
    fn check_raw_approvals(
        raw_steps: &[RawStep],
        functions: &BTreeMap<String, Vec<RawStep>>,
        variables: &HashMap<String, String>,
        workspace: &Path,
        approved: &mut HashSet<PathBuf>,
        bound: &mut Vec<String>,
        call_stack: &mut Vec<String>,
    ) -> Result<(), String> {
        for step in raw_steps {
            match step.kind().map_err(str::to_string)? {
                StepKind::Command(command) => {
                    validate_external_variable_paths(
                        command, variables, workspace, approved, bound,
                    )?;
                }
                StepKind::Approved(paths) => {
                    for path in paths.values() {
                        let path = interpolate_scoped(path, variables, bound)?;
                        let path = existing_path(&path, workspace)
                            .map_err(|error| format!("approved path {error}"))?;
                        approved.insert(path);
                    }
                }
                StepKind::Loop(spec) => {
                    for check in &spec.review.checks {
                        validate_external_variable_paths(
                            check, variables, workspace, approved, bound,
                        )?;
                    }
                }
                StepKind::Call(name) => {
                    if call_stack.iter().any(|called| called == name) {
                        let mut cycle = call_stack.clone();
                        cycle.push(name.to_string());
                        return Err(format!("recursive function call: {}", cycle.join(" -> ")));
                    }
                    let body = functions
                        .get(name)
                        .ok_or_else(|| format!("calls unknown function '{name}'"))?;
                    call_stack.push(name.to_string());
                    check_raw_approvals(
                        body, functions, variables, workspace, approved, bound, call_stack,
                    )?;
                    call_stack.pop();
                }
                StepKind::If(body) => {
                    validate_external_variable_paths(
                        &body.condition,
                        variables,
                        workspace,
                        approved,
                        bound,
                    )?;
                    check_raw_approvals(
                        &body.then, functions, variables, workspace, approved, bound, call_stack,
                    )?;
                    if let Some(other) = &body.r#else {
                        check_raw_approvals(
                            other, functions, variables, workspace, approved, bound, call_stack,
                        )?;
                    }
                }
                StepKind::For(body) => {
                    bound.push(body.var.clone());
                    check_raw_approvals(
                        &body.body, functions, variables, workspace, approved, bound, call_stack,
                    )?;
                    bound.pop();
                }
                StepKind::While(body) => {
                    validate_external_variable_paths(
                        &body.condition,
                        variables,
                        workspace,
                        approved,
                        bound,
                    )?;
                    check_raw_approvals(
                        &body.body, functions, variables, workspace, approved, bound, call_stack,
                    )?;
                }
                StepKind::Break | StepKind::Return | StepKind::Label(_) | StepKind::Goto(_) => {}
            }
        }
        Ok(())
    }

    /// Like [`interpolate`], but references to active `for` variables are
    /// left as `${var}` placeholders for execution time to substitute.
    fn interpolate_scoped(
        input: &str,
        variables: &HashMap<String, String>,
        bound: &[String],
    ) -> Result<String, String> {
        interpolate_with(input, |name| {
            if bound.iter().any(|active| active == name) {
                Ok(format!("${{{name}}}"))
            } else {
                variables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("references undefined variable '{name}'"))
            }
        })
    }

    /// Paths supplied through variables are the part of a shell command the YAML
    /// layer can identify without attempting to parse a shell language. An
    /// absolute variable that resolves outside the workspace therefore needs a
    /// preceding `approved` step. Literal shell text keeps the existing `/shell`
    /// semantics; the workflow layer does not pretend to be a portable shell
    /// sandbox.
    fn validate_external_variable_paths(
        command: &str,
        variables: &HashMap<String, String>,
        workspace: &Path,
        approved: &HashSet<PathBuf>,
        bound: &[String],
    ) -> Result<(), String> {
        for name in variable_references(command)? {
            if bound.iter().any(|active| active == name) {
                continue;
            }
            let Some(value) = variables.get(name) else {
                continue;
            };
            let expanded = expand_home(value)?;
            let path = PathBuf::from(expanded);
            let path = if path.is_absolute() {
                path
            } else {
                workspace.join(path)
            };
            if !path.exists() {
                continue;
            }
            let canonical = path
                .canonicalize()
                .map_err(|error| format!("variable '{name}' path '{}': {error}", path.display()))?;
            let explicitly_approved = approved.iter().any(|path| canonical.starts_with(path));
            if !canonical.starts_with(workspace) && !explicitly_approved {
                return Err(format!(
                    "command uses variable '{name}' outside the workspace; add an approved step for '{}' before the command",
                    canonical.display()
                ));
            }
        }
        Ok(())
    }

    fn variable_references(input: &str) -> Result<Vec<&str>, String> {
        let mut references = Vec::new();
        let mut rest = input;
        while let Some(start) = rest.find("${") {
            let expression = &rest[start + 2..];
            let Some(end) = expression.find('}') else {
                return Err(format!(
                    "contains an unterminated variable reference: '{input}'"
                ));
            };
            let name = &expression[..end];
            if !valid_identifier(name) {
                return Err(format!("contains invalid variable reference '${{{name}}}'"));
            }
            references.push(name);
            rest = &expression[end + 1..];
        }
        Ok(references)
    }

    fn compile_loop(
        raw: &RawLoop,
        variables: &HashMap<String, String>,
        _workspace: &Path,
        bound: &[String],
    ) -> Result<WorkflowLoop, String> {
        let objective = interpolate_scoped(&raw.objective, variables, bound)?;
        if objective.trim().is_empty() {
            return Err("has an empty loop objective".to_string());
        }
        let stop = match &raw.stop {
            RawLoopStop::Turns { count } if *count > 0 => WorkflowLoopStop::Turns(*count),
            RawLoopStop::Turns { .. } => {
                return Err("loop turn count must be greater than zero".to_string());
            }
            RawLoopStop::Time { duration } => {
                let duration = interpolate_scoped(duration, variables, bound)?;
                WorkflowLoopStop::Time(parse_duration(&duration)?)
            }
            RawLoopStop::Goal { condition } => {
                let condition = interpolate_scoped(condition, variables, bound)?;
                if condition.trim().is_empty() {
                    return Err("has an empty loop goal condition".to_string());
                }
                WorkflowLoopStop::Goal(condition)
            }
        };

        let mut checks = Vec::with_capacity(raw.review.checks.len());
        for check in &raw.review.checks {
            checks.push(interpolate_scoped(check, variables, bound)?);
        }
        let rubric = raw
            .review
            .rubric
            .iter()
            .map(|criterion| interpolate_scoped(criterion, variables, bound))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(WorkflowLoop {
            objective,
            stop,
            checks,
            rubric,
        })
    }

    fn parse_duration(input: &str) -> Result<Duration, String> {
        let input = input.trim();
        let Some(index) = input.find(|character: char| !character.is_ascii_digit()) else {
            return Err(format!(
                "invalid loop duration '{input}'; use a positive value such as 30m or 2h"
            ));
        };
        let (number, unit) = input.split_at(index);
        if number.is_empty()
            || unit.is_empty()
            || unit.chars().any(|character| character.is_whitespace())
        {
            return Err(format!(
                "invalid loop duration '{input}'; use a positive value such as 30m or 2h"
            ));
        }
        let value: u64 = number
            .parse()
            .map_err(|_| format!("invalid loop duration '{input}'"))?;
        if value == 0 {
            return Err("loop duration must be greater than zero".to_string());
        }
        let seconds = match unit {
            "s" => value,
            "m" => value
                .checked_mul(60)
                .ok_or_else(|| "loop duration is too large".to_string())?,
            "h" => value
                .checked_mul(60 * 60)
                .ok_or_else(|| "loop duration is too large".to_string())?,
            "d" => value
                .checked_mul(24 * 60 * 60)
                .ok_or_else(|| "loop duration is too large".to_string())?,
            _ => {
                return Err(format!(
                    "invalid loop duration unit in '{input}'; use s, m, h, or d"
                ));
            }
        };
        Ok(Duration::from_secs(seconds))
    }

    fn existing_path(raw: &str, relative_to: &Path) -> Result<PathBuf, String> {
        let expanded = expand_home(raw)?;
        let path = PathBuf::from(expanded);
        let path = if path.is_absolute() {
            path
        } else {
            relative_to.join(path)
        };
        path.canonicalize()
            .map_err(|error| format!("'{}' cannot be resolved: {error}", path.display()))
    }

    fn expand_home(path: &str) -> Result<String, String> {
        if path == "~" || path.starts_with("~/") {
            let home = home::home_dir()
                .ok_or_else(|| "uses '~' but no home directory is known".to_string())?;
            if path == "~" {
                Ok(home.display().to_string())
            } else {
                Ok(home.join(&path[2..]).display().to_string())
            }
        } else {
            Ok(path.to_string())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;
        use tempfile::tempdir;

        #[test]
        fn compiles_export_workflow_for_every_job() {
            let root = tempdir().expect("root");
            let upstream = root.path().join("Upstream");
            let pgagroal = root.path().join("pgagroal");
            let orangu = root.path().join("orangu");
            fs::create_dir_all(&upstream).expect("upstream");
            fs::create_dir_all(&pgagroal).expect("pgagroal");
            fs::create_dir_all(&orangu).expect("orangu");

            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    upstream: {upstream:?}
    root: {root:?}
  jobs:
    - job: pgagroal
      workspace: ${{root}}/pgagroal
    - job: orangu
      workspace: ${{root}}/orangu
      role: code
  functions:
    export_pr:
      - command: /export pr
    move_pdf:
      - command: /shell mv ${{job}}-pr.pdf ${{upstream}}/
    check_orangu:
      - approved: ${{upstream}}
      - call: export_pr
      - call: move_pdf
  main:
    - call: check_orangu
"#,
                upstream = upstream.display().to_string(),
                root = root.path().display().to_string(),
            );

            let plan = compile(&yaml, root.path()).expect("valid workflow");
            assert_eq!(plan.version, 1);
            assert_eq!(plan.jobs.len(), 2);
            assert_eq!(plan.jobs[0].name, "pgagroal");
            assert_eq!(plan.jobs[0].role, "all");
            assert_eq!(plan.jobs[1].role, "code");
            // Calls stay calls: functions run at execution time so `return`
            // and `break` behave across call boundaries.
            assert_eq!(
                plan.jobs[0].steps,
                vec![WorkflowStep::Call("check_orangu".to_string())]
            );
            let functions = &plan.jobs[0].functions;
            assert_eq!(
                functions["export_pr"],
                vec![WorkflowStep::Command("/export pr".to_string())]
            );
            assert_eq!(
                functions["move_pdf"],
                vec![WorkflowStep::Command(format!(
                    "/shell mv pgagroal-pr.pdf {}/",
                    upstream.display()
                ))]
            );
            assert_eq!(
                functions["check_orangu"],
                vec![
                    WorkflowStep::Approved(upstream.canonicalize().expect("canonical upstream")),
                    WorkflowStep::Call("export_pr".to_string()),
                    WorkflowStep::Call("move_pdf".to_string()),
                ]
            );
        }

        #[test]
        fn job_variables_override_global_variables() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            fs::create_dir(&workspace).expect("workspace");
            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    action: status
  jobs:
    - job: repo
      workspace: {workspace:?}
      variables:
        action: diff
  functions: {{}}
  main:
    - command: /${{action}}
"#,
                workspace = workspace.display().to_string(),
            );
            let plan = compile(&yaml, root.path()).expect("valid workflow");
            assert_eq!(
                plan.jobs[0].steps,
                vec![WorkflowStep::Command("/diff".to_string())]
            );
        }

        #[test]
        fn rejects_unknown_fields_and_malformed_steps() {
            let yaml = r#"orangu:
  version: 1
  surprise: true
  jobs: []
  main: []
"#;
            let error = compile(yaml, Path::new(".")).expect_err("unknown field");
            assert!(error.to_string().contains("unknown field `surprise`"));

            let root = tempdir().expect("root");
            let yaml = format!(
                r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - command: /status
      call: other
"#,
                workspace = root.path().display().to_string(),
            );
            let error = compile(&yaml, root.path()).expect_err("two actions");
            assert!(error.to_string().contains("exactly one of"));
        }

        #[test]
        fn validates_all_jobs_before_returning() {
            let root = tempdir().expect("root");
            let yaml = r#"orangu:
  version: 1
  jobs:
    - job: first
      workspace: missing-one
    - job: second
      workspace: missing-two
  main:
    - command: /export ${missing}
"#;
            let error = compile(yaml, root.path()).expect_err("invalid jobs");
            let message = error.to_string();
            assert!(message.contains("job 'first'"));
            assert!(message.contains("job 'second'"));
            assert!(message.contains("undefined variable 'missing'"));
        }

        #[test]
        fn rejects_unknown_and_recursive_function_calls() {
            let root = tempdir().expect("root");
            let yaml = format!(
                r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: {workspace:?}
  functions:
    first:
      - call: second
    second:
      - call: first
    unused:
      - call: missing
  main:
    - call: first
"#,
                workspace = root.path().display().to_string(),
            );
            let error = compile(&yaml, root.path()).expect_err("invalid calls");
            let message = error.to_string();
            assert!(message.contains("recursive function call"));
            assert!(message.contains("calls unknown function 'missing'"));
        }

        #[test]
        fn rejects_recursive_and_undefined_variables() {
            let root = tempdir().expect("root");
            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    first: ${{second}}
    second: ${{first}}
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - command: /export ${{missing}}
"#,
                workspace = root.path().display().to_string(),
            );
            let error = compile(&yaml, root.path()).expect_err("invalid variables");
            assert!(error.to_string().contains("recursive variable reference"));
        }

        #[test]
        fn external_variable_path_requires_prior_approval() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            let external = root.path().join("output");
            fs::create_dir(&workspace).expect("workspace");
            fs::create_dir(&external).expect("external output");

            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    output: {external:?}
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - command: /shell mv report.pdf ${{output}}/
"#,
                external = external.display().to_string(),
                workspace = workspace.display().to_string(),
            );
            let error = compile(&yaml, root.path()).expect_err("external path is not approved");
            let message = error.to_string();
            assert!(message.contains("variable 'output' outside the workspace"));
            assert!(message.contains("add an approved step"));
        }

        #[test]
        fn external_approval_must_precede_the_command() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            let external = root.path().join("output");
            fs::create_dir(&workspace).expect("workspace");
            fs::create_dir(&external).expect("external output");

            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    output: {external:?}
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - command: /shell mv report.pdf ${{output}}/
    - approved: ${{output}}
"#,
                external = external.display().to_string(),
                workspace = workspace.display().to_string(),
            );
            let error = compile(&yaml, root.path()).expect_err("approval comes too late");
            assert!(error.to_string().contains("before the command"));
        }

        #[test]
        fn relative_variable_cannot_escape_the_workspace() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            let external = root.path().join("output");
            fs::create_dir(&workspace).expect("workspace");
            fs::create_dir(&external).expect("external output");

            let yaml = r#"orangu:
  version: 1
  variables:
    output: ../output
  jobs:
    - job: repo
      workspace: repo
  main:
    - command: /shell mv report.pdf ${output}/
"#;
            let error = compile(yaml, root.path()).expect_err("relative escape is not approved");
            assert!(error.to_string().contains("outside the workspace"));
        }

        #[test]
        fn compiles_all_code_review_loop_stop_policies() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            fs::create_dir(&workspace).expect("workspace");
            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    objective: Fix the parser
    check: cargo test
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - loop:
        objective: "${{objective}}"
        stop:
          type: turns
          count: 3
        review:
          checks: ["${{check}}"]
          rubric: [correctness, regressions]
    - loop:
        objective: Work within the time budget
        stop:
          type: time
          duration: 30m
    - loop:
        objective: Finish the migration
        stop:
          type: goal
          condition: All tests pass
"#,
                workspace = workspace.display().to_string(),
            );

            let plan = compile(&yaml, root.path()).expect("valid loops");
            assert_eq!(
                plan.jobs[0].steps[0],
                WorkflowStep::Loop(WorkflowLoop {
                    objective: "Fix the parser".to_string(),
                    stop: WorkflowLoopStop::Turns(3),
                    checks: vec!["cargo test".to_string()],
                    rubric: vec!["correctness".to_string(), "regressions".to_string()],
                })
            );
            assert!(matches!(
                &plan.jobs[0].steps[1],
                WorkflowStep::Loop(WorkflowLoop {
                    stop: WorkflowLoopStop::Time(duration),
                    ..
                }) if *duration == Duration::from_secs(30 * 60)
            ));
            assert!(matches!(
                &plan.jobs[0].steps[2],
                WorkflowStep::Loop(WorkflowLoop {
                    stop: WorkflowLoopStop::Goal(condition),
                    ..
                }) if condition == "All tests pass"
            ));
        }

        #[test]
        fn rejects_invalid_code_review_loops_before_execution() {
            let root = tempdir().expect("root");
            let yaml = format!(
                r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: {workspace:?}
  functions:
    invalid:
      - loop:
          objective: ""
          stop:
            type: turns
            count: 0
          review:
            checks: [""]
            rubric: [""]
  main:
    - call: invalid
"#,
                workspace = root.path().display().to_string(),
            );
            let message = compile(&yaml, root.path())
                .expect_err("invalid loop")
                .to_string();
            assert!(message.contains("empty loop objective"));
            assert!(message.contains("turn count must be greater than zero"));
            assert!(message.contains("empty loop validation command"));
            assert!(message.contains("empty loop review criterion"));
        }

        #[test]
        fn compiles_control_flow_steps() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            fs::create_dir(&workspace).expect("workspace");
            let yaml = format!(
                r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: {workspace:?}
  functions:
    build_one:
      - command: /build one
      - return: true
      - command: /build unreachable
  main:
    - if:
        condition: /status
        then:
          - call: build_one
        else:
          - command: /diff
    - for:
        var: target
        range: 1..2
        do:
          - command: /build ${{target}}
          - break: true
    - while:
        condition: /status
        max_turns: 3
        do:
          - command: /test
    - label: done
    - goto: done
"#,
                workspace = workspace.display().to_string(),
            );
            let plan = compile(&yaml, root.path()).expect("valid control flow");
            assert!(matches!(
                &plan.jobs[0].steps[0],
                WorkflowStep::If { then_branch, else_branch, .. }
                if then_branch.len() == 1 && else_branch.len() == 1
            ));
            assert!(matches!(
                &plan.jobs[0].steps[1],
                WorkflowStep::For { var, items, .. }
                if var == "target" && items == &vec!["1".to_string(), "2".to_string()]
            ));
            assert!(matches!(
                &plan.jobs[0].steps[2],
                WorkflowStep::While { max_turns: 3, .. }
            ));
            assert_eq!(
                plan.jobs[0].steps[3],
                WorkflowStep::Label("done".to_string())
            );
            assert_eq!(
                plan.jobs[0].steps[4],
                WorkflowStep::Goto("done".to_string())
            );
            assert_eq!(plan.jobs[0].functions["build_one"][1], WorkflowStep::Return);
        }

        #[test]
        fn rejects_misplaced_break_return_and_jumps() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            fs::create_dir(&workspace).expect("workspace");
            let prefix = format!(
                "orangu:\n  version: 1\n  jobs:\n    - job: repo\n      workspace: {workspace:?}\n  main:\n",
                workspace = workspace.display().to_string(),
            );
            for (name, step, message) in [
                (
                    "break",
                    "    - break: true\n",
                    "break outside a for or while loop",
                ),
                (
                    "return",
                    "    - return: true\n",
                    "return outside a function",
                ),
                ("goto", "    - goto: nowhere\n", "unknown label 'nowhere'"),
                (
                    "for",
                    "    - for:\n        var: x\n        do:\n          - command: /status\n",
                    "for needs items or range",
                ),
                (
                    "while",
                    "    - while:\n        condition: /status\n        max_turns: 0\n        do:\n          - command: /status\n",
                    "max_turns must be between 1",
                ),
            ] {
                let yaml = format!("{prefix}{step}");
                let error = compile(&yaml, root.path()).expect_err(name);
                assert!(error.to_string().contains(message), "{name}: {error}");
            }
        }

        #[test]
        fn rejects_shadowed_loop_variables_and_cross_scope_jumps() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            fs::create_dir(&workspace).expect("workspace");
            let yaml = format!(
                r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: {workspace:?}
  functions:
    helper:
      - label: inner
      - command: /status
  main:
    - for:
        var: x
        items: [a]
        do:
          - for:
              var: x
              items: [b]
              do:
                - command: /status
    - goto: inner
"#,
                workspace = workspace.display().to_string(),
            );
            let message = compile(&yaml, root.path())
                .expect_err("shadowing and cross-scope goto")
                .to_string();
            assert!(
                message.contains("shadows the outer for variable 'x'"),
                "{message}"
            );
            assert!(message.contains("unknown label 'inner'"), "{message}");
        }

        #[test]
        fn approvals_flow_through_calls_in_execution_order() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            let external = root.path().join("output");
            fs::create_dir(&workspace).expect("workspace");
            fs::create_dir(&external).expect("external output");
            // An approval authorizes later uses wherever the run reaches them:
            // inside a called function, or after it returns.
            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    output: {external:?}
  jobs:
    - job: repo
      workspace: {workspace:?}
  functions:
    use_output:
      - command: /shell ls ${{output}}/
  main:
    - approved: ${{output}}
    - call: use_output
    - command: /shell ls ${{output}}/
"#,
                external = external.display().to_string(),
                workspace = workspace.display().to_string(),
            );
            let plan = compile(&yaml, root.path()).expect("approvals flow through calls");
            assert_eq!(plan.jobs[0].steps.len(), 3);
        }

        #[test]
        fn rejects_invalid_loop_duration_before_execution() {
            let root = tempdir().expect("root");
            let yaml = format!(
                r#"orangu:
  version: 1
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - loop:
        objective: Fix parser
        stop:
          type: time
          duration: 10weeks
"#,
                workspace = root.path().display().to_string(),
            );
            let message = compile(&yaml, root.path())
                .expect_err("invalid duration")
                .to_string();
            assert!(message.contains("invalid loop duration unit"));
        }

        #[test]
        fn loop_checks_obey_external_path_approval_order() {
            let root = tempdir().expect("root");
            let workspace = root.path().join("repo");
            let external = root.path().join("reports");
            fs::create_dir(&workspace).expect("workspace");
            fs::create_dir(&external).expect("reports");
            let yaml = format!(
                r#"orangu:
  version: 1
  variables:
    reports: {external:?}
  jobs:
    - job: repo
      workspace: {workspace:?}
  main:
    - loop:
        objective: Fix parser
        stop:
          type: turns
          count: 1
        review:
          checks:
            - cp report.txt ${{reports}}/
"#,
                external = external.display().to_string(),
                workspace = workspace.display().to_string(),
            );
            let message = compile(&yaml, root.path())
                .expect_err("unapproved loop check")
                .to_string();
            assert!(message.contains("variable 'reports' outside the workspace"));

            let approved = yaml.replace("  main:\n", "  main:\n    - approved: ${reports}\n");
            compile(&approved, root.path()).expect("approval precedes loop check");
        }
    }
}

pub(crate) use language::{WorkflowLoop, WorkflowLoopStop, WorkflowPlan, WorkflowStep, compile};
