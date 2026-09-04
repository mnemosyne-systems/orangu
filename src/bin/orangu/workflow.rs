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

use std::path::PathBuf;

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
        for (index, step) in job.steps.iter().enumerate() {
            let WorkflowStep::Command(command) = step else {
                continue;
            };
            if let Err(error) = validate_workflow_input(command, &skills) {
                failures.push(format!("job '{}' step {}: {error}", job.name, index + 1));
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

/// Run jobs and their expanded steps in declaration order. A job owns one
/// session, so all model prompts in that job share conversation history.
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

        let mut session = None;

        for (index, step) in job.steps.into_iter().enumerate() {
            match step {
                WorkflowStep::Approved(path) => {
                    if !quiet {
                        eprintln!("workflow: approved {}", path.display());
                    }
                }
                WorkflowStep::Command(command) => {
                    if !quiet {
                        eprintln!(
                            "workflow: job '{}' step {}: {}",
                            job.name,
                            index + 1,
                            command
                        );
                    }
                    if session.is_none() {
                        session = Some(
                            OneshotSession::new(OneshotContext {
                                config: config.clone(),
                                config_path: config_path.clone(),
                                workspace: job.workspace.clone(),
                                role: Some(job.role.clone()),
                                quiet,
                            })
                            .await
                            .with_context(|| {
                                format!("workflow job '{}' could not start", job.name)
                            })?,
                        );
                    }
                    session
                        .as_mut()
                        .expect("workflow session was initialized")
                        .run(&command)
                        .await
                        .with_context(|| {
                            format!("workflow job '{}' step {} failed", job.name, index + 1)
                        })?;
                }
                WorkflowStep::Loop(spec) => {
                    if !quiet {
                        eprintln!(
                            "workflow: job '{}' step {}: code-and-review loop",
                            job.name,
                            index + 1
                        );
                    }
                    r#loop::run_workflow(
                        spec,
                        r#loop::LoopRunContext {
                            config: config.clone(),
                            workspace: job.workspace.clone(),
                            role: Some(job.role.clone()),
                            quiet,
                        },
                    )
                    .await
                    .with_context(|| {
                        format!("workflow job '{}' step {} failed", job.name, index + 1)
                    })?;
                }
            }
        }
    }

    Ok(())
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
        pub steps: Vec<WorkflowStep>,
    }

    /// A primitive operation remaining after named function calls are expanded.
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
    }

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

        let mut steps = Vec::new();
        if !variables.is_empty() {
            let mut stack = Vec::new();
            let mut approved = HashSet::new();
            if let Err(error) = expand_steps(
                &workflow.main,
                &workflow.functions,
                &variables,
                workspace.as_deref().unwrap_or(base_dir),
                &mut stack,
                &mut approved,
                &mut steps,
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
            validate_step_shapes(steps, &format!("function '{name}'"), errors);
        }
        validate_step_shapes(main, "main", errors);

        let mut visiting = Vec::new();
        let mut visited = HashSet::new();
        for name in functions.keys() {
            visit_function(name, functions, &mut visiting, &mut visited, errors);
        }
        for step in main {
            if let Ok(StepKind::Call(name)) = step.kind()
                && !functions.contains_key(name)
            {
                errors.push(format!("main calls unknown function '{name}'"));
            }
        }
    }

    fn validate_step_shapes(steps: &[RawStep], context: &str, errors: &mut Vec<String>) {
        for (index, step) in steps.iter().enumerate() {
            match step.kind() {
                Ok(StepKind::Command(command)) if command.trim().is_empty() => {
                    errors.push(format!("{context} step {} has an empty command", index + 1));
                }
                Ok(StepKind::Call(name)) if name.trim().is_empty() => {
                    errors.push(format!("{context} step {} has an empty call", index + 1));
                }
                Ok(StepKind::Approved(values))
                    if values.values().next().is_none()
                        || values.values().any(|path| path.trim().is_empty()) =>
                {
                    errors.push(format!(
                        "{context} step {} has an empty approved path",
                        index + 1
                    ));
                }
                Ok(StepKind::Loop(spec)) => validate_loop_shape(spec, context, index, errors),
                Ok(_) => {}
                Err(error) => errors.push(format!("{context} step {} {error}", index + 1)),
            }
        }
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
        for step in steps {
            if let Ok(StepKind::Call(called)) = step.kind() {
                if functions.contains_key(called) {
                    visit_function(called, functions, visiting, visited, errors);
                } else {
                    errors.push(format!(
                        "function '{name}' calls unknown function '{called}'"
                    ));
                }
            }
        }
        visiting.pop();
        visited.insert(name);
    }

    impl RawStep {
        fn kind(&self) -> Result<StepKind<'_>, &'static str> {
            let count = usize::from(self.command.is_some())
                + usize::from(self.call.is_some())
                + usize::from(self.approved.is_some())
                + usize::from(self.r#loop.is_some());
            if count != 1 {
                return Err("must contain exactly one of 'command', 'call', 'approved', or 'loop'");
            }
            if let Some(command) = &self.command {
                Ok(StepKind::Command(command))
            } else if let Some(call) = &self.call {
                Ok(StepKind::Call(call))
            } else if let Some(approved) = &self.approved {
                Ok(StepKind::Approved(approved))
            } else {
                Ok(StepKind::Loop(
                    self.r#loop.as_ref().expect("action count checked"),
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

    fn expand_steps(
        raw_steps: &[RawStep],
        functions: &BTreeMap<String, Vec<RawStep>>,
        variables: &HashMap<String, String>,
        workspace: &Path,
        stack: &mut Vec<String>,
        approved: &mut HashSet<PathBuf>,
        output: &mut Vec<WorkflowStep>,
    ) -> Result<(), String> {
        for step in raw_steps {
            match step.kind().map_err(str::to_string)? {
                StepKind::Command(command) => {
                    validate_external_variable_paths(command, variables, workspace, approved)?;
                    let command = interpolate(command, variables)?;
                    output.push(WorkflowStep::Command(command));
                }
                StepKind::Approved(paths) => {
                    for path in paths.values() {
                        let path = interpolate(path, variables)?;
                        let path = existing_path(&path, workspace)
                            .map_err(|error| format!("approved path {error}"))?;
                        approved.insert(path.clone());
                        output.push(WorkflowStep::Approved(path));
                    }
                }
                StepKind::Loop(spec) => output.push(WorkflowStep::Loop(compile_loop(
                    spec, variables, workspace, approved,
                )?)),
                StepKind::Call(name) => {
                    if stack.iter().any(|called| called == name) {
                        let mut cycle = stack.clone();
                        cycle.push(name.to_string());
                        return Err(format!("recursive function call: {}", cycle.join(" -> ")));
                    }
                    let function = functions
                        .get(name)
                        .ok_or_else(|| format!("calls unknown function '{name}'"))?;
                    stack.push(name.to_string());
                    expand_steps(
                        function, functions, variables, workspace, stack, approved, output,
                    )?;
                    stack.pop();
                }
            }
        }
        Ok(())
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
    ) -> Result<(), String> {
        for name in variable_references(command)? {
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
        workspace: &Path,
        approved: &HashSet<PathBuf>,
    ) -> Result<WorkflowLoop, String> {
        let objective = interpolate(&raw.objective, variables)?;
        if objective.trim().is_empty() {
            return Err("has an empty loop objective".to_string());
        }
        let stop = match &raw.stop {
            RawLoopStop::Turns { count } if *count > 0 => WorkflowLoopStop::Turns(*count),
            RawLoopStop::Turns { .. } => {
                return Err("loop turn count must be greater than zero".to_string());
            }
            RawLoopStop::Time { duration } => {
                let duration = interpolate(duration, variables)?;
                WorkflowLoopStop::Time(parse_duration(&duration)?)
            }
            RawLoopStop::Goal { condition } => {
                let condition = interpolate(condition, variables)?;
                if condition.trim().is_empty() {
                    return Err("has an empty loop goal condition".to_string());
                }
                WorkflowLoopStop::Goal(condition)
            }
        };

        let mut checks = Vec::with_capacity(raw.review.checks.len());
        for check in &raw.review.checks {
            validate_external_variable_paths(check, variables, workspace, approved)?;
            checks.push(interpolate(check, variables)?);
        }
        let rubric = raw
            .review
            .rubric
            .iter()
            .map(|criterion| interpolate(criterion, variables))
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
            assert_eq!(
                plan.jobs[0].steps,
                vec![
                    WorkflowStep::Approved(upstream.canonicalize().expect("canonical upstream")),
                    WorkflowStep::Command("/export pr".to_string()),
                    WorkflowStep::Command(format!(
                        "/shell mv pgagroal-pr.pdf {}/",
                        upstream.display()
                    )),
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
