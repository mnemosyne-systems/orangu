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

use serde::Deserialize;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillSource {
    Project,
    User,
}

impl std::fmt::Display for SkillSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkillSource::Project => write!(f, "project"),
            SkillSource::User => write!(f, "user"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
    pub source: SkillSource,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: String,
}

impl Skill {
    pub fn render_body(&self, arguments: &str) -> String {
        self.body.replace("$ARGUMENTS", arguments)
    }

    pub fn render_activation(&self, arguments: &str) -> String {
        let body = self.render_body(arguments);
        let skill_dir = self
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .display()
            .to_string();
        let resources = self.list_resources();
        let mut rendered = format!(
            "<skill_content name=\"{}\">\n{}\n\nSkill directory: {}\nRelative paths in this skill are relative to the skill directory.",
            self.name, body, skill_dir
        );
        if !resources.is_empty() {
            rendered.push_str("\n<skill_resources>\n");
            for resource in resources {
                rendered.push_str(&format!("<file>{resource}</file>\n"));
            }
            rendered.push_str("</skill_resources>");
        }
        rendered.push_str("\n</skill_content>");
        rendered
    }

    fn list_resources(&self) -> Vec<String> {
        let Some(skill_dir) = self.path.parent() else {
            return Vec::new();
        };
        let mut resources = Vec::new();
        for entry in WalkDir::new(skill_dir)
            .min_depth(1)
            .max_depth(4)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry.path() == self.path {
                continue;
            }
            if let Ok(relative) = entry.path().strip_prefix(skill_dir) {
                resources.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
        resources.sort();
        resources
    }
}

pub struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    /// Discover skills from workspace and user directories.
    /// Project skills override user skills with the same name.
    pub fn discover(workspace: &Path) -> Self {
        let mut skills = Vec::new();

        // Discover user skills from the client-native and cross-client paths.
        if let Some(home) = home::home_dir() {
            for dir in [
                home.join(".agents").join("skills"),
                home.join(".orangu").join("skills"),
            ] {
                if dir.exists() {
                    Self::scan_dir(&dir, SkillSource::User, &mut skills);
                }
            }
        }

        // Discover project skills from the client-native and cross-client paths.
        for dir in [
            workspace.join(".agents").join("skills"),
            workspace.join(".orangu").join("skills"),
        ] {
            if dir.exists() {
                Self::scan_dir(&dir, SkillSource::Project, &mut skills);
            }
        }

        // Project skills override user skills with the same name.
        let mut deduplicated = Vec::new();
        let mut seen = HashSet::new();

        for skill in skills.into_iter().rev() {
            if seen.insert(skill.name.clone()) {
                deduplicated.push(skill);
            }
        }

        deduplicated.reverse();

        // Sort by name for deterministic order in `/skills` and system prompt
        deduplicated.sort_by(|a, b| a.name.cmp(&b.name));

        Self {
            skills: deduplicated,
        }
    }

    fn scan_dir(dir: &Path, source: SkillSource, skills: &mut Vec<Skill>) {
        for entry in WalkDir::new(dir)
            .min_depth(1)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file()
                && entry.file_name() == "SKILL.md"
                && let Ok(content) = fs::read_to_string(entry.path())
            {
                let default_name = entry
                    .path()
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                if let Some(skill) = Self::parse_skill(
                    &content,
                    default_name,
                    entry.path().to_path_buf(),
                    source.clone(),
                ) {
                    skills.push(skill);
                }
            }
        }
    }

    fn parse_skill(
        content: &str,
        default_name: String,
        path: PathBuf,
        source: SkillSource,
    ) -> Option<Skill> {
        let (frontmatter_str, body_str) = if content.starts_with("---") {
            // Find the second "---"
            let mut parts = content.splitn(3, "---");
            parts.next(); // Skip the empty part before the first "---"
            let frontmatter = parts.next().unwrap_or("").trim();
            let body = parts.next().unwrap_or("").trim();
            (frontmatter, body)
        } else {
            ("", content.trim())
        };

        if frontmatter_str.is_empty() {
            return None; // No valid frontmatter found
        }

        let frontmatter: Result<SkillFrontmatter, _> = serde_yaml::from_str(frontmatter_str);

        if let Ok(fm) = frontmatter {
            let name = fm.name.unwrap_or(default_name);
            Some(Skill {
                name,
                description: fm.description,
                body: body_str.to_string(),
                path,
                source,
            })
        } else {
            None
        }
    }

    pub fn find(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    pub fn all(&self) -> &[Skill] {
        &self.skills
    }

    pub fn system_prompt_index(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }

        let mut out = String::from(
            "<skills>\nThe following skills provide specialized instructions for specific tasks.\nWhen a task matches a skill's description, load the SKILL.md at the listed location with the show_file tool before proceeding.\nWhen a skill references relative paths, resolve them against the skill directory, which is the parent directory of SKILL.md.\nUsers can also invoke a skill directly with /skill-name.\n\n<available_skills>\n",
        );
        for skill in &self.skills {
            out.push_str(&format!(
                "<skill>\n<name>{}</name>\n<description>{}</description>\n<location>{}</location>\n</skill>\n",
                skill.name,
                skill.description.trim(),
                skill.path.display()
            ));
        }
        out.push_str("</available_skills>\n</skills>");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_valid_skill() {
        let content = "---
name: code-review
description: |
  Reviews code for common issues.
---
# Code Review
Check for these things: $ARGUMENTS
";
        let skill = SkillRegistry::parse_skill(
            content,
            "default".to_string(),
            PathBuf::from("SKILL.md"),
            SkillSource::Project,
        )
        .unwrap();

        assert_eq!(skill.name, "code-review");
        assert_eq!(skill.description.trim(), "Reviews code for common issues.");
        assert_eq!(
            skill.body.trim(),
            "# Code Review\nCheck for these things: $ARGUMENTS"
        );
        assert_eq!(skill.source, SkillSource::Project);

        let rendered = skill.render_body("security vulns");
        assert_eq!(
            rendered.trim(),
            "# Code Review\nCheck for these things: security vulns"
        );
    }

    #[test]
    fn parse_skill_without_name() {
        let content = "---
description: A cool skill
---
Body text
";
        let skill = SkillRegistry::parse_skill(
            content,
            "dir-name".to_string(),
            PathBuf::from("SKILL.md"),
            SkillSource::Project,
        )
        .unwrap();

        assert_eq!(skill.name, "dir-name");
        assert_eq!(skill.description.trim(), "A cool skill");
    }

    #[test]
    fn parse_skill_invalid_frontmatter() {
        let content = "---
invalid yaml
---
Body text
";
        let skill = SkillRegistry::parse_skill(
            content,
            "dir-name".to_string(),
            PathBuf::from("SKILL.md"),
            SkillSource::Project,
        );
        assert!(skill.is_none());
    }

    #[test]
    fn parse_skill_no_frontmatter() {
        let content = "Just body text";
        let skill = SkillRegistry::parse_skill(
            content,
            "dir-name".to_string(),
            PathBuf::from("SKILL.md"),
            SkillSource::Project,
        );
        assert!(skill.is_none());
    }

    #[test]
    fn discover_scans_cross_client_locations() {
        let workspace = tempdir().expect("workspace");
        let project_skill_dir = workspace.path().join(".agents/skills/project-review");
        std::fs::create_dir_all(&project_skill_dir).expect("project skill dir");
        std::fs::write(
            project_skill_dir.join("SKILL.md"),
            "---\ndescription: Project skill\n---\nProject body\n",
        )
        .expect("project skill");

        let registry = SkillRegistry::discover(workspace.path());
        let skill = registry.find("project-review").expect("skill discovered");
        assert_eq!(skill.source, SkillSource::Project);
    }

    #[test]
    fn activation_wraps_skill_content_and_resources() {
        let workspace = tempdir().expect("workspace");
        let skill_dir = workspace.path().join("code-review");
        std::fs::create_dir_all(skill_dir.join("references")).expect("refs dir");
        let skill_path = skill_dir.join("SKILL.md");
        std::fs::write(
            &skill_path,
            "---\ndescription: Review code\n---\nCheck: $ARGUMENTS\n",
        )
        .expect("skill");
        std::fs::write(skill_dir.join("references/checklist.md"), "checklist").expect("resource");

        let skill = SkillRegistry::parse_skill(
            &std::fs::read_to_string(&skill_path).expect("read skill"),
            "code-review".to_string(),
            skill_path,
            SkillSource::Project,
        )
        .expect("parsed");

        let rendered = skill.render_activation("security");
        assert!(rendered.contains("<skill_content name=\"code-review\">"));
        assert!(rendered.contains("Check: security"));
        assert!(rendered.contains("<skill_resources>"));
        assert!(rendered.contains("<file>references/checklist.md</file>"));
    }
}
