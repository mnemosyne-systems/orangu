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

use anyhow::{Context, Result, anyhow};
use ratatui::style::{Color, Modifier, Style};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::{OnceLock, RwLock},
};

/// Legacy built-in theme identifiers kept for config compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThemeKind {
    Classic,
    OranguDay,
    TokyoNight,
    RosePineMoon,
    /// Meta-variant: follow system dark/light appearance.
    Auto,
}

impl ThemeKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::OranguDay => "oranguday",
            Self::TokyoNight => "tokyonight",
            Self::RosePineMoon => "rosepine-moon",
            Self::Auto => "auto",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match normalize_theme_name(name).as_str() {
            "auto" | "system" => Some(Self::Auto),
            "classic" | "orangunight" | "orangu-night" | "dark" => Some(Self::Classic),
            "oranguday" | "orangu-day" | "light" | "day" => Some(Self::OranguDay),
            "tokyonight" | "tokyo-night" | "tokyo" => Some(Self::TokyoNight),
            "rosepine" | "rose-pine" | "rosepine-moon" | "rose-pine-moon" => {
                Some(Self::RosePineMoon)
            }
            _ => None,
        }
    }
}

/// Centralized semantic styling for the Ratatui components.
#[derive(Clone, Debug)]
pub struct Theme {
    pub success: Style,
    pub error: Style,
    pub ignore: Style,
    pub deep: Style,
    pub muted: Style,
    pub cursor_line_bg: Style,
    pub selected_file: Style,
    pub comment_bg: Style,
    pub code_block_bg: Style,
    pub highlight: Style,
    pub warning: Style,
    pub user_input: Style,
    pub bg_base: Color,
    pub text_primary: Color,
}

impl Default for Theme {
    fn default() -> Self {
        classic_fallback()
    }
}

#[derive(Clone)]
enum ActiveTheme {
    Named { name: String, theme: Box<Theme> },
    Auto,
}

#[derive(Clone)]
struct ThemeState {
    active: ActiveTheme,
    /// Temporary overlay for live dropdown previews; does not persist or save.
    preview: Option<(String, Theme)>,
    auto_dark_name: String,
    auto_dark_theme: Theme,
    auto_light_name: String,
    auto_light_theme: Theme,
}

impl Default for ThemeState {
    fn default() -> Self {
        Self {
            active: ActiveTheme::Auto,
            preview: None,
            auto_dark_name: "classic".to_string(),
            auto_dark_theme: load_theme_by_name("classic")
                .map(|(_, theme)| theme)
                .unwrap_or_else(|_| classic_fallback()),
            auto_light_name: "oranguday".to_string(),
            auto_light_theme: load_theme_by_name("oranguday")
                .map(|(_, theme)| theme)
                .unwrap_or_else(|_| oranguday_fallback()),
        }
    }
}

fn resolved_active_theme(state: &ThemeState) -> Theme {
    match &state.active {
        ActiveTheme::Named { theme, .. } => (**theme).clone(),
        ActiveTheme::Auto => {
            let dark =
                dark_light::detect().unwrap_or(dark_light::Mode::Dark) != dark_light::Mode::Light;
            if dark {
                state.auto_dark_theme.clone()
            } else {
                state.auto_light_theme.clone()
            }
        }
    }
}

#[derive(Clone, Copy)]
struct BuiltInTheme {
    name: &'static str,
    source: &'static str,
    aliases: &'static [&'static str],
}

const BUILT_IN_THEMES: &[BuiltInTheme] = &[
    BuiltInTheme {
        name: "classic",
        source: include_str!("../../contrib/themes/classic.theme"),
        aliases: &["orangunight", "orangu-night", "dark"],
    },
    BuiltInTheme {
        name: "oranguday",
        source: include_str!("../../contrib/themes/oranguday.theme"),
        aliases: &["orangu-day", "light", "day"],
    },
    BuiltInTheme {
        name: "tokyonight",
        source: include_str!("../../contrib/themes/tokyonight.theme"),
        aliases: &["tokyo-night", "tokyo"],
    },
    BuiltInTheme {
        name: "rosepine-moon",
        source: include_str!("../../contrib/themes/rosepine-moon.theme"),
        aliases: &["rosepine", "rose-pine", "rose-pine-moon"],
    },
];

fn theme_state() -> &'static RwLock<ThemeState> {
    static STATE: OnceLock<RwLock<ThemeState>> = OnceLock::new();
    STATE.get_or_init(|| RwLock::new(ThemeState::default()))
}

fn normalize_theme_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn user_theme_dir() -> Option<PathBuf> {
    Some(home::home_dir()?.join(".orangu/themes"))
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home::home_dir()
    {
        return home.join(rest);
    }
    if path == "~"
        && let Some(home) = home::home_dir()
    {
        return home;
    }
    PathBuf::from(path)
}

fn is_path_like(spec: &str) -> bool {
    spec.contains(std::path::MAIN_SEPARATOR)
        || spec.starts_with('.')
        || spec.starts_with('~')
        || spec.ends_with(".theme")
}

fn built_in_theme(name: &str) -> Option<&'static BuiltInTheme> {
    let normalized = normalize_theme_name(name);
    BUILT_IN_THEMES.iter().find(|theme| {
        theme.name == normalized || theme.aliases.iter().any(|alias| *alias == normalized)
    })
}

fn shipped_theme_names() -> Vec<String> {
    BUILT_IN_THEMES
        .iter()
        .map(|theme| theme.name.to_string())
        .collect()
}

fn parse_color(value: &str) -> Result<Color> {
    let hex = value
        .trim()
        .strip_prefix('#')
        .ok_or_else(|| anyhow!("expected #RRGGBB, found '{value}'"))?;
    if hex.len() != 6 {
        return Err(anyhow!("expected 6 hex digits, found '{value}'"));
    }
    let red = u8::from_str_radix(&hex[0..2], 16)
        .with_context(|| format!("invalid red component in '{value}'"))?;
    let green = u8::from_str_radix(&hex[2..4], 16)
        .with_context(|| format!("invalid green component in '{value}'"))?;
    let blue = u8::from_str_radix(&hex[4..6], 16)
        .with_context(|| format!("invalid blue component in '{value}'"))?;
    Ok(Color::Rgb(red, green, blue))
}

fn parse_style(value: &str) -> Result<Style> {
    let mut style = Style::default();
    for token in value.split_whitespace() {
        if let Some(color) = token.strip_prefix("fg:") {
            style = style.fg(parse_color(color)?);
        } else if let Some(color) = token.strip_prefix("bg:") {
            style = style.bg(parse_color(color)?);
        } else if token == "bold" {
            style = style.add_modifier(Modifier::BOLD);
        } else if token == "italic" {
            style = style.add_modifier(Modifier::ITALIC);
        } else if token == "underlined" {
            style = style.add_modifier(Modifier::UNDERLINED);
        } else {
            return Err(anyhow!("unknown style token '{token}'"));
        }
    }
    Ok(style)
}

fn parse_theme(source: &str, origin: &str) -> Result<Theme> {
    let mut fields = std::collections::HashMap::new();
    for (index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            return Err(anyhow!(
                "{origin}:{}: expected `key = value`",
                index.saturating_add(1)
            ));
        };
        fields.insert(key.trim().to_string(), value.trim().to_string());
    }

    let field = |name: &str| -> Result<&str> {
        fields
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("{origin}: missing `{name}`"))
    };

    Ok(Theme {
        success: parse_style(field("success")?)?,
        error: parse_style(field("error")?)?,
        ignore: parse_style(field("ignore")?)?,
        deep: parse_style(field("deep")?)?,
        muted: parse_style(field("muted")?)?,
        cursor_line_bg: parse_style(field("cursor_line_bg")?)?,
        selected_file: parse_style(field("selected_file")?)?,
        comment_bg: parse_style(field("comment_bg")?)?,
        code_block_bg: parse_style(field("code_block_bg")?)?,
        highlight: parse_style(field("highlight")?)?,
        warning: parse_style(field("warning")?)?,
        user_input: parse_style(field("user_input")?)?,
        bg_base: parse_color(field("bg_base")?)?,
        text_primary: parse_color(field("text_primary")?)?,
    })
}

fn load_theme_path(path: &Path) -> Result<(String, Theme)> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read theme file {}", path.display()))?;
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("custom")
        .to_string();
    Ok((
        name,
        parse_theme(&contents, &path.display().to_string())
            .with_context(|| format!("failed to parse theme {}", path.display()))?,
    ))
}

fn named_user_theme_path(name: &str) -> Option<PathBuf> {
    let dir = user_theme_dir()?;
    let normalized = name.trim();
    let path = dir.join(normalized);
    if path.is_file() {
        return Some(path);
    }
    let with_extension = dir.join(format!("{normalized}.theme"));
    with_extension.is_file().then_some(with_extension)
}

fn load_theme_by_name(name: &str) -> Result<(String, Theme)> {
    let normalized = normalize_theme_name(name);
    if normalized == "auto" || normalized == "system" {
        return Err(anyhow!("'auto' is a selector, not a concrete theme file"));
    }
    if let Some(theme) = built_in_theme(&normalized) {
        return Ok((
            theme.name.to_string(),
            parse_theme(theme.source, theme.name)
                .with_context(|| format!("failed to parse built-in theme {}", theme.name))?,
        ));
    }
    if let Some(path) = named_user_theme_path(name) {
        return load_theme_path(&path);
    }
    Err(anyhow!(
        "unknown theme '{name}'. Available: {}",
        Theme::available_theme_names().join(", ")
    ))
}

fn apply_terminal_palette(theme: &Theme) {
    if let Color::Rgb(r, g, b) = theme.bg_base {
        print!("\x1b]11;#{:02x}{:02x}{:02x}\x07", r, g, b);
    }
    if let Color::Rgb(r, g, b) = theme.text_primary {
        print!("\x1b]10;#{:02x}{:02x}{:02x}\x07", r, g, b);
        print!("\x1b]12;#{:02x}{:02x}{:02x}\x07", r, g, b);
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

fn color_luma(color: Color) -> u16 {
    match color {
        Color::Rgb(red, green, blue) => u16::from(red) * 3 + u16::from(green) * 6 + u16::from(blue),
        Color::Black => 0,
        Color::White => 255 * 10,
        _ => 0,
    }
}

fn classic_fallback() -> Theme {
    Theme {
        success: Style::default().fg(Color::Rgb(80, 200, 120)),
        error: Style::default().fg(Color::Rgb(220, 80, 80)),
        ignore: Style::default().fg(Color::Rgb(100, 160, 230)),
        deep: Style::default().fg(Color::Rgb(170, 120, 220)),
        muted: Style::default().fg(Color::Rgb(88, 88, 88)),
        cursor_line_bg: Style::default()
            .bg(Color::Rgb(40, 40, 45))
            .fg(Color::Rgb(255, 245, 230)),
        selected_file: Style::default().bg(Color::Rgb(59, 66, 97)).fg(Color::White),
        comment_bg: Style::default().bg(Color::Rgb(31, 35, 53)),
        code_block_bg: Style::default().bg(Color::Rgb(25, 25, 35)),
        highlight: Style::default().fg(Color::Rgb(122, 162, 247)),
        warning: Style::default().fg(Color::Rgb(230, 200, 120)),
        user_input: Style::default().fg(Color::White).bg(Color::Rgb(45, 35, 20)),
        bg_base: Color::Rgb(24, 24, 24),
        text_primary: Color::Rgb(240, 240, 240),
    }
}

fn oranguday_fallback() -> Theme {
    Theme {
        success: Style::default().fg(Color::Rgb(30, 140, 60)),
        error: Style::default().fg(Color::Rgb(200, 40, 40)),
        ignore: Style::default().fg(Color::Rgb(40, 100, 180)),
        deep: Style::default().fg(Color::Rgb(120, 70, 180)),
        muted: Style::default().fg(Color::Rgb(120, 120, 120)),
        cursor_line_bg: Style::default()
            .bg(Color::Rgb(230, 230, 235))
            .fg(Color::Black),
        selected_file: Style::default()
            .bg(Color::Rgb(190, 190, 190))
            .fg(Color::Black),
        comment_bg: Style::default().bg(Color::Rgb(200, 210, 200)),
        code_block_bg: Style::default().bg(Color::Rgb(240, 240, 240)),
        highlight: Style::default().fg(Color::Rgb(20, 100, 180)),
        warning: Style::default().fg(Color::Rgb(200, 130, 20)),
        user_input: Style::default()
            .fg(Color::Black)
            .bg(Color::Rgb(240, 230, 210)),
        bg_base: Color::Rgb(250, 250, 250),
        text_primary: Color::Rgb(30, 30, 30),
    }
}

impl Theme {
    pub fn current() -> Self {
        let state = theme_state().read().expect("theme state lock poisoned");
        if let Some((_, theme)) = &state.preview {
            return theme.clone();
        }
        resolved_active_theme(&state)
    }

    pub fn is_dark() -> bool {
        color_luma(Self::current().bg_base) < 1280
    }

    pub fn apply_kind(kind: ThemeKind) {
        let _ = Self::apply_named(kind.display_name());
    }

    /// Temporarily show `name` for UI overview (e.g. `/theme` dropdown).
    /// Does not change the committed theme or session persistence.
    /// `default` / `global` clear any preview so the committed theme shows.
    pub fn preview_named(name: &str) -> Result<()> {
        let normalized = normalize_theme_name(name);
        if matches!(normalized.as_str(), "default" | "global" | "") {
            Self::clear_preview();
            return Ok(());
        }

        {
            let state = theme_state().read().expect("theme state lock poisoned");
            if let Some((preview_name, _)) = &state.preview
                && preview_name == &normalized
            {
                return Ok(());
            }
        }

        if matches!(normalized.as_str(), "auto" | "system") {
            let theme = {
                let mut state = theme_state().write().expect("theme state lock poisoned");
                let dark = dark_light::detect().unwrap_or(dark_light::Mode::Dark)
                    != dark_light::Mode::Light;
                let theme = if dark {
                    state.auto_dark_theme.clone()
                } else {
                    state.auto_light_theme.clone()
                };
                state.preview = Some(("auto".to_string(), theme.clone()));
                theme
            };
            apply_terminal_palette(&theme);
            return Ok(());
        }

        let (canonical_name, theme) = load_theme_by_name(name)?;
        {
            let mut state = theme_state().write().expect("theme state lock poisoned");
            state.preview = Some((canonical_name, theme.clone()));
        }
        apply_terminal_palette(&theme);
        Ok(())
    }

    /// Drop a live preview and restore the committed theme palette.
    pub fn clear_preview() {
        let theme = {
            let mut state = theme_state().write().expect("theme state lock poisoned");
            if state.preview.is_none() {
                return;
            }
            state.preview = None;
            resolved_active_theme(&state)
        };
        apply_terminal_palette(&theme);
    }

    pub fn is_previewing() -> bool {
        theme_state()
            .read()
            .expect("theme state lock poisoned")
            .preview
            .is_some()
    }

    pub fn apply_named(name: &str) -> Result<String> {
        if matches!(normalize_theme_name(name).as_str(), "auto" | "system") {
            let theme = {
                let mut state = theme_state().write().expect("theme state lock poisoned");
                state.preview = None;
                state.active = ActiveTheme::Auto;
                let dark = dark_light::detect().unwrap_or(dark_light::Mode::Dark)
                    != dark_light::Mode::Light;
                if dark {
                    state.auto_dark_theme.clone()
                } else {
                    state.auto_light_theme.clone()
                }
            };
            apply_terminal_palette(&theme);
            return Ok("auto".to_string());
        }

        let (canonical_name, theme) = load_theme_by_name(name)?;
        {
            let mut state = theme_state().write().expect("theme state lock poisoned");
            state.preview = None;
            state.active = ActiveTheme::Named {
                name: canonical_name.clone(),
                theme: Box::new(theme.clone()),
            };
        }
        apply_terminal_palette(&theme);
        Ok(canonical_name)
    }

    pub fn apply_cli_override(spec: &str) -> Result<String> {
        if is_path_like(spec) {
            let path = expand_tilde(spec);
            let (name, theme) = load_theme_path(&path)?;
            {
                let mut state = theme_state().write().expect("theme state lock poisoned");
                state.preview = None;
                state.active = ActiveTheme::Named {
                    name: name.clone(),
                    theme: Box::new(theme.clone()),
                };
            }
            apply_terminal_palette(&theme);
            return Ok(name);
        }
        Self::apply_named(spec)
    }

    pub fn set_auto_themes(dark: ThemeKind, light: ThemeKind) {
        Self::set_auto_theme_names(dark.display_name(), light.display_name());
    }

    pub fn set_auto_theme_names(dark: &str, light: &str) {
        let (dark_name, dark_theme) = load_theme_by_name(dark)
            .unwrap_or_else(|_| ("classic".to_string(), classic_fallback()));
        let (light_name, light_theme) = load_theme_by_name(light)
            .unwrap_or_else(|_| ("oranguday".to_string(), oranguday_fallback()));

        let active_theme = {
            let mut state = theme_state().write().expect("theme state lock poisoned");
            state.auto_dark_name = dark_name;
            state.auto_dark_theme = dark_theme;
            state.auto_light_name = light_name;
            state.auto_light_theme = light_theme;
            match &state.active {
                ActiveTheme::Auto => {
                    let dark = dark_light::detect().unwrap_or(dark_light::Mode::Dark)
                        != dark_light::Mode::Light;
                    if dark {
                        Some(state.auto_dark_theme.clone())
                    } else {
                        Some(state.auto_light_theme.clone())
                    }
                }
                _ => None,
            }
        };

        if let Some(theme) = active_theme {
            apply_terminal_palette(&theme);
        }
    }

    pub fn available_theme_names() -> Vec<String> {
        let mut names = BTreeSet::new();
        names.insert("auto".to_string());
        names.extend(shipped_theme_names());
        if let Some(dir) = user_theme_dir()
            && let Ok(entries) = std::fs::read_dir(dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("theme")
                    && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
                    && !stem.is_empty()
                {
                    names.insert(stem.to_string());
                }
            }
        }
        names.into_iter().collect()
    }

    pub fn available_theme_summary() -> String {
        Self::available_theme_names().join(", ")
    }

    pub fn available_session_theme_names() -> Vec<String> {
        let mut names = vec!["default".to_string()];
        names.extend(Self::available_theme_names());
        names
    }

    pub fn available_session_theme_summary() -> String {
        Self::available_session_theme_names().join(", ")
    }

    pub fn current_theme_name() -> String {
        let state = theme_state().read().expect("theme state lock poisoned");
        match &state.active {
            ActiveTheme::Named { name, .. } => name.clone(),
            ActiveTheme::Auto => "auto".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_alias_resolves_to_classic() {
        let (name, theme) = load_theme_by_name("orangunight").expect("classic alias");
        assert_eq!(name, "classic");
        assert!(matches!(theme.bg_base, Color::Rgb(24, 24, 24)));
    }

    #[test]
    fn shipped_theme_is_listed() {
        let names = Theme::available_theme_names();
        assert!(names.contains(&"classic".to_string()));
        assert!(names.contains(&"auto".to_string()));
    }

    #[test]
    fn theme_file_parser_reads_styles() {
        let theme = parse_theme(
            "success = fg:#010203 bold\nerror = fg:#040506\nignore = fg:#070809\ndeep = fg:#0a0b0c\nmuted = fg:#0d0e0f\ncursor_line_bg = fg:#111213 bg:#141516\nselected_file = fg:#171819 bg:#1a1b1c\ncomment_bg = bg:#1d1e1f\ncode_block_bg = bg:#202122\nhighlight = fg:#232425\nwarning = fg:#262728\nuser_input = fg:#292a2b bg:#2c2d2e\nbg_base = #2f3031\ntext_primary = #323334\n",
            "inline",
        )
        .expect("parse theme");
        assert!(theme.success.add_modifier.contains(Modifier::BOLD));
        assert!(matches!(theme.bg_base, Color::Rgb(47, 48, 49)));
        assert!(matches!(theme.text_primary, Color::Rgb(50, 51, 52)));
    }

    #[test]
    fn preview_named_does_not_commit_theme() {
        Theme::apply_named("classic").expect("commit classic");
        assert_eq!(Theme::current_theme_name(), "classic");
        assert!(!Theme::is_previewing());

        Theme::preview_named("oranguday").expect("preview oranguday");
        assert!(Theme::is_previewing());
        // Committed name stays classic; only the render palette switches.
        assert_eq!(Theme::current_theme_name(), "classic");
        assert!(matches!(Theme::current().bg_base, Color::Rgb(r, ..) if r > 100));

        Theme::clear_preview();
        assert!(!Theme::is_previewing());
        assert_eq!(Theme::current_theme_name(), "classic");
        assert!(matches!(Theme::current().bg_base, Color::Rgb(24, 24, 24)));
    }

    #[test]
    fn apply_named_clears_preview() {
        Theme::apply_named("classic").expect("commit classic");
        Theme::preview_named("tokyonight").expect("preview");
        assert!(Theme::is_previewing());

        Theme::apply_named("oranguday").expect("commit oranguday");
        assert!(!Theme::is_previewing());
        assert_eq!(Theme::current_theme_name(), "oranguday");
    }
}
