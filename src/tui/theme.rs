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
    ModernDark,
    ModernLight,
    OranguDay,
    TokyoNight,
    RosePineMoon,
    /// Meta-variant: one of the available themes, drawn at random.
    Random,
}

impl ThemeKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::ModernDark => "modern_dark",
            Self::ModernLight => "modern_light",
            Self::OranguDay => "oranguday",
            Self::TokyoNight => "tokyonight",
            Self::RosePineMoon => "rosepine-moon",
            Self::Random => RANDOM_SELECTOR,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match normalize_theme_name(name).as_str() {
            "random" => Some(Self::Random),
            "classic" | "orangunight" | "orangu-night" | "dark" => Some(Self::Classic),
            "modern" | "modern_dark" | "modern-dark" => Some(Self::ModernDark),
            "modern_light" | "modern-light" => Some(Self::ModernLight),
            "oranguday" | "orangu-day" | "light" | "day" => Some(Self::OranguDay),
            "tokyonight" | "tokyo-night" | "tokyo" => Some(Self::TokyoNight),
            "rosepine" | "rose-pine" | "rosepine-moon" | "rose-pine-moon" => {
                Some(Self::RosePineMoon)
            }
            _ => None,
        }
    }
}

/// The screen furniture a theme asks for, independent of its palette.
///
/// `Classic` is the frame orangu has always drawn: the boxed banner pinned to
/// the top of every screen, the output window directly below it, full-width
/// `━` separators above and below the input window, and the status line under
/// them. `Modern` is the Ratatui-native frame: the banner only on the empty
/// landing screen, an inset output window, and a rounded input box. Both frames
/// draw the same status line and leave the input window bare.
///
/// A theme file selects one with `chrome = classic` / `chrome = modern`;
/// omitting the key keeps the classic frame, so a palette-only theme (shipped
/// or user-written) never changes the layout by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Chrome {
    #[default]
    Classic,
    Modern,
}

impl Chrome {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "classic" => Ok(Self::Classic),
            "modern" => Ok(Self::Modern),
            other => Err(anyhow!(
                "unknown chrome '{other}', expected 'classic' or 'modern'"
            )),
        }
    }

    pub fn is_classic(self) -> bool {
        self == Self::Classic
    }
}

/// Centralized semantic styling for the Ratatui components.
#[derive(Clone, Debug, PartialEq)]
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
    pub chrome: Chrome,
}

impl Default for Theme {
    fn default() -> Self {
        classic_fallback()
    }
}

/// The `theme` value that is not a palette of its own but a draw from the
/// ones that are. Mirrored as [`Theme::RANDOM_SELECTOR`] for callers outside
/// this module.
const RANDOM_SELECTOR: &str = "random";

#[derive(Clone)]
enum ActiveTheme {
    Named {
        name: String,
        theme: Box<Theme>,
    },
    /// `theme = random`: one of the available themes, drawn once when the
    /// selector is applied and then held for the rest of the process, so the
    /// UI doesn't reshuffle under the user mid-session. `picked` is the theme
    /// that came up; the committed *name* stays `random`, so the config or
    /// session value round-trips and the next launch draws again.
    Random {
        picked: String,
        theme: Box<Theme>,
    },
}

#[derive(Clone)]
struct ThemeState {
    active: ActiveTheme,
    /// Temporary overlay for live dropdown previews; does not persist or save.
    preview: Option<(String, Theme)>,
}

impl Default for ThemeState {
    fn default() -> Self {
        Self {
            active: ActiveTheme::Named {
                name: "classic".to_string(),
                theme: Box::new(
                    load_theme_by_name("classic")
                        .map(|(_, theme)| theme)
                        .unwrap_or_else(|_| classic_fallback()),
                ),
            },
            preview: None,
        }
    }
}

fn resolved_active_theme(state: &ThemeState) -> Theme {
    match &state.active {
        ActiveTheme::Named { theme, .. } | ActiveTheme::Random { theme, .. } => (**theme).clone(),
    }
}

/// Draw one of the available themes — the shipped ones plus any user file in
/// `~/.orangu/themes` — skipping the selector itself. A theme file that fails
/// to parse is passed over rather than aborting the draw; if none of them load,
/// the built-in classic palette stands in.
fn random_theme() -> (String, Theme) {
    let mut candidates: Vec<String> = Theme::available_theme_names()
        .into_iter()
        .filter(|name| name != RANDOM_SELECTOR)
        .collect();
    while !candidates.is_empty() {
        let index = rand::random_range(0..candidates.len());
        let name = candidates.swap_remove(index);
        if let Ok(resolved) = load_theme_by_name(&name) {
            return resolved;
        }
    }
    ("classic".to_string(), classic_fallback())
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
        name: "modern_dark",
        source: include_str!("../../contrib/themes/modern_dark.theme"),
        aliases: &["modern", "modern-dark"],
    },
    BuiltInTheme {
        name: "modern_light",
        source: include_str!("../../contrib/themes/modern_light.theme"),
        aliases: &["modern-light"],
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

/// Parse a theme colour. `default` means "leave it to the terminal" — the
/// classic frame never repaints the user's background or foreground, exactly
/// as orangu behaved before themes existed.
fn parse_color(value: &str) -> Result<Color> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("default") {
        return Ok(Color::Reset);
    }
    let hex = value
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
        } else if token == "reversed" {
            style = style.add_modifier(Modifier::REVERSED);
        } else {
            return Err(anyhow!("unknown style token '{token}'"));
        }
    }
    Ok(style)
}

/// Parse a theme file over the classic theme as its base.
///
/// Every key is optional: whatever a file leaves out keeps its classic value,
/// so a user theme can override one colour and inherit the rest, and adding a
/// key to `Theme` never invalidates the themes already out there. An
/// unrecognized key is an error rather than a silent no-op — with all keys
/// optional a typo would otherwise resolve quietly to the classic value.
fn parse_theme(source: &str, origin: &str) -> Result<Theme> {
    let mut theme = classic_fallback();

    for (index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let line_number = index.saturating_add(1);
        let Some((key, value)) = trimmed.split_once('=') else {
            return Err(anyhow!("{origin}:{line_number}: expected `key = value`"));
        };
        let (key, value) = (key.trim(), value.trim());
        let invalid = || format!("{origin}:{line_number}: invalid `{key}`");

        match key {
            "success" => theme.success = parse_style(value).with_context(invalid)?,
            "error" => theme.error = parse_style(value).with_context(invalid)?,
            "ignore" => theme.ignore = parse_style(value).with_context(invalid)?,
            "deep" => theme.deep = parse_style(value).with_context(invalid)?,
            "muted" => theme.muted = parse_style(value).with_context(invalid)?,
            "cursor_line_bg" => theme.cursor_line_bg = parse_style(value).with_context(invalid)?,
            "selected_file" => theme.selected_file = parse_style(value).with_context(invalid)?,
            "comment_bg" => theme.comment_bg = parse_style(value).with_context(invalid)?,
            "code_block_bg" => theme.code_block_bg = parse_style(value).with_context(invalid)?,
            "highlight" => theme.highlight = parse_style(value).with_context(invalid)?,
            "warning" => theme.warning = parse_style(value).with_context(invalid)?,
            "user_input" => theme.user_input = parse_style(value).with_context(invalid)?,
            "bg_base" => theme.bg_base = parse_color(value).with_context(invalid)?,
            "text_primary" => theme.text_primary = parse_color(value).with_context(invalid)?,
            "chrome" => theme.chrome = Chrome::parse(value).with_context(invalid)?,
            other => {
                return Err(anyhow!("{origin}:{line_number}: unknown key `{other}`"));
            }
        }
    }

    Ok(theme)
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
    if normalized == RANDOM_SELECTOR {
        return Err(anyhow!(
            "'{RANDOM_SELECTOR}' is a selector, not a concrete theme file"
        ));
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

/// The OSC sequence that puts the terminal's own background, foreground and
/// cursor colours in step with `theme`.
///
/// A `default` colour must emit the *reset* form (OSC 110/111/112) rather than
/// nothing: the previous theme may have set a concrete colour, and staying
/// silent would leave the terminal wearing it. That is what made a light theme
/// bleed into a following `default` one.
fn terminal_palette_sequence(theme: &Theme) -> String {
    let mut sequence = String::new();
    match theme.bg_base {
        Color::Rgb(red, green, blue) => {
            sequence.push_str(&format!("\x1b]11;#{red:02x}{green:02x}{blue:02x}\x07"));
        }
        _ => sequence.push_str("\x1b]111\x07"),
    }
    match theme.text_primary {
        Color::Rgb(red, green, blue) => {
            sequence.push_str(&format!("\x1b]10;#{red:02x}{green:02x}{blue:02x}\x07"));
            sequence.push_str(&format!("\x1b]12;#{red:02x}{green:02x}{blue:02x}\x07"));
        }
        _ => sequence.push_str("\x1b]110\x07\x1b]112\x07"),
    }
    sequence
}

fn apply_terminal_palette(theme: &Theme) {
    use std::io::Write;
    print!("{}", terminal_palette_sequence(theme));
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

/// The palette `contrib/themes/classic.theme` ships, duplicated in code so a
/// broken or unreadable theme file still lands on the original orangu look
/// rather than on Ratatui's defaults.
fn classic_fallback() -> Theme {
    Theme {
        success: Style::default().fg(Color::Rgb(80, 200, 120)),
        error: Style::default().fg(Color::Rgb(220, 80, 80)),
        ignore: Style::default().fg(Color::Rgb(100, 160, 230)),
        deep: Style::default().fg(Color::Rgb(170, 120, 220)),
        muted: Style::default().fg(Color::Rgb(88, 88, 88)),
        cursor_line_bg: Style::default().bg(Color::Rgb(60, 60, 90)),
        selected_file: Style::default().add_modifier(Modifier::REVERSED),
        comment_bg: Style::default().bg(Color::Rgb(38, 48, 38)),
        // `bg:default` in the file: the classic frame prints code blocks against
        // the terminal's own background rather than a panel colour.
        code_block_bg: Style::default().bg(Color::Reset),
        highlight: Style::default().fg(Color::Rgb(102, 178, 255)),
        warning: Style::default().fg(Color::Rgb(230, 200, 120)),
        user_input: Style::default().bg(Color::Rgb(44, 44, 44)),
        bg_base: Color::Reset,
        text_primary: Color::Reset,
        chrome: Chrome::Classic,
    }
}

impl Theme {
    /// The `theme` value that draws one of the available themes instead of
    /// naming a palette directly.
    pub const RANDOM_SELECTOR: &'static str = RANDOM_SELECTOR;

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

    /// The frame the active theme asks for. Read at render time so a `/theme`
    /// switch (or a live dropdown preview) reshapes the screen immediately.
    pub fn chrome() -> Chrome {
        Self::current().chrome
    }

    pub fn apply_kind(kind: ThemeKind) {
        let _ = Self::apply_named(kind.display_name());
    }

    /// Temporarily show `name` for UI overview (e.g. `/theme` dropdown).
    /// Does not change the committed theme or session persistence. An empty
    /// name clears any preview, so backing out of a half-typed theme restores
    /// the committed one.
    pub fn preview_named(name: &str) -> Result<()> {
        let normalized = normalize_theme_name(name);
        if normalized.is_empty() {
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

        if normalized == RANDOM_SELECTOR {
            let (_, theme) = random_theme();
            {
                let mut state = theme_state().write().expect("theme state lock poisoned");
                state.preview = Some((RANDOM_SELECTOR.to_string(), theme.clone()));
            }
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
        if normalize_theme_name(name) == RANDOM_SELECTOR {
            let (picked, theme) = random_theme();
            {
                let mut state = theme_state().write().expect("theme state lock poisoned");
                state.preview = None;
                state.active = ActiveTheme::Random {
                    picked,
                    theme: Box::new(theme.clone()),
                };
            }
            apply_terminal_palette(&theme);
            return Ok(RANDOM_SELECTOR.to_string());
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

    /// The canonical name `spec` resolves to: `modern` becomes `modern_dark`,
    /// a user file's stem stays itself. A spec that names nothing — including
    /// the `random` selector — comes back normalized, so two spellings of the
    /// same theme always compare equal.
    pub fn canonical_theme_name(spec: &str) -> String {
        let normalized = normalize_theme_name(spec);
        if normalized == RANDOM_SELECTOR {
            return normalized;
        }
        load_theme_by_name(&normalized)
            .map(|(name, _)| name)
            .unwrap_or(normalized)
    }

    /// The themes compiled into the binary, in shipped order. Callers that
    /// offer a fixed menu — `--init`, the shell completions — derive it from
    /// here so their list can never drift from `BUILT_IN_THEMES`.
    pub fn built_in_theme_names() -> Vec<String> {
        shipped_theme_names()
    }

    pub fn available_theme_names() -> Vec<String> {
        let mut names = BTreeSet::new();
        names.insert(RANDOM_SELECTOR.to_string());
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

    /// The concrete theme actually in effect. Same as
    /// [`Self::current_theme_name`] for a named theme; under `random` it is the
    /// theme that came up rather than the selector.
    pub fn resolved_theme_name() -> String {
        let state = theme_state().read().expect("theme state lock poisoned");
        match &state.active {
            ActiveTheme::Named { name, .. } => name.clone(),
            ActiveTheme::Random { picked, .. } => picked.clone(),
        }
    }

    pub fn current_theme_name() -> String {
        let state = theme_state().read().expect("theme state lock poisoned");
        match &state.active {
            ActiveTheme::Named { name, .. } => name.clone(),
            // The selector, not the draw: persisting `random` is what makes the
            // next launch draw again.
            ActiveTheme::Random { .. } => RANDOM_SELECTOR.to_string(),
        }
    }
}

/// Serializes tests that pin the process-wide theme. The active theme is
/// global state, and because a theme now also selects the screen chrome, two
/// tests racing on it would reshape each other's layout.
#[cfg(test)]
pub(crate) fn theme_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_alias_resolves_to_classic() {
        let (name, theme) = load_theme_by_name("orangunight").expect("classic alias");
        assert_eq!(name, "classic");
        // The classic frame leaves the terminal's own background alone.
        assert!(matches!(theme.bg_base, Color::Reset));
        assert_eq!(theme.chrome, Chrome::Classic);
    }

    #[test]
    fn modern_themes_select_the_modern_chrome() {
        // The bare `modern` alias resolves to the dark variant.
        let (name, theme) = load_theme_by_name("modern").expect("modern theme");
        assert_eq!(name, "modern_dark");
        assert_eq!(theme.chrome, Chrome::Modern);
        assert!(matches!(theme.bg_base, Color::Rgb(24, 24, 24)));

        let (name, theme) = load_theme_by_name("modern_light").expect("modern light theme");
        assert_eq!(name, "modern_light");
        assert_eq!(theme.chrome, Chrome::Modern);
        assert!(matches!(theme.bg_base, Color::Rgb(250, 250, 250)));
        assert!(
            color_luma(theme.bg_base) >= 1280,
            "modern_light reads as dark"
        );
    }

    #[test]
    fn random_draws_a_concrete_theme_but_commits_the_selector() {
        let _guard = theme_test_guard();

        Theme::apply_named(Theme::RANDOM_SELECTOR).expect("random");
        // The config/session value round-trips as the selector, so the next
        // launch draws again...
        assert_eq!(Theme::current_theme_name(), Theme::RANDOM_SELECTOR);
        // ...while the theme actually in effect is one of the concrete ones.
        let picked = Theme::resolved_theme_name();
        assert_ne!(picked, Theme::RANDOM_SELECTOR);
        assert!(
            Theme::available_theme_names().contains(&picked),
            "drew an unavailable theme: {picked}"
        );

        Theme::apply_named("classic").expect("restore classic");
    }

    #[test]
    fn default_colors_reset_the_terminal_palette() {
        // Regression: switching from a theme that paints the terminal to one
        // that leaves it alone must undo the paint, or the old background
        // survives the switch (modern_light -> classic used to stay light).
        let (_, light) = load_theme_by_name("modern_light").expect("modern_light");
        let painted = terminal_palette_sequence(&light);
        assert!(painted.contains("\x1b]11;#fafafa\x07"), "{painted:?}");

        let (_, classic) = load_theme_by_name("classic").expect("classic");
        let reset = terminal_palette_sequence(&classic);
        assert_eq!(reset, "\x1b]111\x07\x1b]110\x07\x1b]112\x07");
        assert!(!reset.contains("#"), "{reset:?}");
    }

    #[test]
    fn canonical_theme_name_resolves_aliases() {
        assert_eq!(Theme::canonical_theme_name("modern"), "modern_dark");
        assert_eq!(Theme::canonical_theme_name("  MODERN  "), "modern_dark");
        assert_eq!(Theme::canonical_theme_name("dark"), "classic");
        assert_eq!(Theme::canonical_theme_name("classic"), "classic");
        assert_eq!(
            Theme::canonical_theme_name(Theme::RANDOM_SELECTOR),
            Theme::RANDOM_SELECTOR
        );
        assert_eq!(Theme::canonical_theme_name("nope"), "nope");
    }

    #[test]
    fn random_is_a_selector_not_a_theme_file() {
        let error = load_theme_by_name(Theme::RANDOM_SELECTOR).expect_err("selector");
        assert!(format!("{error:#}").contains("selector"), "{error:#}");
    }

    #[test]
    fn palette_only_themes_keep_the_classic_chrome() {
        for name in ["oranguday", "tokyonight", "rosepine-moon"] {
            let (_, theme) = load_theme_by_name(name).expect("shipped theme");
            assert_eq!(theme.chrome, Chrome::Classic, "{name}");
        }
    }

    #[test]
    fn unknown_chrome_is_rejected() {
        let source = format!("chrome = neon\n{}", MINIMAL_THEME);
        let error = parse_theme(&source, "inline").expect_err("unknown chrome");
        assert!(format!("{error:#}").contains("neon"), "{error:#}");
    }

    #[test]
    fn shipped_theme_is_listed() {
        let names = Theme::available_theme_names();
        assert!(names.contains(&"classic".to_string()));
        assert!(names.contains(&"modern_dark".to_string()));
        assert!(names.contains(&"modern_light".to_string()));
        assert!(names.contains(&Theme::RANDOM_SELECTOR.to_string()));
    }

    /// Every required key, with values distinct enough that a mix-up shows up.
    const MINIMAL_THEME: &str = "success = fg:#010203 bold\nerror = fg:#040506\nignore = fg:#070809\ndeep = fg:#0a0b0c\nmuted = fg:#0d0e0f\ncursor_line_bg = fg:#111213 bg:#141516\nselected_file = fg:#171819 bg:#1a1b1c\ncomment_bg = bg:#1d1e1f\ncode_block_bg = bg:#202122\nhighlight = fg:#232425\nwarning = fg:#262728\nuser_input = fg:#292a2b bg:#2c2d2e\nbg_base = #2f3031\ntext_primary = #323334\n";

    #[test]
    fn classic_theme_file_matches_its_in_code_base() {
        // `classic_fallback` is the base every theme file is parsed over and
        // the stand-in when a file can't be read, so it must stay byte-for-byte
        // equivalent to the shipped classic.theme.
        let (name, theme) = load_theme_by_name("classic").expect("classic");
        assert_eq!(name, "classic");
        assert_eq!(theme, classic_fallback());
    }

    #[test]
    fn omitted_keys_inherit_the_classic_base() {
        // A one-line theme is legal: everything it doesn't mention keeps its
        // classic value.
        let theme = parse_theme("highlight = fg:#010203\n", "inline").expect("partial theme");
        let base = classic_fallback();
        assert_eq!(theme.highlight, parse_style("fg:#010203").unwrap());
        assert_eq!(theme.success, base.success);
        assert_eq!(theme.user_input, base.user_input);
        assert_eq!(theme.bg_base, base.bg_base);
        assert_eq!(theme.chrome, base.chrome);

        // An empty file is the classic theme outright.
        assert_eq!(parse_theme("", "inline").expect("empty theme"), base);
    }

    #[test]
    fn unknown_key_is_rejected() {
        // With every key optional, a typo would otherwise resolve silently to
        // the classic value.
        let error = parse_theme("hilight = fg:#010203\n", "inline").expect_err("typo");
        let message = format!("{error:#}");
        assert!(message.contains("unknown key"), "{message}");
        assert!(message.contains("hilight"), "{message}");
        assert!(message.contains("inline:1"), "{message}");
    }

    #[test]
    fn theme_file_parser_reads_styles() {
        let theme = parse_theme(MINIMAL_THEME, "inline").expect("parse theme");
        assert!(theme.success.add_modifier.contains(Modifier::BOLD));
        assert!(matches!(theme.bg_base, Color::Rgb(47, 48, 49)));
        assert!(matches!(theme.text_primary, Color::Rgb(50, 51, 52)));
        // An omitted `chrome` key keeps the original frame.
        assert_eq!(theme.chrome, Chrome::Classic);
    }

    #[test]
    fn theme_file_parser_reads_default_colors_and_reversed() {
        let source = MINIMAL_THEME
            .replace("bg_base = #2f3031", "bg_base = default")
            .replace("text_primary = #323334", "text_primary = default")
            .replace(
                "selected_file = fg:#171819 bg:#1a1b1c",
                "selected_file = reversed",
            );
        let theme = parse_theme(&source, "inline").expect("parse theme");
        assert!(matches!(theme.bg_base, Color::Reset));
        assert!(matches!(theme.text_primary, Color::Reset));
        assert!(
            theme
                .selected_file
                .add_modifier
                .contains(Modifier::REVERSED)
        );
    }

    #[test]
    fn preview_named_does_not_commit_theme() {
        let _guard = theme_test_guard();
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
        assert!(matches!(Theme::current().bg_base, Color::Reset));
    }

    #[test]
    fn apply_named_clears_preview() {
        let _guard = theme_test_guard();
        Theme::apply_named("classic").expect("commit classic");
        Theme::preview_named("tokyonight").expect("preview");
        assert!(Theme::is_previewing());

        Theme::apply_named("oranguday").expect("commit oranguday");
        assert!(!Theme::is_previewing());
        assert_eq!(Theme::current_theme_name(), "oranguday");

        Theme::apply_named("classic").expect("restore classic");
    }
}
