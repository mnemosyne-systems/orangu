use ratatui::Frame;

use crate::tui::screen::ScreenRenderArgs;

pub fn render(frame: &mut Frame, args: &ScreenRenderArgs<'_>) {
    crate::tui::screen::draw_screen(frame, args.clone());
}
