//! Modern dark theme: near-black slate surfaces, electric-blue accent,
//! rounded corners, Inter typography.

use egui::{Color32, Context, CornerRadius, FontDefinitions, FontFamily, FontId, TextStyle, Visuals};

pub const SURFACE_BASE: Color32 = Color32::from_rgb(14, 16, 20);
pub const SURFACE_PANEL: Color32 = Color32::from_rgb(19, 22, 28);
pub const SURFACE_CARD: Color32 = Color32::from_rgb(26, 30, 38);
pub const SURFACE_HOVER: Color32 = Color32::from_rgb(33, 38, 48);
pub const ACCENT: Color32 = Color32::from_rgb(79, 140, 255);
pub const ACCENT_SOFT: Color32 = Color32::from_rgb(58, 92, 153);
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(230, 233, 239);
pub const TEXT_DIM: Color32 = Color32::from_rgb(139, 147, 163);
pub const SUCCESS: Color32 = Color32::from_rgb(74, 200, 140);
pub const WARNING: Color32 = Color32::from_rgb(240, 180, 80);
pub const DANGER: Color32 = Color32::from_rgb(235, 100, 100);

/// Registers embedded fonts and applies the visual style.
pub fn apply(context: &Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "inter".into(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/fonts/Inter-Variable.ttf"
        ))),
    );
    fonts.font_data.insert(
        "jetbrains-mono".into(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/fonts/JetBrainsMono-Regular.ttf"
        ))),
    );
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "inter".into());
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, "jetbrains-mono".into());
    context.set_fonts(fonts);

    context.all_styles_mut(|style| {
        style.visuals = Visuals::dark();
        style.visuals.panel_fill = SURFACE_BASE;
        style.visuals.window_fill = SURFACE_CARD;
        style.visuals.extreme_bg_color = SURFACE_PANEL;
        style.visuals.faint_bg_color = SURFACE_PANEL;
        style.visuals.hyperlink_color = ACCENT;
        style.visuals.warn_fg_color = WARNING;
        style.visuals.error_fg_color = DANGER;
        style.visuals.selection.bg_fill = ACCENT_SOFT;
        style.visuals.selection.stroke.color = TEXT_PRIMARY;
        style.visuals.widgets.noninteractive.weak_bg_fill = SURFACE_PANEL;
        style.visuals.widgets.inactive.weak_bg_fill = SURFACE_CARD;
        style.visuals.widgets.hovered.weak_bg_fill = SURFACE_HOVER;
        style.visuals.widgets.active.weak_bg_fill = SURFACE_HOVER;
        style.visuals.widgets.open.weak_bg_fill = SURFACE_HOVER;
        style.visuals.widgets.hovered.bg_fill = ACCENT_SOFT;
        style.visuals.widgets.active.bg_fill = ACCENT;
        for widget in [
            &mut style.visuals.widgets.noninteractive,
            &mut style.visuals.widgets.inactive,
            &mut style.visuals.widgets.hovered,
            &mut style.visuals.widgets.active,
            &mut style.visuals.widgets.open,
        ] {
            widget.corner_radius = CornerRadius::same(6);
            widget.fg_stroke.color = TEXT_PRIMARY;
        }
        style.visuals.window_corner_radius = CornerRadius::same(10);
        style.visuals.menu_corner_radius = CornerRadius::same(8);
        style.visuals.clip_rect_margin = 0.0;
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.text_styles = [
            (TextStyle::Heading, FontId::new(22.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
            (TextStyle::Button, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(12.0, FontFamily::Proportional)),
        ]
        .into();
    });
}
