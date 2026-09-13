//! Lumen dark theme: graphite canvas, hairline borders, one lumen-cyan
//! accent. Status hues (green/amber/red) are reserved for index state and
//! never used decoratively.

use egui::{Color32, Context, CornerRadius, FontDefinitions, FontFamily, FontId, RichText, Stroke, TextStyle, Ui, Visuals};

pub const SURFACE_BASE: Color32 = Color32::from_rgb(13, 14, 18); // app canvas
pub const SURFACE_PANEL: Color32 = Color32::from_rgb(19, 20, 26); // top bar, sidebar, footer
pub const SURFACE_CARD: Color32 = Color32::from_rgb(26, 28, 36); // inputs, result cards
pub const SURFACE_ELEVATED: Color32 = Color32::from_rgb(32, 34, 43); // preview excerpts
pub const SURFACE_HOVER: Color32 = Color32::from_rgb(34, 36, 46);
pub const SURFACE_SUNKEN: Color32 = Color32::from_rgb(16, 17, 22); // command field well
pub const BORDER: Color32 = Color32::from_rgb(38, 41, 51);
pub const BORDER_STRONG: Color32 = Color32::from_rgb(52, 56, 70);
pub const ACCENT: Color32 = Color32::from_rgb(56, 189, 248); // lumen cyan
pub const ACCENT_DEEP: Color32 = Color32::from_rgb(18, 135, 194); // solid accent fills
pub const ON_ACCENT: Color32 = Color32::from_rgb(8, 19, 26); // text on accent fills
pub const ACCENT_SOFT: Color32 = Color32::from_rgb(21, 41, 58); // selection fill
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(232, 234, 238);
pub const TEXT_DIM: Color32 = Color32::from_rgb(155, 161, 174);
pub const SUCCESS: Color32 = Color32::from_rgb(61, 201, 139);
pub const WARNING: Color32 = Color32::from_rgb(232, 169, 78);
pub const DANGER: Color32 = Color32::from_rgb(240, 86, 94);

/// Material Icons (classic) codepoints used across the UI.
pub mod icons {
    pub const SEARCH: &str = "\u{e8b6}";
    pub const MIC: &str = "\u{e029}";
    pub const SETTINGS: &str = "\u{e8b8}";
    pub const INFO: &str = "\u{e88f}";
    pub const BACK: &str = "\u{e5c3}";
    pub const FORWARD: &str = "\u{e5c4}";
    pub const UP: &str = "\u{e5d8}";
    pub const REFRESH: &str = "\u{e5d5}";
    pub const NEW_FOLDER: &str = "\u{e2cc}";
    pub const CLOSE: &str = "\u{e5cd}";
    pub const SORT: &str = "\u{e164}";
    pub const FOLDER: &str = "\u{e2c7}";
    pub const FOLDER_OPEN: &str = "\u{e2c8}";
    pub const CHECK: &str = "\u{e876}";
    pub const DELETE: &str = "\u{e872}";
    pub const OPEN: &str = "\u{e89e}";
    pub const CONTENT_COPY: &str = "\u{e14d}";
    pub const DRIVE_FILE_RENAME: &str = "\u{e923}";
    pub const CONTENT_PASTE: &str = "\u{e14f}";
    pub const VISIBILITY: &str = "\u{e8f4}";
    // Content kind glyphs (replace emoji, which render inconsistently).
    pub const DESCRIPTION: &str = "\u{e873}"; // text
    pub const PICTURE_AS_PDF: &str = "\u{ea0b}"; // pdf
    pub const AUDIOTRACK: &str = "\u{e3a1}"; // audio
    pub const MOVIE: &str = "\u{e02c}"; // video
    pub const IMAGE: &str = "\u{e3f4}"; // image
    pub const INSERT_DRIVE_FILE: &str = "\u{e24d}"; // other
}

/// Material icon codepoint rendered in the icon font at `size`.
fn icon_font(codepoint: &str, size: f32) -> egui::RichText {
    egui::RichText::new(codepoint)
        .font(egui::FontId::new(size, FontFamily::Name("material-icons".into())))
}

/// A Material icon sized for buttons (primary color).
pub fn icon(codepoint: &str, size: f32) -> egui::RichText {
    icon_font(codepoint, size).color(TEXT_PRIMARY)
}

/// A Material icon in the secondary color (kind glyphs, subdued controls).
pub fn icon_dim(codepoint: &str, size: f32) -> egui::RichText {
    icon_font(codepoint, size).color(TEXT_DIM)
}

/// A Material icon in an explicit color.
pub fn icon_colored(codepoint: &str, size: f32, color: Color32) -> egui::RichText {
    icon_font(codepoint, size).color(color)
}

/// Button content combining a Material icon and a text label in one widget
/// (mixed fonts need a layout job; a plain string would tofu the codepoint).
pub fn icon_label(codepoint: &str, text: &str, text_color: Color32) -> egui::WidgetText {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        codepoint,
        0.0,
        egui::TextFormat::simple(
            FontId::new(15.0, FontFamily::Name("material-icons".into())),
            TEXT_DIM,
        ),
    );
    job.append(
        "  ",
        0.0,
        egui::TextFormat::simple(FontId::new(13.0, FontFamily::Proportional), text_color),
    );
    job.append(
        text,
        0.0,
        egui::TextFormat::simple(FontId::new(13.0, FontFamily::Proportional), text_color),
    );
    egui::WidgetText::LayoutJob(job.into())
}

// ---------- instrument-style composition helpers ----------

/// Uppercase micro section label (telemetry-panel register).
pub fn micro(text: &str) -> RichText {
    RichText::new(text.to_uppercase())
        .size(10.0)
        .strong()
        .monospace()
        .color(TEXT_DIM)
}

/// Monospace secondary readout (counts, rates, timestamps).
pub fn mono_dim(text: impl Into<String>) -> RichText {
    RichText::new(text.into()).size(12.0).monospace().color(TEXT_DIM)
}

/// Monospace primary readout (live values the eye tracks).
pub fn mono_primary(text: impl Into<String>) -> RichText {
    RichText::new(text.into()).size(12.0).monospace().color(TEXT_PRIMARY)
}

/// Paints a small filled status dot and returns its rect.
pub fn status_dot(ui: &mut Ui, color: Color32, diameter: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(diameter + 4.0, diameter + 4.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), diameter / 2.0, color);
}

/// Thin horizontal proportion bar painted at the current cursor.
pub fn prop_bar(ui: &mut Ui, width: f32, height: f32, fraction: f32, fill: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(height as u8 / 2), SURFACE_HOVER);
    let filled = egui::Rect::from_min_size(
        rect.min,
        egui::vec2(width * fraction.clamp(0.0, 1.0), height),
    );
    painter.rect_filled(filled, CornerRadius::same(height as u8 / 2), fill);
}

/// Small keycap chip (⌘F, ↵, ↑↓) painted at the current cursor.
pub fn kbd_chip(ui: &mut Ui, text: &str) {
    let label = RichText::new(text).size(10.0).monospace().color(TEXT_DIM);
    let response = ui.add(
        egui::Button::new(label)
            .fill(SURFACE_CARD)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(4)),
    );
    let _ = response;
}

/// Hairline frame used for grouped controls (segmented nav, readout chips).
pub fn group_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE_CARD)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(4, 3))
}

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
        "material-icons".into(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/fonts/MaterialIcons-Regular.ttf"
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
    fonts.families.insert(FontFamily::Name("material-icons".into()), vec!["material-icons".into()]);
    context.set_fonts(fonts);

    context.all_styles_mut(|style| {
        style.visuals = Visuals::dark();
        style.visuals.panel_fill = SURFACE_BASE;
        style.visuals.window_fill = SURFACE_CARD;
        // TextEdit backgrounds come from extreme_bg_color.
        style.visuals.extreme_bg_color = SURFACE_CARD;
        style.visuals.faint_bg_color = SURFACE_PANEL;
        style.visuals.hyperlink_color = ACCENT;
        style.visuals.warn_fg_color = WARNING;
        style.visuals.error_fg_color = DANGER;
        style.visuals.selection.bg_fill = ACCENT_SOFT;
        style.visuals.selection.stroke.color = TEXT_PRIMARY;
        // Hairline borders, not fills, define interactive surfaces.
        let border = Stroke::new(1.0, BORDER);
        style.visuals.widgets.noninteractive.weak_bg_fill = SURFACE_PANEL;
        style.visuals.widgets.noninteractive.bg_stroke = border;
        style.visuals.widgets.inactive.weak_bg_fill = SURFACE_CARD;
        style.visuals.widgets.inactive.bg_stroke = border;
        style.visuals.widgets.hovered.weak_bg_fill = SURFACE_HOVER;
        style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.5));
        style.visuals.widgets.active.weak_bg_fill = SURFACE_HOVER;
        style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
        style.visuals.widgets.open.weak_bg_fill = SURFACE_HOVER;
        style.visuals.widgets.open.bg_stroke = border;
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
        style.visuals.popup_shadow = egui::Shadow::NONE;
        style.visuals.window_shadow = egui::Shadow::NONE;
        style.visuals.clip_rect_margin = 0.0;
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.text_styles = [
            (TextStyle::Heading, FontId::new(20.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace)),
            (TextStyle::Button, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(12.0, FontFamily::Proportional)),
        ]
        .into();
    });
}
