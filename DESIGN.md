---
name: RheoJAX Desktop GUI
description: "\"Precision Laboratory\" — a PySide6 scientific-tool interface for rheological data analysis, indigo-and-violet on warm-stone neutrals."
colors:
  deep-indigo: "#4338CA"
  indigo-hover: "#4F46E5"
  indigo-pressed: "#3730A3"
  indigo-tint: "#E0E7FF"
  indigo-whisper: "#EEF2FF"
  bayesian-violet: "#7C3AED"
  violet-hover: "#6D28D9"
  violet-pressed: "#5B21B6"
  violet-tint: "#EDE9FE"
  success: "#059669"
  success-hover: "#047857"
  success-tint: "#D1FAE5"
  warning: "#D97706"
  warning-hover: "#B45309"
  warning-tint: "#FEF3C7"
  error: "#DC2626"
  error-hover: "#B91C1C"
  error-tint: "#FEE2E2"
  info: "#2563EB"
  info-tint: "#DBEAFE"
  canvas-white: "#FFFFFF"
  warm-stone-surface: "#FAFAF9"
  stone-hover: "#F5F5F4"
  stone-active: "#E7E5E4"
  warm-charcoal: "#1C1917"
  stone-600: "#57534E"
  stone-500: "#78716C"
  stone-400: "#A8A29E"
  paper: "#FFFFFF"
  stone-border: "#D6D3D1"
  stone-border-hover: "#A8A29E"
  focus-indigo: "#6366F1"
  stone-divider: "#E7E5E4"
  chart-1: "#4338CA"
  chart-2: "#7C3AED"
  chart-3: "#059669"
  chart-4: "#D97706"
  chart-5: "#DC2626"
  chart-6: "#DB2777"
typography:
  display:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "36pt"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "normal"
  heading:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "24pt"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "normal"
  title:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "14pt"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "normal"
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "12pt"
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: "normal"
  label:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "10pt"
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: "0.5px"
  mono:
    fontFamily: "JetBrains Mono, Cascadia Code, SF Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"
    fontSize: "10pt"
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: "normal"
rounded:
  none: "0px"
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  full: "9999px"
spacing:
  xxs: "2px"
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  xxl: "32px"
  xxxl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.deep-indigo}"
    textColor: "{colors.paper}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  button-primary-hover:
    backgroundColor: "{colors.indigo-hover}"
    textColor: "{colors.paper}"
  button-primary-pressed:
    backgroundColor: "{colors.indigo-pressed}"
    textColor: "{colors.paper}"
  button-secondary:
    backgroundColor: "{colors.warm-stone-surface}"
    textColor: "{colors.warm-charcoal}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  input:
    backgroundColor: "{colors.canvas-white}"
    textColor: "{colors.warm-charcoal}"
    rounded: "{rounded.md}"
    padding: "8px 12px"
  input-focus:
    backgroundColor: "{colors.canvas-white}"
    textColor: "{colors.warm-charcoal}"
  card:
    backgroundColor: "{colors.warm-stone-surface}"
    rounded: "{rounded.xl}"
    padding: "24px"
  tab-selected:
    textColor: "{colors.indigo-hover}"
    typography: "{typography.title}"
---

# Design System: RheoJAX Desktop GUI

## 1. Overview

**Creative North Star: "The Precision Laboratory"**

RheoJAX's GUI is a PySide6 instrument panel for JAX-accelerated rheological analysis — fitting constitutive models to experimental data, running Bayesian inference (NLSQ → NUTS), and inspecting the results through interactive plots and diagnostics. It is a scientist's workbench, not a marketing surface: deep indigo (#4338CA) carries primary actions and focus states, a violet accent (#7C3AED) is reserved for the Bayesian/scientific-inference surfaces specifically, and everything else sits on warm-stone neutrals (#FAFAF9 surfaces, #1C1917 warm-charcoal text) rather than clinical gray. Inter carries every UI role at a tight, many-stepped point scale (9–36pt) tuned for a desktop app viewed at consistent DPI, not a fluid web scale.

This is explicitly **not** a generic SaaS dashboard. It rejects the cookie-cutter admin-template look — pastel badges, rounded-everything card grids, dashboard-as-a-service genericness — in favor of a tool that behaves like calibrated lab equipment: sharp small-radius edges (4–8px, not 16px+), exact gradient states on primary actions, deliberate 2px focus rings, and state color used only for state (hover/focus/active/disabled/error/success/warning/info), never decoration.

**Key Characteristics:**
- Two full themes (light default, dark), plus a "System" mode that follows the OS color scheme live via `QStyleHints.colorSchemeChanged` (Qt 6.5+).
- Flat by default: no drop shadows anywhere in the live stylesheet. Depth comes from tonal layering (base → surface → elevated → hover → active) and 1px borders, not elevation.
- Underline-indicator tab navigation (Material-style), not boxed/pill tabs.
- Qt has no native card widget; `QGroupBox` with a `[gbStyle="card"]` property variant is the app's real card primitive — a plain `QFrame.card` class exists as a simpler secondary option.
- Precise > decorative: gradients and focus rings are used deliberately on primary actions and inputs; nothing else gets a gradient.

## 2. Colors

Restrained-to-Committed: warm-stone neutrals carry the vast majority of every screen, with deep indigo as the single primary accent and violet reserved for Bayesian/inference-specific surfaces — never both used for the same role on one screen.

### Primary
- **Deep Indigo** (`#4338CA`): the one primary-action color — buttons, active tab underline, checked toggle states. Hover lightens to Indigo Hover (`#4F46E5`); press darkens to Indigo Pressed (`#3730A3`). Indigo Tint (`#E0E7FF`) and Indigo Whisper (`#EEF2FF`) are the tint/subtle backgrounds for selected rows and light fills — never used as text or border color.

### Secondary
- **Bayesian Violet** (`#7C3AED`): reserved specifically for Bayesian/NUTS-inference surfaces (the "scientific accent," distinct from the general-purpose primary). Hover `#6D28D9`, pressed `#5B21B6`, tint `#EDE9FE`.

### Neutral
- **Canvas White** (`#FFFFFF`): base app background, and the inverse text color on filled indigo/violet surfaces (aliased here as **Paper**).
- **Warm Stone Surface** (`#FAFAF9`): the surface color for cards, panels, elevated content — a warm, not cool, near-white (the source docstring calls this "warm stone neutrals" explicitly).
- **Stone Hover** (`#F5F5F4`) / **Stone Active** (`#E7E5E4`): hover and pressed-background states for surfaces and list rows.
- **Warm Charcoal** (`#1C1917`): primary body text — warm-toned, not pure black, matching the stone neutral family.
- **Stone 600** (`#57534E`): secondary text (field descriptions, less-emphasized labels).
- **Stone 500** (`#78716C`): muted text (placeholder-adjacent, tertiary info).
- **Stone 400** (`#A8A29E`): disabled text.
- **Stone Border** (`#D6D3D1`) / **Stone Border Hover** (`#A8A29E`): default and hover-state 1px borders on inputs, cards, tables.
- **Stone Divider** (`#E7E5E4`): subtle/light dividers, distinct from the slightly darker default border.
- **Focus Indigo** (`#6366F1`): the 2px focus-ring color on every focusable input and button — always this exact indigo, regardless of the element's own accent color.

### Semantic State
- **Success** (`#059669`) / hover `#047857` / tint `#D1FAE5`: successful fits, passing diagnostics, float64-enabled indicator.
- **Warning** (`#D97706`) / hover `#B45309` / tint `#FEF3C7`.
- **Error** (`#DC2626`) / hover `#B91C1C` / tint `#FEE2E2`: failed fits, validation errors, disabled float64 indicator.
- **Info** (`#2563EB`) / tint `#DBEAFE`: informational badges and callouts.

### Data Visualization
- **Chart 1–6** (`#4338CA`, `#7C3AED`, `#059669`, `#D97706`, `#DC2626`, `#DB2777`): the categorical palette for multi-series rheological plots (fit curves, multi-mode decompositions). Chart 1 and 2 intentionally reuse Deep Indigo and Bayesian Violet so plotted series stay visually anchored to the same brand hues used in the chrome around them.

### Dark Theme
The frontmatter above documents the light theme (default); `DarkColorPalette` mirrors every role with dark-appropriate values rather than a naive invert:
- Deep Indigo → `#6366F1` (lighter, for contrast against a dark surface); hover `#818CF8`; pressed `#4F46E5`.
- Bayesian Violet → `#A78BFA`; hover `#8B5CF6`; pressed `#7C3AED`.
- Canvas White / Warm Stone Surface → `#0F172A` (base) / `#1E293B` (surface, slate-800) — the dark theme moves off the warm-stone family into slate, since a warm-tinted dark surface reads muddy rather than warm.
- Warm Charcoal (text) → `#F8FAFC` (slate-50); Stone 600/500/400 → `#94A3B8` / `#64748B` / `#475569`.
- Stone Border → `#334155`; Focus Indigo stays `#6366F1` unchanged in both themes — the one color that never shifts.
- Tooltips invert again on top of the theme inversion: dark-theme tooltips get a near-white background with near-black text, the opposite of the surrounding dark surface, preserving the "inverted-contrast chip" identity from Section 5 in both themes.

### Named Rules
**The One Accent, One Purpose Rule.** Deep Indigo is the general primary action color; Bayesian Violet is reserved for inference-specific surfaces. They are never substituted for each other, and never both used as the primary accent on the same screen.

**The Warm Neutral Rule (light theme).** Every light-theme neutral (surfaces, text, borders) carries a slight warm (stone) tint, never a cool gray or a true achromatic scale — deliberate brand identity, not an accident. The dark theme intentionally breaks from this into a cooler slate family (see Dark Theme below); that's a considered per-theme choice, not drift to fix.

## 3. Typography

**Display Font:** Inter (with `-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif` fallbacks)
**Mono Font:** JetBrains Mono (with `Cascadia Code, SF Mono, Menlo, Consolas, DejaVu Sans Mono, monospace` fallbacks) — defined as a token but not yet wired into any live component treatment; reserved for future numeric/code display.

**Character:** A single, well-tuned UI sans carries every role from body text to buttons to data tables — no display/body pairing. Product UI, not marketing: familiarity over expression.

### Hierarchy
- **Display** (700, 36pt, line-height 1.1): reserved token, no current component uses it — the largest step in the scale, held for a future hero/splash treatment.
- **Heading** (600, 24pt, line-height 1.2): reserved token, similarly unused today; the natural size for a dialog or page-level title if one is ever needed.
- **Title** (600, 14pt, line-height 1.3): section headers (`QGroupBox` titles, panel headers) and the selected-tab label.
- **Body** (400, 12pt, line-height 1.4): default widget text throughout — the workhorse size. Prose inside the app (help text, descriptions) should stay within 65–75ch; data tables can run denser.
- **Label** (500, 10pt, letter-spacing 0.5px, line-height 1.3): field labels, badges, unselected tab text, table column headers (headers additionally go uppercase in `base.qss`).

### Named Rules
**The Fixed-Scale Rule.** Every size in this scale is a fixed point value (9pt–36pt), never `clamp()` or a fluid unit. Desktop users view at consistent DPI; a shrinking heading in a resized panel is a bug, not responsiveness.

## 4. Elevation

Flat by default — there is no `box-shadow` anywhere in the live stylesheet (Qt's shadow support via QSS is unreliable, so the codebase avoids it entirely). Depth is conveyed through tonal layering (Canvas White → Warm Stone Surface → Stone Hover → Stone Active, each one step darker) and 1px borders, not shadow. A `Shadows` token set exists in code (`SHADOW_SM/MD/LG` as x/y/blur/rgba tuples) intended for `QGraphicsDropShadowEffect` on individual widgets, but as of this scan no component wires it in — treat it as a reserved affordance, not a documented pattern to imitate yet.

### Named Rules
**The No-Shadow Rule.** Never introduce a `box-shadow` in QSS. If a component needs to read as "elevated," lighten its background one step (surface → hover-tone) and/or add a 1px border — don't reach for a shadow.

## 5. Components

Precise and instrumented: buttons carry an exact gradient and a firm pressed-state depth cue, inputs get a deliberate 2px focus ring, and every interactive state (hover/focus/active/disabled) is specified — nothing is left to Qt's unstyled defaults.

### Buttons
- **Shape:** 6px radius (`rounded.md`), `min-width: 80px`, 8px/16px padding.
- **Primary:** a vertical gradient from Deep Indigo to Indigo Pressed (`linear-gradient(#4338CA → #3730A3)` light theme; `#6366F1 → #4F46E5` dark theme), white/inverse text, weight 600. This is the one gradient surface in the system — reserved for primary actions only.
- **Hover:** brightens to a lighter gradient stop plus an Indigo Pressed border.
- **Pressed:** collapses to a flat Indigo Pressed fill with asymmetric top/bottom padding (9px/7px) to read as physically depressed.
- **Focus:** 2px Focus Indigo ring, with padding reduced by 1px on that side to avoid layout shift.
- **Disabled:** flat Stone Hover background, Stone Border outline, Stone 400 text — no gradient.
- **Secondary:** flat Warm Stone Surface background, 1px Stone Border outline, Warm Charcoal text; hover brightens to Stone Hover / Stone Border Hover.
- **Compact density:** 4px/12px padding, 10pt, 60px min-width — used where the primary/secondary treatment would be too heavy (dense toolbars, inline table actions).

### Inputs / Fields
- **Style:** flat Canvas White fill, 1px Stone Border outline, 6px radius, 8px/12px padding.
- **Focus:** 2px Focus Indigo ring (1px padding compensation to avoid shift) — identical treatment to buttons, so focus reads consistently across every interactive control.
- **Disabled:** Warm Stone Surface fill, Stone 400 text, Stone Border outline.
- Dropdowns (`QComboBox`) match the input treatment exactly, plus a borderless dropdown-arrow zone; their popup list uses the same visual language as tables (see below).

### Lists / Tables
- **Style:** Warm Stone Surface background, 1px border (Stone Border for tables/trees, Stone Divider for simple lists), 8px radius, no focus outline (Qt's default dotted rect is suppressed), zebra striping via an alternate-row background.
- **Selection:** Indigo Tint background with matching selection text color.
- **List items** specifically render as discrete rounded pills (6px item radius, small margin) rather than full-bleed rows, with a separate hover fill distinct from selection.
- **Column headers:** Stone Hover background, uppercase Label-style text (10pt/600/0.5px tracking), bottom + right 1px Stone Border dividers forming a grid.

### Cards / Containers
Qt has no native card widget; this system uses two:
- **`QGroupBox` (primary card primitive):** 1px Stone Border, 8px radius by default, floating uppercase Label-style title. A `[gbStyle="card"]` variant steps up to 12px radius / 20px padding for a more prominent "elevated" feel (via background/border only — no shadow, per Section 4); a `[gbStyle="minimal"]` variant strips the border entirely for lightweight grouping.
- **`QFrame.card` (secondary/simple option):** Warm Stone Surface fill, 1px Stone Border, 12px radius, 24px padding. A `.card-clickable` variant swaps the border to Focus Indigo on hover — no background or shadow change, just the border-color cue.
- **Corner style:** 12px radius (`rounded.xl`) is the card standard; plain panels/sidebars use a smaller 8–10px radius.

### Navigation (Tabs)
- **Style:** underline-indicator, not boxed. Unselected tabs: Stone 600 text, Label weight (500), 11pt, transparent 3px bottom border reserved as a placeholder.
- **Hover:** text brightens to Warm Charcoal, subtle background wash.
- **Selected:** text becomes Indigo Hover, the 3px bottom border fills solid Deep Indigo, weight steps up to 600 (Title-level) — a classic Material-style underline indicator.

### Tooltips
Inverted-contrast floating chip: background and text swap relative to the surrounding surface (light theme: Warm Charcoal background with off-white text; dark theme: the inverse) — no border, 6px radius, 6px/10px padding, 10pt. Deliberately high-contrast and surface-independent by design.

### Dialogs
Minimal treatment: only the content-area background is themed (Warm Stone Surface); dialog chrome/frame is left to the native OS window decoration. No custom radius, border, or padding beyond that. Notably thinner than the card treatment — dialogs are not meant to compete visually with in-app cards.

## 6. Do's and Don'ts

### Do:
- **Do** reserve the primary gradient (`#4338CA → #3730A3`) for primary-action buttons only — it is the system's one deliberately decorative surface, and its rarity is what makes it read as "the important action."
- **Do** use the exact same 2px Focus Indigo (`#6366F1`) ring on every focusable control — buttons, inputs, and any future interactive component. Consistency of the focus treatment is what makes keyboard navigation legible.
- **Do** keep every neutral (surface, text, border) on the warm-stone family. A cool gray anywhere in this system is a bug.
- **Do** use `QGroupBox` as the card primitive for new panels — it already carries the title-label convention the rest of the app expects.
- **Do** convey depth with one tonal step (surface → hover-tone) plus a 1px border, never a shadow.

### Don't:
- **Don't** build a generic SaaS-dashboard look: no pastel badge grids, no rounded-everything card walls, no dashboard-as-a-service genericness. This is a precision instrument panel for researchers, not a lifestyle admin template.
- **Don't** introduce `box-shadow` anywhere in QSS — see The No-Shadow Rule (Section 4).
- **Don't** use Bayesian Violet as a general-purpose accent. It signals "this is Bayesian/inference-specific" and loses that meaning if reused elsewhere.
- **Don't** let the `LIGHT_TOKENS`/`DARK_TOKENS` QSS-substitution dicts drift from the `ColorPalette`/`DarkColorPalette` dataclasses documented above. As of this scan the two have already diverged in places (e.g. the QSS dict's `bg_surface` is `#FFFFFF` where `ColorPalette.BG_SURFACE` is `#FAFAF9`) — they're meant to encode the same design intent and should be resynced, not treated as two independent sources of truth.
- **Don't** use a fluid/`clamp()` type scale. Every size here is fixed-pt; desktop viewing at consistent DPI doesn't need fluid typography, and it would break Qt's own font metrics.
