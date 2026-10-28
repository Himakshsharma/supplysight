---
name: SupplySight
description: AI procurement and demand intelligence for SAP MM/PP – an operator's console for buyers, planners and managers.
colors:
  ink: "#0f172a"
  muted: "#475569"
  line: "#e2e8f0"
  surface: "#ffffff"
  surface-2: "#f1f5f9"
  accent: "#2563eb"
  neutral-series: "#94a3b8"
  risk-high: "#b91c1c"
  risk-medium: "#b45309"
  risk-low: "#15803d"
  series-teal: "#0d9488"
  series-violet: "#7c3aed"
  series-slate: "#64748b"
typography:
  title:
    fontFamily: "Source Sans Pro, system-ui, sans-serif"
    fontWeight: 700
    letterSpacing: "-0.02em"
  metric:
    fontFamily: "Source Sans Pro, system-ui, sans-serif"
    fontSize: "1.6rem"
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  body:
    fontFamily: "Source Sans Pro, system-ui, sans-serif"
    fontSize: "1rem"
  chart:
    fontFamily: "Source Sans Pro, system-ui, sans-serif"
    fontSize: "12px"
rounded:
  badge: "999px"
  focus: "4px"
components:
  risk-badge-high:
    backgroundColor: "{colors.risk-high}"
    textColor: "{colors.surface}"
    rounded: "{rounded.badge}"
    padding: "1px 8px"
  risk-badge-medium:
    backgroundColor: "{colors.risk-medium}"
    textColor: "{colors.surface}"
    rounded: "{rounded.badge}"
    padding: "1px 8px"
  risk-badge-low:
    backgroundColor: "{colors.risk-low}"
    textColor: "{colors.surface}"
    rounded: "{rounded.badge}"
    padding: "1px 8px"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface}"
---

# Design System: SupplySight

## Overview

**Creative North Star: "The Plant Control Room"**

SupplySight is an Operate-mode surface. A buyer, planner or manager sits at a plant-office desk in daylight, works through a queue, and decides. The interface should disappear into that task: familiar Streamlit controls, dense tables, quiet charts, and colour that appears only when it means something. Brand lives in precision: consistent labels, tabular numerals, one accent, and risk colours that never decorate.

Source of truth: `dashboard/common.py` (Python tokens, CSS, chart styling) and `.streamlit/config.toml` (Streamlit theme). Change them together with this file.

**Key Characteristics:**
- Light theme, chosen for daytime office use.
- One accent (blue) for primary actions, current selection and the forecast line.
- Risk colour (red / amber / green) carries meaning only, and the level is always written out as text.
- Material Symbols icons; no emoji used as icons.
- Human column labels in every table (SAP field names stay in the data layer).

## Colors

Restrained slate neutrals with one blue accent and a three-step risk scale.

### Primary
- **Console Blue** (#2563eb): primary buttons, selection, focus ring, the P50 forecast line and its band (15 % alpha). 5.2:1 on white.

### Neutral
- **Ink** (#0f172a): primary text and chart titles.
- **Slate Muted** (#475569): metric labels, axis ticks, legends, captions. 7.6:1 on white, 6.9:1 on the sidebar.
- **Hairline** (#e2e8f0): chart gridlines and axis lines.
- **Sidebar Mist** (#f1f5f9): the second neutral layer (sidebar, inputs).
- **De-emphasis Grey** (#94a3b8): marks that should recede (features that have not drifted).

### Risk (semantic only)
- **High** (#b91c1c) 6.5:1 with white text · **Medium** (#b45309) 5.0:1 · **Low** (#15803d) 5.0:1.

### Chart series
Accent, then teal #0d9488, violet #7c3aed, slate #64748b, in that order (Plotly `colorway`).

### Named Rules
**The Meaning-Only Rule.** Red, amber and green appear only for risk or drift status, never as decoration or category colour.
**The One Accent Rule.** Blue marks the thing to act on or the model's answer. Everything else is neutral.

## Typography

**Body / UI Font:** Source Sans Pro (Streamlit's default), reused in Plotly charts so charts and UI share one face.

### Hierarchy
- **Title** (700, Streamlit h1, -0.02em): one per page, no icon prefix.
- **Subheader** (Streamlit h3): section labels such as "Top flagged purchase orders".
- **Metric** (1.6rem, tabular numerals): KPI values line up digit by digit.
- **Body** (1rem): explanations, reasons, citations.
- **Chart** (12px; 14px titles, left-aligned).

## Layout

Streamlit `layout="wide"`. Metric rows on top (5 on Overview, 3–4 elsewhere), then a 3:2 or 1:1 split for chart and table pairs, then detail. Filters sit in a single row above the table they control.

## Elevation & Depth

Flat. Depth comes from the sidebar's tonal layer and from bordered containers (vendor duplicate pairs). No shadows.

## Shapes

Streamlit default control radius. Risk badges are full pills.

## Components

- **Risk badge** (`common.badge`): pill, white text on the risk colour, with the level always written out.
- **Tables** (`common.show_df`): full width, no index, column labels mapped through `common.LABELS`. `column_config` keys use the labelled names.
- **Charts** (`common.style_fig`): `plotly_white` base, theme font, hairline grid, muted ticks, horizontal legend above the plot.
- **Submit buttons** (`common.submit`): Material icon with a fallback for older Streamlit. The primary verdict ("True issue") uses `type="primary"`.
- **Empty states**: `st.info` with an icon that says what to change (widen filters, run a script), never just "nothing here".
- **Missing metrics** (`common.num`): an en dash, never "nan".
- **Focus**: 2px accent outline with a 2px offset on links, buttons, tabs and expanders.

## Do's and Don'ts

- **Do** use `:material/<name>:` shortcodes for icons in page_icon, buttons, expanders and alerts.
- **Do** add new colours to `common.py` first, then to this file.
- **Don't** hard-code hex values in pages.
- **Don't** use emoji as icons or as decoration in titles.
- **Don't** show raw SAP field names (ebeln, bedat, werks) as table headers.
- **Don't** rely on colour alone to convey risk.
