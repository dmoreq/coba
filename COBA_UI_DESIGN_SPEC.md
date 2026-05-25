# COBA Web UI — Design Specification

**Version:** 3.0 (minimalist revision)
**Style direction:** Minimalist / Progressive Disclosure
**Framework:** Flet (Python)
**Reference apps:** Linear, Raycast, Vercel dashboard

> **v3 summary:** All individual zone card boxes are eliminated. The workspace is one unified surface with internal column dividers. Secondary information (arm scores, param values, knowledge values, stage labels) is hidden by default and revealed on hover. A single teal accent (`#059669`) replaces the dual teal/amber system. Configuration and theory content collapse to near-invisible affordances until intentionally opened.

---

## 0. v3 Minimalist Design System

### 0.1 Core Philosophy Change

The v1/v2 design used explicit zone cards with colored backgrounds and top-border accents to communicate structure. v3 removes all that surface chrome and relies on **whitespace, typographic hierarchy, and hover-reveal** to create order.

The guiding constraint: **if you can't see it at a glance, you can see it in one hover**. Nothing important is more than one mouse movement away. Nothing unnecessary occupies persistent space.

### 0.2 Unified Workspace Surface

The three-pane workspace is a **single bordered container**, not three separate cards:

```css
.workspace {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  border: 0.5px solid var(--color-border-tertiary);
  border-radius: 10px;
  overflow: hidden;
}
.zone + .zone {
  border-left: 0.5px solid var(--color-border-tertiary);
}
```

No individual zone cards. No zone accent top borders. No zone background tints. One box, three columns.

### 0.3 Single Accent Color

Drop the dual teal/amber system. Use one accent color everywhere:

| Usage | Value |
|---|---|
| Selected arm border | `#059669` |
| Phase badge dot + text | `#059669` |
| Active stage dot | `#059669` |
| Active tab underline | `#059669` |
| Hover link / configure | `#059669` |
| Reward chart line | `#059669` |

Amber is removed entirely. The agent zone uses the same neutral border treatment as everything else.

### 0.4 Progressive Disclosure Rules

| Element | Default state | Revealed on |
|---|---|---|
| Arm UCB score | `opacity: 0` | `.arm-row:hover` |
| Stage stepper labels | `color: transparent` | `.st-node:hover` or active node |
| Knowledge table numeric values | `color: transparent` | `.know-row:hover` |
| Configure panel (arm rates) | Hidden (`display: none`) | Click "configure" ghost link |
| Tooltip content | `opacity: 0` | `[data-tip]` element hover |
| Formula tooltip on param | Mouse-following div | `?` badge hover |

**Rule:** counts and labels that are always useful (arm name, step number, phase name) are always visible. Computed values (scores, means, variances) are secondary — hide by default, reveal on hover.

### 0.5 Mouse-Following Tooltip

A single `<div id="tt">` sits at root level (above all pane content). It follows `mousemove` and activates when the cursor enters any `[data-tip]` element.

```js
const tt = document.getElementById('tt');
document.addEventListener('mousemove', e => {
  tt.style.left = (e.clientX + 14) + 'px';
  tt.style.top  = (e.clientY + 14) + 'px';
});
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mouseenter', () => {
    tt.textContent = el.dataset.tip;
    tt.style.opacity = '1';
  });
  el.addEventListener('mouseleave', () => {
    tt.style.opacity = '0';
  });
});
```

Tooltip div styles:
```css
#tt {
  position: fixed; /* avoids overflow:hidden clipping on workspace */
  background: var(--color-background-primary);
  border: 0.5px solid var(--color-border-secondary);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  max-width: 220px;
  line-height: 1.5;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.12s;
  z-index: 9999;
}
```

Use `position: fixed` (not `absolute`) to avoid clipping by `overflow: hidden` on the workspace grid.

### 0.6 Arm Rows (not arm cards)

Arms are rendered as simple rows inside the environment zone — no individual card boxes:

```css
.arm-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 0;
  border-bottom: 0.5px solid var(--color-border-tertiary);
  transition: padding-left 0.12s;
}
.arm-row.selected {
  border-left: 2px solid #059669;
  padding-left: 6px;
}
.arm-score {
  opacity: 0;
  transition: opacity 0.15s;
}
.arm-row:hover .arm-score { opacity: 1; }
```

The selected arm gets a left border accent. Score values are opacity-hidden until hover.

### 0.7 Stage Stepper (dots only)

The stepper is a row of 8px dots connected by 0.5px lines. Labels have `color: transparent` and transition to `text_tertiary` on hover or when active.

```css
.st-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--color-border-secondary); }
.st-node.done .st-dot, .st-node.cur .st-dot { background: #059669; }
.st-node.cur .st-dot { transform: scale(1.4); }
.st-label { color: transparent; font-size: 11px; position: absolute; top: 14px; transition: color 0.15s; }
.st-node:hover .st-label, .st-node.cur .st-label { color: var(--color-text-tertiary); }
```

### 0.8 Configure Ghost Link

The configure affordance inside the environment zone is a near-invisible text link that becomes visible on hover:

```css
.cfg-link { opacity: 0.45; font-size: 11px; color: var(--color-text-tertiary); transition: opacity 0.2s; }
.cfg-link:hover { opacity: 1; color: #059669; }
```

On click, the arm-rate slider panel toggles `display: block`. No modal, no slide animation — just appears.

### 0.9 Knowledge Table — hover-reveal values

Non-obvious numeric values (means, UCB scores, variances) use `color: transparent` on `.know-val` elements:

```css
.know-val { color: transparent; transition: color 0.12s; }
.know-row:hover .know-val { color: var(--color-text-secondary); }
```

Counts and arm names remain always-visible (`.know-label` class, no transparency).

### 0.10 Theory Section (no card chrome)

The theory block sits inside the interaction zone with no card wrapper. It's separated from the chart area by a `border-top: 0.5px solid`. Zone title is uppercase 10px. Formula is a `<code>` block with `background: var(--color-background-secondary)` — no separate border or radius needed.

### 0.11 KPI Row (Arena)

Three numbers in a borderless flex row. No metric card boxes. Pure typographic hierarchy:
- 16px / 500 value
- 10px / `text_tertiary` label below

---

## 1. Design Philosophy

COBA is an educational tool for researchers and practitioners learning contextual bandit algorithms. The design should feel like a well-made academic notebook — clear, readable, structured — not a flashy product dashboard.

**Principles:**
- **Legibility first.** Every number, label, and formula must be effortlessly readable. No decorative noise competing with content.
- **Spatial consistency.** A predictable 8px grid means users never have to scan for where things are.
- **Progressive disclosure.** Complexity is revealed as the user advances through lesson stages; it is never dumped on them at once.
- **State is always visible.** The current step count, selected arm, phase of the interaction loop, and lesson objective should be visible at a glance without hunting.

---

## 2. Design Tokens

### 2.1 Color

| Token | Light | Dark | Usage |
|---|---|---|---|
| `bg_primary` | `#FFFFFF` | `#1A1A1A` | Card surfaces |
| `bg_secondary` | `#F7F7F5` | `#242424` | Page background, metric cards |
| `bg_tertiary` | `#F0EFEB` | `#2E2E2E` | Input fills, code blocks |
| `surface_border` | `#E8E8E4` | `#383838` | All card / zone borders |
| `environment_zone_bg` | `#F0F9F5` | `#0D2420` | Environment pane tint |
| `agent_zone_bg` | `#FEF9F0` | `#24180A` | Agent pane tint |
| `interaction_zone_bg` | `#FFFFFF` | `#1A1A1A` | Interaction pane (neutral) |
| `environment_accent` | `#1D9E75` | `#3DC99B` | Teal — environment zone header, selected arm border, chart reward line |
| `agent_accent` | `#BA7517` | `#F5B840` | Amber — agent zone header, exploration indicators |
| `success_feedback` | `#1D9E75` | `#3DC99B` | Positive reward, objective met |
| `regret_feedback` | `#C0392B` | `#E57373` | Negative reward, regret chart line |
| `text_primary` | `#1A1A1A` | `#E8E8E4` | Main text |
| `text_secondary` | `#6B7280` | `#9CA3AF` | Labels, captions, muted content |
| `text_muted` | `#9CA3AF` | `#6B7280` | Placeholders, disabled |
| `text_on_teal` | `#FFFFFF` | `#0D2420` | Text on teal accent backgrounds |
| `chart_line_reward` | `#1D9E75` | `#3DC99B` | Cumulative reward series |
| `chart_line_regret` | `#C0392B` | `#E57373` | Cumulative regret series |
| `chart_grid` | `#E8E8E4` | `#383838` | Chart grid lines |

**Zone accent stripe (2px rule):** Each zone card has a 2px left border in its accent color to anchor visual identity without occupying header space.

### 2.2 Typography

| Scale | Size | Weight | Usage |
|---|---|---|---|
| `CAPTION` | 11px | 400 | Pull-count labels, axis ticks, timestamps |
| `SMALL` | 12px | 400 | Secondary labels, chip text, tooltips |
| `BODY` | 14px | 400 | Default running text, table cells |
| `LABEL` | 12px | 500 | Zone titles (uppercase + letter-spacing), section headers |
| `TITLE` | 15px | 500 | Card headings, arm labels |
| `HEADING` | 20px | 500 | Page headings |

> Note: Remove `CAPTION = 10` from `theme/constants.py` — 10px is below WCAG legibility thresholds. Replace with `CAPTION = 11`.

**Monospace font** (`var(--font-mono)` / `"JetBrains Mono", "Fira Code", monospace`): Use for all formula blocks, score values, parameter readouts, and trace table numbers.

**Zone title style:** `11px, UPPERCASE, letter-spacing: 0.06em, font-weight: 500, text_secondary color`. Never use colored text for zone titles; color is communicated by the zone dot and accent stripe.

### 2.3 Spacing (8px grid)

| Token | Value | Usage |
|---|---|---|
| `XS` | 4px | Icon-to-label gaps, within table rows |
| `SM` | 8px | Between chips, between arm cards |
| `MD` | 12px | Card internal padding |
| `LG` | 16px | Between major sections |
| `XL` | 24px | Page-level padding |
| `XXL` | 32px | Between the stepper and workspace |

### 2.4 Borders & Radius

- All card borders: `0.5px solid surface_border`
- Zone cards: `border-radius: 10px`
- Chips, metric badges: `border-radius: 6px`
- Formula blocks: `border-radius: 6px`
- Selected arm card: `border-left: 3px solid environment_accent` (overrides the 0.5px left border)
- Buttons: `border-radius: 8px`

### 2.5 Elevation

COBA uses **no drop shadows**. Depth is communicated through background color contrast:
- Page bg (`bg_secondary`) is slightly darker than card bg (`bg_primary`).
- Zone backgrounds tint distinguishes zone identity from neutral surfaces.
- Active/selected states use border color changes and background tints, never elevation.

---

## 3. App Shell

### 3.1 Top Navigation Bar (replaces sidebar)

The sidebar is eliminated. A single compact top nav gives the workspace the full page width — critical for the lesson's 3-pane layout.

| Property | Value |
|---|---|
| Height | 46px |
| Background | `bg_primary` |
| Border-bottom | `0.5px solid surface_border` |
| Padding | `0 14px` |
| Layout | flex row: `[Logo] [Nav tabs] [Nav controls → right]` |

**Logo** (left): `"COBA"` 14px / 500 + `"Bandit Lab"` subtitle 11px / `text_secondary` inline.

**Nav tabs** (center-left, after logo): `Home | Lesson | Arena | Sandbox | Compare`
- Padding: `5px 9px`, border-radius: `6px`
- Home: icon-only (`ti-home`), all others text labels at 12px
- Default: `text_secondary`, transparent background
- Hover: `bg_secondary`
- Active: `bg = #E1F5EE`, `color = #0F6E56`, `font-weight: 500`

**Nav controls** (right, `margin-left: auto`):
- World selector: `<select>` at 11px with 9px "WORLD" label above
- Policy selector: `<select>` at 11px with 9px "POLICY" label above
- Theme toggle: 28×28px icon button

> Moving world/policy selectors into the top nav (not the AppBar title area and not a sidebar footer) keeps them visible and accessible on every route without consuming vertical or horizontal real estate.

---

## 4. Home Page

The home page is currently a dead-end text placeholder. Replace with:

### 4.1 Hero

```
Contextual Bandit Lab                    [20px / 500]
An interactive environment for exploring [13px / text_secondary]
sequential decision algorithms.
```

No illustration, no hero image — keep it text-forward and academic.

### 4.2 Destination Cards (2×2 grid)

Each card:
- White background, `border-radius: 12px`, `0.5px solid surface_border`
- Top: small icon in a colored 32×32 rounded square
- Title (14px / 500), description (12px / text_secondary, 2 lines max)
- Bottom: `"Open →"` link in accent color (12px / 500)

| Route | Icon | Icon bg | Accent |
|---|---|---|---|
| Lesson | `school` | `environment_zone_bg` | `environment_accent` (teal) |
| Arena | `chart-line` | blue-50 (`#E6F1FB`) | blue-600 (`#185FA5`) |
| Sandbox | `flask` | amber-50 (`#FAEEDA`) | amber-600 (`#854F0B`) |
| Compare | `columns` | gray-50 | `text_secondary` |

### 4.3 Continue Card

Below the grid, a single card showing the last active session:
- World/policy thumbnail, lesson title, stage progress, step count
- `"Resume →"` filled button (teal)

---

## 5. Workspace Layout (3-Pane)

All active routes (Lesson, Arena, Sandbox, Compare) use the same base workspace layout.

### 5.1 Grid specification

```
[Environment 1fr] [Interaction 2fr] [Agent 1fr]
```

Gap between panes: `SM (8px)`.

On screens narrower than 900px (future): stack vertically.

### 5.2 Zone Cards — Unified Color System

All three zones use **identical backgrounds** (`bg_primary` / white). Zone identity comes exclusively from a 2px colored top border — no background tinting. This creates one cohesive visual theme instead of three competing color stories.

| Zone | Top border | Dot color | Tab accent |
|---|---|---|---|
| Environment | `#1D9E75` (teal) | `#1D9E75` | teal |
| Interaction | `surface_border` (neutral) | `text_secondary` | — |
| Agent | `#BA7517` (amber) | `#BA7517` | amber |

Card properties (all zones):
- `background: bg_primary`
- `border: 0.5px solid surface_border`
- `border-top: 2px solid [zone-accent]`
- `border-radius: 10px`
- `padding: 11px 12px`

**Zone header:**
```
● ENVIRONMENT                    [⚙ Configure]
```
- 6px colored dot + UPPERCASE zone title (10px / 500 / `text_secondary`)
- Optional action button (right-aligned) for Environment and Agent
- No separator line — the 2px top border already anchors the zone identity

---

## 6. Component Specifications

### 6.1 Context Chips (Environment zone)

Display each context feature as a key:value chip.

```
[time_slot: evening]    [user_type: genre_fan]
[platform: mobile]
```

- `border-radius: 6px`
- `bg_tertiary` fill, `surface_border` border
- Key: 12px / 400 / `text_secondary`
- Value: 12px / 500 / `text_primary`
- Colon as separator: no space before, 1 space after
- Flow: `Row(wrap=True)` with `SM` gap

### 6.2 Arm Cards (Interaction zone)

**Default state:**
- White background, `surface_border` border, `border-radius: 8px`
- Arm label: 13px / 500 / `text_primary`
- Score value: 12px / 400 / `text_secondary`, monospace, right-aligned
- Score bar: 3px height, `bg_tertiary` track, `text_muted` fill, fills proportional to score

**Selected state:**
- `border-left: 3px solid environment_accent`
- Background: `environment_zone_bg`
- Score bar fill: `environment_accent`
- Score value: `environment_accent`
- No "← selected" string — the visual state is self-evident

**Hover state:**
- `surface_border` → `environment_accent` border (all sides 0.5px, keep left accent on selected)
- Subtle background tint: `environment_zone_bg` at 50% opacity

### 6.3 Loop Visualizer (Interaction zone)

Four phases in a horizontal row with connector lines:

```
[①] ──── [②] ──── [③] ──── [④]
Context  Arm    Reward   Learn
```

Phase node: 28×28px circle
- Done: `environment_accent` fill, white text
- Active: `agent_accent` fill, white text, subtle pulse animation (optional)
- Upcoming: `bg_tertiary` fill, `text_muted` text, `surface_border` border

Connector line: 1px / `surface_border` for upcoming, `environment_accent` for done segments.
Phase label: 10px / `text_secondary` centered below node. Active: `agent_accent` / 500.

Position: sticky at top of the Interaction zone, always visible.

### 6.4 Step Control Bar (Interaction zone)

Pinned to the bottom of the Interaction zone card.
Top separator: `0.5px solid surface_border`.

```
[▶ Step (42)]  [⏵ Play]  [↺ Reset]           ε = 0.10
```

- "Step" button: filled (teal background, white text) — primary action
- "Play" / "Pause": outlined button — toggles, label changes based on state
- "Reset": outlined button with restart icon — destructive, right-aligned or separated
- Parameter readout (`ε = 0.10`): right-aligned, 12px / monospace / `text_secondary`

**Do not pad button labels with spaces.** Use `min_width=90` on Flet buttons instead.

### 6.5 Knowledge Table (Agent zone)

```
Arm          Est. reward    Pulls
──────────────────────────────────
Action Film     0.720          28
RomCom          0.445           8
Documentary     0.310           6
```

- Column headers: 10px / UPPERCASE / `text_secondary`
- Arm name: 12px / 500 / `text_primary`
- Estimated reward: 12px / monospace / `environment_accent`
- Pull count: 12px / `text_muted`, right-aligned in parens or just number
- Row separator: `0.5px solid surface_border` (no separator on last row)
- Best arm row: subtle `environment_zone_bg` background tint

### 6.6 Environment Configure Panel (Environment zone)

A collapsible panel inside the Environment zone card, toggled by a `⚙ Configure` button in the zone header.

**Collapsed (default):** the button reads `⚙ Configure` in `text_secondary`.
**Expanded:** button turns teal-tinted (`#E1F5EE` bg, `#0F6E56` text), panel slides open below a separator.

Panel contents:
```
Arm base rates
──────────────────────────────────
Action Film    [════════░░░░]   0.75
Romantic Comedy [════░░░░░░░]   0.45
Documentary    [███░░░░░░░░░]   0.30

[Apply changes]
```
- Each row: arm name (82px fixed) + range slider (flex:1) + value readout (monospace, 28px)
- Slider: step 0.05, range 0.0–1.0
- "Apply changes" button: teal outlined, calls `SandboxEditor.build_world_override()` and resets simulator

> This makes the hidden `SandboxEditor` class finally reachable from the UI, on every workspace route — not just Sandbox.

### 6.7 Agent Zone Tabs (Knowledge / Config)

The Agent zone has two tabs replacing the flat single-view panel:

```
[Knowledge]  [Config]
──────────────────────────────────
```

Tab style: `3px 9px` padding, `border-radius: 5px`, inactive = outlined neutral, active = amber-tinted (`#FEF3E2` bg, `#854F0B` text, `#FAC775` border).

**Knowledge tab (default):** knowledge table + pull distribution bars.

**Config tab:** all algorithm parameter sliders for the active policy, with formula and tuning hint displayed inline below each slider — not hidden behind hover.

```
Epsilon (ε)                      0.10
[══════════○─────────────────]
P(explore) = ε
Hint: Start at 0.1; reduce when reward variance is low.

────────────────────────────────
Run settings
  Horizon  [10000]
  Seed     [0]
```

### 6.8 Parameter Sliders (Agent zone Config tab)

Each slider spec:
```
Epsilon (ε)                    [tooltip ?]
[══════════○──────────] 0.10
P(explore) = epsilon
```

- Label: 12px / 500 / `text_primary`
- Tooltip icon (optional): `info-circle` at 14px, shows `ParamTooltip` on hover
- Slider: full width, teal thumb (`environment_accent`), gray track
- Value readout: 12px / monospace, right-aligned
- Formula line below: 11px / monospace / `text_secondary` — always visible (not just on hover)
- Tuning hint: 11px / italic / `text_muted` — shown below formula

> `ParamTooltip.formula` and `ParamTooltip.tuning_hint` are valuable pedagogy. Show them inline below the slider, not hidden behind hover.

### 6.7 Theory Card (Lesson — full width, below workspace)

```
┌────────────────────────────────────────────────┐
│ STAGE 2 · DECISION RULE                        │
│ How ε-Greedy selects an action            [1/5 > 2/5 > 3/5 > 4/5 > 5/5]  │
│                                                │
│ ┌────────────────────────────────────────────┐ │
│ │ a_t = random with P=ε  else argmax(Q)      │ │
│ └────────────────────────────────────────────┘ │
│                                                │
│ Inspect per-arm scores before each action      │
│ to see when exploration fires.                 │
└────────────────────────────────────────────────┘
```

- Stage label: 11px / UPPERCASE / `environment_accent` / 500
- Title: 15px / 500 / `text_primary`
- Formula block: monospace / `bg_tertiary` background / `surface_border` border / `border-radius: 6px` / `12px padding`
- Intuition text: 13px / `text_secondary`
- Practical hint: 12px / italic / `text_muted`

The theory card occupies the full workspace width below the 3-pane row. It scrolls with the page — do not float it.

### 6.8 Stage Stepper (Lesson — above workspace)

```
[✓] ──── [2] ──── [ 3 ] ──── [ 4 ] ──── [ 5 ]
Framing  Decision  Update  Failures  Deploy
```

- Container: white card, `border-radius: 10px`, 12px padding
- Completed circle: `environment_accent` fill + checkmark icon
- Active circle: dark teal (`#0F6E56`) fill + step number
- Upcoming circle: `bg_secondary` fill + `surface_border` border + `text_muted` number
- Connector lines: full width flex, `surface_border` color for upcoming, `environment_accent` for completed segments
- Labels: 10px / centered, `text_secondary` for upcoming, `environment_accent` / 500 for active

**Clicking a completed step** should navigate back to that stage (future enhancement, but include the visual affordance — completed circles have a `cursor: pointer` style).

### 6.9 Objective Meter (Lesson — inside Theory Card)

Shown as two labeled progress bars beside the theory text:

```
Steps    42 / 80    [════════════░░░░░░░░]
Reward  22.4 / 36  [═══════════════░░░░░]
```

- Track: `bg_tertiary`, height 6px, `border-radius: 3px`
- Fill: `environment_accent` for on-track, `regret_feedback` if regret limit is close
- Labels: 11px / `text_secondary`
- When complete: fill changes to full, a green checkmark appears

### 6.10 KPI Metric Cards (Arena — above workspace)

Four cards in a 4-column grid:

```
[Steps] [Cum. Reward] [Cum. Regret] [Best Arm]
 247      148.3          43.7       Action Film
                                    147 pulls (60%)
```

- Background: `bg_secondary` (slightly off-white)
- Label: 11px / UPPERCASE / letter-spacing / `text_secondary`
- Value: 22px / 500 / `text_primary`
- Cum. Reward value: `environment_accent` (green tint)
- Cum. Regret value: `regret_feedback` (red tint)
- Best Arm: 14px / 500 with `metric_sub` at 11px / `text_muted`
- No borders — the background difference from the page is sufficient
- `border-radius: 8px`, `12px padding`

### 6.11 Pull Distribution Bars (Agent zone)

Replace the fixed `100px` max-width bars with responsive proportional bars:

```
Action Film   [══════════════════════] 147
RomCom        [═══════░░░░░░░░░░░░░░]  50
Documentary   [═══════░░░░░░░░░░░░░░]  50
```

- Label: 11px / `text_secondary`, fixed 80px width, truncate with ellipsis
- Track: `flex: 1`, height 8px, `bg_tertiary`, `border-radius: 4px`
- Fill: best arm uses `environment_accent`; other arms use `text_muted`
- Count: 11px / monospace / `text_secondary`, right-aligned 32px

The bar fill width should be computed as `(count / max_count) * 100%` applied as a flex proportion — do not use a fixed pixel width.

### 6.12 Reward / Regret Line Chart (Arena — full width)

Chart panel below the 3-pane workspace:
- Card container, full width
- Header: `"Performance over time"` (13px / 500) + legend (reward teal, regret red, each 12px / line-swatch)
- Axes: left y-axis (value), bottom x-axis (step), 10px / `text_muted` labels
- Grid lines: horizontal only, `chart_grid` color, 0.5px
- Reward series: `chart_line_reward` (`#1D9E75`), 1.5px stroke
- Regret series: `chart_line_regret` (`#C0392B`), 1.5px stroke, dashed if it helps readability
- Padding: `top: 12, right: 12, bottom: 24, left: 40`
- Height: 160px is adequate for a secondary diagnostic chart

### 6.13 Reward Feedback (Interaction zone)

Replace the binary "Success ✓ / No reward ✗" with:

```
+ 1.0  ↑ reward        [green]
  0.0  No reward        [text_muted, not red — neutral is not failure]
- 0.5  Negative reward  [red]
```

- Show the actual numeric reward delta prominently (16px / 500 / colored)
- Subtext: "Cumulative: 22.4" in `text_secondary`
- Duration: flash for `FEEDBACK_FADE (600ms)` then fade to muted

---

## 7. Route-Specific Layouts

### 7.1 Home (`/`)
- Hero section (heading + subtitle)
- Destination card grid (2×2)
- Continue card (if session exists)

### 7.2 Lesson (`/lesson`)
- Stage stepper (full width)
- 3-pane workspace (Environment | Interaction | Agent)
  - Environment: world card + context chips
  - Interaction: loop visualizer (sticky) + arm cards + step control bar
  - Agent: tabbed panel (`Knowledge` tab + `Params` tab)
- Theory card (full width)
- Objective meters (inside theory card, right column)

**Stage navigation:** "Next stage →" button appears inside the theory card once the objective is met. Clicking calls `LessonProgressState.advance()` — this must be wired up.

### 7.3 Arena (`/arena`)
- KPI metric cards (4-column row)
- 3-pane workspace
  - Environment: context chips + step counter
  - Interaction: arm cards + step controls + recent reward feedback
  - Agent: knowledge table + pull distribution bars + policy state
- Charts panel (full width, collapsible — default open)

### 7.4 Sandbox (`/sandbox`)
- 3-pane workspace (same as Arena)
- **Add: World parameter editor panel** (currently built in `sandbox.py` but unconnected)
  - Arm base-rate sliders, one per arm
  - "Apply overrides" button that calls `SandboxEditor.build_world_override()`
  - Horizon slider
  - Show a "World diff" callout when overrides are active

### 7.5 Comparison (`/comparison`)
- Two side-by-side policy selectors (Policy A / Policy B)
- Two mini 1-pane workspaces running in parallel
- Shared bottom chart with two series (one per policy)
- Difference summary card (delta in reward, regret, pull share)

---

## 8. Animation & Transitions

Use `AnimationDurations` from `theme/constants.py`:

| Duration | Usage |
|---|---|
| 300ms | Route transitions (fade), phase context change |
| 400ms | Arm selection highlight |
| 600ms | Reward feedback fade-out |
| 300ms | Chart series update |

For Flet:
- Use `animate_opacity` on reward feedback container
- Use `animate` on arm card background color change (selected → deselected)
- Do NOT animate layout changes (pane width, card resize) — too expensive

---

## 9. Bug Fixes Required (from audit)

These must be resolved alongside the visual redesign:

| # | Fix | File(s) |
|---|---|---|
| 1 | Wire autoplay loop with `asyncio.create_task` | `app.py` |
| 2 | Connect `build_chart_data` → Flet chart render | `ui/charts.py`, `app.py` |
| 3 | Call `lesson_progress.advance()` from UI | `app.py`, new button callback |
| 4 | Use `build_arm_cards()` from `components/interaction.py` | `app.py` |
| 5 | Re-render after theme toggle (`_refresh_view()`) | `components/theme_toggle.py` |
| 6 | Connect `SandboxEditor` to Sandbox route | `sandbox.py`, `app.py` |
| 7 | Connect `AdvancedDebugPane` builders to Arena/Agent | `debug/advanced.py`, `app.py` |
| 8 | Fix global session state for multi-user web deploy | `main.py` |
| 9 | Use `ThreePaneLayoutSpec` width ratios in layout | `layouts/split_workspace.py` |
| 10 | Implement Comparison route with dual-policy support | `app.py`, new route builder |
| 11 | Replace space-padded button labels with `min_width` | `components/interaction.py` |
| 12 | Fix pull bars to use responsive width, not 100px fixed | `components/agent.py` |
| 13 | Replace `CAPTION=10` with `CAPTION=11` | `theme/constants.py` |
| 14 | Neutral reward feedback (0.0 is not a failure) | `components/interaction.py` |

---

## 10. Implementation Checklist

### Phase 1 — Foundation (no visual changes)
- [ ] Fix `CAPTION = 11` in `theme/constants.py`
- [ ] Fix module-level globals in `main.py` (use `page.session`)
- [ ] Replace space-padded button labels with `min_width`
- [ ] Fix pull bar `width=int(100 * pct)` → use expand/proportion
- [ ] Wire `_refresh_view()` after theme toggle

### Phase 2 — Layout & Navigation
- [ ] Replace `NavigationRail` with sidebar layout (180px, with logo + world/policy at bottom)
- [ ] Remove world/policy dropdowns from AppBar
- [ ] Implement Home page (hero + destination cards + continue card)
- [ ] Add stage stepper above lesson workspace
- [ ] Add lesson "Next stage →" button + wire `advance()`

### Phase 3 — Component Upgrades
- [ ] Use `build_arm_cards()` component (not inline text)
- [ ] Add score bar to arm cards
- [ ] Add "selected" badge / left border treatment to selected arm
- [ ] Show `ParamTooltip.formula` and `.tuning_hint` inline below slider
- [ ] Add KPI metric cards row to Arena
- [ ] Add responsive pull distribution bars to Agent zone
- [ ] Redesign reward feedback (numeric delta, neutral for 0)
- [ ] Connect theory card below lesson workspace

### Phase 4 — Charts & Data
- [ ] Implement `build_chart_data()` → Flet `LineChart` render
- [ ] Wire chart controls into `SplitWorkspaceLayout`
- [ ] Fix Arena to pass charts to layout, not `[]`

### Phase 5 — Autoplay & Advanced Features
- [ ] Wire autoplay with `asyncio.create_task`
- [ ] Connect `SandboxEditor` world override to Sandbox route
- [ ] Connect `AdvancedDebugPane` builders to Agent debug tab
- [ ] Implement Comparison route skeleton

---

## 11. File Change Summary

| File | Change |
|---|---|
| `theme/constants.py` | `CAPTION = 11` |
| `theme/tokens.py` | Update light/dark token values to match spec §2.1 |
| `components/theme_toggle.py` | Call `page._refresh_view()` after toggle |
| `components/interaction.py` | Remove space-padding; add score bars to arm cards; neutral reward feedback |
| `components/agent.py` | Responsive pull bars; inline formula/hint below slider |
| `components/shared.py` | Add `build_metric_badge` grid, update section header style |
| `layouts/split_workspace.py` | Respect `ThreePaneLayoutSpec` ratios; always show charts if ArenaMetrics exists |
| `app.py` | Use `build_arm_cards()`; add autoplay task; wire lesson advance; wire charts; add KPI row |
| `main.py` | Replace module globals with `page.session`; wire world/policy selectors to sidebar |
| `ui/charts.py` | Add Flet `LineChart` renderer consuming `ChartData` |
| `ui/view_models.py` | Pass `ArenaMetrics` → `ChartData` into view model |
| `sandbox.py` | Expose `SandboxEditor` controls as Flet param sliders in Sandbox route |

---

*Last updated: 2026-05-25 — v3 minimalist revision (section 0)*
