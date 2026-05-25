# COBA Frontend Redesign — Comprehensive Implementation Plan

> **Architect:** Command Code (Plan Mode)
> **Date:** 2025-05-25
> **Target:** Absolute beginners learning Contextual Bandit algorithms through interactive simulation
> **Tech Stack:** Python 3.10+, Flet, uv (dependency management)

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Design Philosophy & Guiding Principles](#2-design-philosophy--guiding-principles)
3. [Five Layout Design Options](#3-five-layout-design-options)
4. [Recommended Layout: Option C — Split-Workspace Dashboard](#4-recommended-layout-option-c--split-workspace-dashboard)
5. [Folder Structure](#5-folder-structure)
6. [Component Tree & Hierarchy](#6-component-tree--hierarchy)
7. [State Management Architecture](#7-state-management-architecture)
8. [Theme System Design](#8-theme-system-design)
9. [Charting & Data Visualization](#9-charting--data-visualization)
10. [Core Mechanics: Environment vs Agent Visual Separation](#10-core-mechanics-environment-vs-agent-visual-separation)
11. [Interaction Loop Visualization](#11-interaction-loop-visualization)
12. [Algorithm Curriculum & Progression](#12-algorithm-curriculum--progression)
13. [Wireframe Descriptions](#13-wireframe-descriptions)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Verification & Testing Strategy](#15-verification--testing-strategy)

---

## 1. Current State Assessment

### What Works Well (Keep)
- **VM+P (View-Model + Props) pattern**: `RouteUIModel` and frozen dataclass view-models are clean and testable. Keep this separation.
- **DiscreteSimulator**: The agent-environment loop (`World → Context → Policy.select_arm → World.sample_reward → Policy.update`) is well-designed with Protocol-based contracts.
- **World/Preset system**: Seven narrative-themed worlds (Rural Clinic, RidePilot, MovieMatch, etc.) with real-world framing. Excellent for beginners.
- **17 policy implementations**: Rich library of algorithms ready to be exposed visually.
- **TraceBuffer/State**: Clean serialization pipeline for step history.

### What Needs Redesign (Change)
- **No graphical charts**: Chart data is rendered as plain `ft.Text`. Need real-time updating LineChart and BarChart.
- **No theming system**: All colors/styling are hardcoded inline. No dark/light mode.
- **Three-pane layout is rigid**: Left (scene) / Center (treatment) / Right (metrics) doesn't visually separate Environment from Agent — they're mixed across panes.
- **No visual feedback loop**: Steps happen silently. Users don't see the causal chain: Context generated → Arm selected → Reward received → Knowledge updated.
- **Monolithic `main.py`** (661 lines): Rendering, state, and event handling all in one file.
- **No real-time probability sliders**: Users can't tweak ground-truth parameters to test algorithm adaptability.
- **No step-by-step narrative guidance**: Lessons exist but are text-heavy markdown, not visual storytelling.

---

## 2. Design Philosophy & Guiding Principles

### Visual Language
- **Minimalist, high-contrast, web-first**: Generous whitespace, clean typography (system font stack), no decorative chrome.
- **Environment = Cool tones** (blues, teals): Represents the hidden, unknowable world.
- **Agent = Warm tones** (amber, coral): Represents the active, learning decision-maker.
- **Feedback = Transient accents** (green pulse for success, red pulse for regret): Brief, meaningful color shifts that don't persist.
- **Data = Neutral palette** (grays, monochrome charts): Let the data speak without competing with the narrative zones.

### Information Hierarchy
1. **What's happening right now?** (current step, largest visual element)
2. **What has the agent learned?** (knowledge visualization, mid-size)
3. **How well is it doing?** (cumulative metrics, compact)
4. **What can I control?** (settings panel, collapsed/expandable)

### Audience: Absolute Beginners
- Every concept gets a real-world metaphor before any math.
- Visual cause-and-effect replaces text explanations.
- Guided path (Lesson mode) before free exploration (Arena/Sandbox).
- Tooltips appear on hover, not as permanent text clutter.

---

## 3. Five Layout Design Options

### Option A: Storybook Scroll
```
┌────────────────────────────────────────┐
│  STEP 1: The World Generates a Context │ ← Large narrative header
│  ┌──────────────────────────────────┐  │
│  │ [Visual: City map with weather]  │  │ ← Immersive scene illustration
│  │  Time: 5pm | Weather: Rain      │  │
│  └──────────────────────────────────┘  │
│  ↓                                     │
│  STEP 2: The Agent Chooses an Action   │
│  ┌──────────────────────────────────┐  │
│  │ [Arm cards with real-time glow]  │  │
│  └──────────────────────────────────┘  │
│  ↓                                     │
│  STEP 3: The World Responds           │
│  ┌──────────────────────────────────┐  │
│  │ [Reward/Regret animation]        │  │
│  └──────────────────────────────────┘  │
│  ↓                                     │
│  STEP 4: The Agent Learns             │
│  ┌──────────────────────────────────┐  │
│  │ [Knowledge table update anim]    │  │
│  └──────────────────────────────────┘  │
│  [Charts below the fold]               │
└────────────────────────────────────────┘
```
- **Pros:** Strong narrative flow, perfect for absolute beginners, mobile-friendly.
- **Cons:** Requires scrolling to see charts, hard to compare steps side-by-side, less "dashboard" feel.

### Option B: Two-Column Environment/Agent Split
```
┌──────────────────────┬──────────────────────┐
│     ENVIRONMENT      │        AGENT         │
│    (Hidden Truth)    │    (Decision Engine)  │
│                      │                      │
│  ┌────────────────┐  │  ┌────────────────┐  │
│  │ World Context  │  │  │ Arm Selection  │  │
│  │ Time: 5pm      │  │  │ ▸ Priority     │  │
│  │ Weather: Rain  │  │  │   Standard     │  │
│  │ Demand: High   │──▶│   Pool Match   │  │
│  └────────────────┘  │  └───────┬────────┘  │
│                      │          │           │
│  ┌────────────────┐  │          ▼           │
│  │ Hidden Probs   │  │  ┌────────────────┐  │
│  │ Priority: 0.72 │◀──│  │ Reward: +1    │  │
│  │ Standard: 0.45 │  │  │ (Success! 🟢) │  │
│  │ Pool:    0.38  │  │  └───────┬────────┘  │
│  └────────────────┘  │          │           │
│                      │          ▼           │
│  ┌────────────────┐  │  ┌────────────────┐  │
│  │ Parameter      │  │  │ Knowledge Base │  │
│  │ Sliders        │  │  │ [Updated!]     │  │
│  │ [===○====]     │  │  └────────────────┘  │
│  └────────────────┘  │                      │
│                      │  ┌────────────────┐  │
│                      │  │ Charts         │  │
│                      │  │ [Regret Curve] │  │
│                      │  │ [Arm Histogram]│  │
│                      │  └────────────────┘  │
└──────────────────────┴──────────────────────┘
```
- **Pros:** Crystal-clear Environment vs Agent separation, great for teaching the dual-reality concept.
- **Cons:** Limited horizontal space per column, charts feel cramped.

### Option C: Split-Workspace Dashboard **(RECOMMENDED)**
```
┌──────────────────────────────────────────────────────────┐
│  TOP BAR: COBA · RidePilot | ε-Greedy | Step 12/100     │
│  [Dark Mode ○] [Speed ▸▸] [Lesson ▸]                    │
├──────────┬───────────────────────┬───────────────────────┤
│          │                       │                       │
│  WORLD   │   INTERACTION LOOP    │   AGENT KNOWLEDGE     │
│  (Teal)  │   (Animated Bridge)   │   (Amber)             │
│          │                       │                       │
│ Context: │ ① Context ────────▶  │  Arm         Pulls    │
│ Time:5pm │                       │  Priority    8 ████   │
│ Rain: Y  │ ② ◀─────── Arm Pick  │  Standard    3 ██     │
│ Demand:H │                       │  Pool        1 █      │
│          │ ③ Reward ────────▶   │                       │
│ ┌──────┐ │                       │  Knowledge Table:     │
│ │Hidden│ │ ④ ◀─── Update Model   │  Priority: μ=0.72    │
│ │Truth │ │                       │  Standard: μ=0.45    │
│ │[⚙]   │ │  [Step ▸] [▶ Play]   │  Pool:     μ=0.38    │
│ └──────┘ │  [↺ Reset] [⏩ Run N] │                       │
│          │                       │                       │
├──────────┴───────────────────────┴───────────────────────┤
│  CHARTS ZONE (collapsible)                               │
│  ┌──────────────────────┐  ┌───────────────────────────┐ │
│  │ Cumulative Regret    │  │ Arm Selection Histogram   │ │
│  │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ │  │ Priority ████████         │ │
│  │                      │  │ Standard ███              │ │
│  └──────────────────────┘  └───────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```
- **Pros:** Three distinct visual zones, natural left-to-right data flow, charts have full-width real estate below, best for dashboard-style interactivity.
- **Cons:** Slightly more complex to implement than two-column split.

### Option D: Tabbed Workbench
```
┌──────────────────────────────────────────┐
│ [Simulation] [Analysis] [Configure] [Help]│ ← Tabs
├──────────────────────────────────────────┤
│  (Tab content fills remaining space)     │
│                                          │
│  Simulation tab: Environment + Agent     │
│  Analysis tab: Full charts + export      │
│  Configure tab: World params + algorithm │
│  Help tab: Theory + real-world examples  │
│                                          │
└──────────────────────────────────────────┘
```
- **Pros:** No scrolling, each concern gets full screen real estate.
- **Cons:** Can't see simulation AND charts simultaneously. Breaks the "immediate feedback" principle. Less dashboard-like.

### Option E: Card Grid Mosaic
```
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Environment │ │  Agent      │ │  Controls   │
│   Card      │ │  Action     │ │   Card      │
│             │ │  Card       │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Regret     │ │  Arm Hist.  │ │  Knowledge  │
│  Chart      │ │  Chart      │ │  Table      │
└─────────────┘ └─────────────┘ └─────────────┘
```
- **Pros:** Highly modular, each card is self-contained, responsive to different screen sizes.
- **Cons:** No clear narrative flow, cards feel disconnected, harder to show the Environment→Agent interaction loop.

---

## 4. Recommended Layout: Option C — Split-Workspace Dashboard

### Why Option C
1. **Natural narrative flow**: Left (Environment) → Center (Interaction) → Right (Agent) mirrors the causal chain.
2. **Full-width charts**: Charts get the entire bottom row, with generous horizontal space for time-series data.
3. **Distinct visual zones**: Three clearly separated zones enforce the Environment vs Agent duality — the core pedagogical concept.
4. **Dashboard feel**: Controls are positioned at the center of the action (the interaction loop bridge), not tucked away.
5. **Responsive**: On narrow screens, zones stack vertically; on wide screens, they spread horizontally.

### Zone Descriptions

| Zone | Color Identity | Content | Purpose |
|---|---|---|---|
| **World (Left, 25%)** | Teal/Cool tones | Context feature display, hidden ground-truth probability sliders, world description | "This is the unknowable reality the agent faces" |
| **Interaction (Center, 35%)** | Neutral bridge | Step-by-step animation of the 4-phase loop, arm cards with selection glow, reward feedback animation, run controls | "This is the moment-by-moment dance between environment and agent" |
| **Agent (Right, 25%)** | Amber/Warm tones | Arm pull counts, running mean rewards, knowledge state visualization | "This is what the agent has learned so far" |
| **Charts (Bottom, full width)** | Monochrome | Cumulative regret curve, arm selection histogram, collapsible/expandable | "This is the long-term story" |

---

## 5. Folder Structure

```
src/
├── assets/                          # Static assets directory (new)
│   ├── icons/                       # SVG icons for UI elements
│   │   ├── environment.svg
│   │   ├── agent.svg
│   │   ├── step.svg
│   │   ├── play.svg
│   │   ├── reset.svg
│   │   ├── sun.svg                  # Light mode icon
│   │   └── moon.svg                 # Dark mode icon
│   └── illustrations/               # World-specific hero illustrations
│       ├── ridepilot.svg
│       ├── rural_clinic.svg
│       ├── moviematch.svg
│       └── ...
│
├── coba/                            # Core library (unchanged)
│   └── ...
│
├── web/                             # Flet web application (redesigned)
│   ├── main.py                      # Entry point — thin, delegates to AppShell
│   ├── app.py                       # AppShell — root Flet View, theme, routing
│   ├── router.py                    # Route definitions (keep, minor updates)
│   ├── shell.py                     # Shell composition (keep, minor updates)
│   │
│   ├── layouts/                     # Layout definitions (new — extracted from main.py)
│   │   ├── __init__.py
│   │   ├── split_workspace.py       # Option C: 3-zone + bottom charts layout
│   │   ├── two_column.py            # Option B: Environment/Agent split
│   │   ├── storybook.py             # Option A: Vertical scroll narrative
│   │   └── base.py                  # BaseLayout protocol + shared utilities
│   │
│   ├── components/                  # Reusable Flet UI components (new)
│   │   ├── __init__.py
│   │   ├── charts/                  # Chart components
│   │   │   ├── __init__.py
│   │   │   ├── regret_chart.py      # Cumulative regret LineChart
│   │   │   ├── arm_histogram.py     # Arm selection BarChart
│   │   │   ├── reward_timeline.py   # Per-step reward sparkline
│   │   │   ├── knowledge_heatmap.py # Arm × Context knowledge grid
│   │   │   └── chart_theme.py       # Theme-aware chart color/sizing constants
│   │   │
│   │   ├── controls/                # Interactive control components
│   │   │   ├── __init__.py
│   │   │   ├── step_controls.py     # Step, Play/Pause, Reset, Run-N buttons
│   │   │   ├── speed_slider.py      # Animation speed control
│   │   │   ├── probability_sliders.py # Hidden ground-truth probability sliders
│   │   │   ├── world_selector.py    # World/environment dropdown
│   │   │   ├── policy_selector.py   # Algorithm selection dropdown
│   │   │   └── theme_toggle.py      # Dark/light mode switch
│   │   │
│   │   ├── environment/             # Environment zone components
│   │   │   ├── __init__.py
│   │   │   ├── context_display.py   # Current context feature visualization
│   │   │   ├── world_card.py        # World info card (name, description, icon)
│   │   │   └── hidden_truth_panel.py # Collapsible ground-truth probability editor
│   │   │
│   │   ├── agent/                   # Agent zone components
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_table.py   # Running mean rewards per arm
│   │   │   ├── pull_counter.py      # Arm selection histogram (compact)
│   │   │   ├── uncertainty_display.py # UCB confidence bounds visualization
│   │   │   └── policy_state_card.py # Algorithm-specific internal state display
│   │   │
│   │   ├── interaction/             # Interaction loop components
│   │   │   ├── __init__.py
│   │   │   ├── loop_visualizer.py   # 4-phase animated interaction bridge
│   │   │   ├── arm_cards.py         # Arm selection cards with glow/highlight
│   │   │   ├── reward_feedback.py   # Success/failure animation overlay
│   │   │   └── step_indicator.py    # Current step / total steps progress
│   │   │
│   │   └── shared/                  # Shared/common components
│   │       ├── __init__.py
│   │       ├── section_header.py    # Consistent zone headers
│   │       ├── tooltip_icon.py      # Info icon with hover tooltip
│   │       ├── metric_badge.py      # Compact metric display (value + label)
│   │       └── empty_state.py       # Placeholder for uninitialized states
│   │
│   ├── theme/                       # Theming system (new)
│   │   ├── __init__.py
│   │   ├── colors.py                # Semantic color tokens (not raw hex values)
│   │   ├── typography.py            # Font sizes, weights, line heights
│   │   ├── spacing.py               # Spacing scale (xs, sm, md, lg, xl)
│   │   ├── themes.py                # LightTheme and DarkTheme dataclass definitions
│   │   └── theme_manager.py         # Applies theme to Flet page, handles mode toggle
│   │
│   ├── state/                       # State management (extracted from main.py)
│   │   ├── __init__.py
│   │   ├── app_state.py             # Central AppState — replaces _SimSession globals
│   │   ├── simulation_controller.py # Orchestrates DiscreteSimulator + RunController
│   │   ├── interaction_state.py     # Current phase of the 4-step loop, animation flags
│   │   └── event_bus.py            # Simple pub/sub for cross-component events
│   │
│   ├── curriculum/                  # Lesson system (keep, enhance visuals)
│   │   └── ...
│   │
│   ├── worlds/                      # Simulation environments (keep)
│   │   └── ...
│   │
│   ├── policies/                    # Policy wrappers (keep)
│   │   └── ...
│   │
│   ├── simulator.py                 # Simulation engine (keep, unchanged)
│   ├── state.py                     # SimulationState, RunConfig (keep)
│   ├── trace.py                     # TraceBuffer (keep)
│   │
│   └── ui/                          # View-models (keep architecture, enhance)
│       ├── __init__.py
│       ├── charts.py                # ChartData (keep, will be consumed by ft.LineChart)
│       ├── context_inspection.py    # (keep)
│       ├── layout.py                # Layout specs (extend for new layouts)
│       ├── lesson_models.py         # (keep)
│       ├── param_controls.py        # (extend with probability slider specs)
│       ├── preferences.py           # (keep)
│       ├── run_controls.py          # (keep)
│       ├── tooltips.py              # (keep)
│       ├── view_models.py           # (extend for new layout types)
│       └── components/              # View-model components (keep pattern)
│           ├── __init__.py
│           ├── scene_panel.py
│           ├── treatment_card.py
│           ├── batch_summary_panel.py
│           ├── snapshot_diff_view.py
│           └── trace_table.py
│
└── tests/
    └── web/                         # Test suite (extend)
        ├── test_components/
        ├── test_theme/
        ├── test_state/
        └── test_layouts/
```

---

## 6. Component Tree & Hierarchy

```
AppShell (app.py)
├── ThemeManager
├── NavigationRail
│   ├── Home "/"
│   ├── Lesson "/lesson"     ← Guided path
│   ├── Arena "/arena"       ← Free experimentation
│   ├── Sandbox "/sandbox"   ← Parameter tweaking
│   └── Comparison "/comparison"
│
└── RouteContent (varies by route)
    │
    ├── [HOME] WelcomeScreen
    │   ├── HeroIllustration
    │   ├── QuickStartCard → navigates to /lesson
    │   └── ExploreCard → navigates to /arena
    │
    ├── [LESSON] GuidedLessonView
    │   ├── LessonProgressBar (Stage 1/5)
    │   ├── NarrativeBanner ("You are a doctor choosing treatments...")
    │   ├── SplitWorkspaceLayout
    │   │   ├── WorldZone
    │   │   │   ├── WorldCard
    │   │   │   ├── ContextDisplay
    │   │   │   └── HiddenTruthPanel (locked until later stages)
    │   │   ├── InteractionZone
    │   │   │   ├── LoopVisualizer
    │   │   │   ├── ArmCards
    │   │   │   ├── RewardFeedback
    │   │   │   ├── StepIndicator
    │   │   │   └── StepControls
    │   │   └── AgentZone
    │   │       ├── KnowledgeTable
    │   │       ├── PullCounter
    │   │       └── PolicyStateCard
    │   ├── ChartsZone (collapsed initially)
    │   │   ├── RegretChart
    │   │   └── ArmHistogram
    │   └── LessonObjectiveCard (bottom overlay)
    │
    ├── [ARENA] ArenaView
    │   ├── SplitWorkspaceLayout
    │   │   ├── WorldZone (with WorldSelector, ProbabilitySliders)
    │   │   ├── InteractionZone (full controls + LoopVisualizer)
    │   │   └── AgentZone (live knowledge + policy state)
    │   ├── ChartsZone (expanded, real-time)
    │   │   ├── RegretChart (animated updates)
    │   │   ├── ArmHistogram (animated updates)
    │   │   └── RewardTimeline
    │   └── PolicySelector + SpeedSlider (top bar)
    │
    ├── [SANDBOX] SandboxView
    │   ├── SplitWorkspaceLayout
    │   │   ├── WorldZone (editable HiddenTruthPanel)
    │   │   ├── InteractionZone
    │   │   └── AgentZone
    │   └── ChartsZone
    │
    └── [COMPARISON] ComparisonView
        ├── PolicySelector (multi-select, 2-3 policies)
        ├── SideBySideCharts
        │   ├── RegretOverlay (multiple policies)
        │   ├── ArmDistributionComparison
        │   └── CumulativeRewardComparison
        └── MetricsTable
```

---

## 7. State Management Architecture

### Design: Centralized AppState with Lightweight Pub/Sub

The current pattern uses module-level globals (`_session`, `_page`, `_pref_store`). The redesign introduces a single `AppState` dataclass with an `EventBus` for cross-component communication, keeping the VM+P pattern for rendering.

```
┌─────────────────────────────────────────────────────┐
│                    AppState                          │
│  ┌─────────────┐  ┌──────────────────┐              │
│  │ ThemeState  │  │ SimulationState  │              │
│  │ - mode      │  │ - config         │              │
│  │ - colors    │  │ - current_step   │              │
│  └─────────────┘  │ - cum_reward     │              │
│                   │ - cum_regret     │              │
│  ┌─────────────┐  │ - trace          │              │
│  │ UIState     │  └──────────────────┘              │
│  │ - route     │                                    │
│  │ - autoplay  │  ┌──────────────────┐              │
│  │ - speed     │  │ InteractionState │              │
│  │ - collapsed │  │ - phase(0-3)     │              │
│  └─────────────┘  │ - selected_arm   │              │
│                   │ - last_reward    │              │
│  ┌─────────────┐  │ - animation_flag │              │
│  │ Preferences │  └──────────────────┘              │
│  │ - world_id  │                                    │
│  │ - policy    │  ┌──────────────────┐              │
│  │ - params    │  │ LessonState      │              │
│  └─────────────┘  │ - stage          │              │
│                   │ - objectives     │              │
│                   └──────────────────┘              │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│                    EventBus                          │
│  Events:                                            │
│  - STEP_COMPLETED(step_result)                      │
│  - ARM_SELECTED(arm_id, context)                    │
│  - REWARD_RECEIVED(reward, regret)                  │
│  - KNOWLEDGE_UPDATED(policy_state)                  │
│  - THEME_CHANGED(mode)                              │
│  - WORLD_CHANGED(world_id)                          │
│  - POLICY_CHANGED(policy_id)                        │
│  - RESET_TRIGGERED                                  │
└──────────┬──────────────────────────────────────────┘
           │ subscribe
           ▼
┌─────────────────────────────────────────────────────┐
│              UI Components (subscribers)             │
│  Each component subscribes to relevant events.       │
│  Components DON'T mutate state directly — they       │
│  call SimulationController methods which update      │
│  AppState and emit events.                           │
└─────────────────────────────────────────────────────┘
```

### Data Flow: Step Interaction

```
User clicks "Step"
  → StepControls.on_step_click()
    → SimulationController.step_once()
      1. Advance DiscreteSimulator by 1 step
      2. Update AppState.simulation_state
      3. Compute new InteractionState (phase transitions)
      4. Emit events: ARM_SELECTED → REWARD_RECEIVED → KNOWLEDGE_UPDATED
    → Each subscribed component receives its event:
      - LoopVisualizer: Animates the 4-phase transition
      - ArmCards: Glows selected arm, dims others
      - RewardFeedback: Shows green/red pulse
      - KnowledgeTable: Updates running means with highlight
      - Charts: Appends new data point with animation
      - StepIndicator: Increments counter
    → Build new RouteUIModel from AppState
    → Render updated widget tree
```

### Key Design Decisions
- **AppState is NOT a global**: It's passed down through Flet's `page.session` (per-user session state in web mode).
- **EventBus is lightweight**: A simple dict of `event_name → list[callable]` with async dispatch. No external dependency.
- **Components never hold mutable simulation state**: They render from AppState → view-model, receive events to trigger re-renders.
- **SimulationController is the single writer**: All state mutations go through it, making the system predictable and debuggable.

---

## 8. Theme System Design

### Approach: Semantic Color Tokens + Flet `Theme` Objects

Instead of inline hex values, define **semantic color tokens** that map to different hex values depending on light/dark mode. Flet's `ft.Theme` and `ft.ThemeMode.SYSTEM` handle the heavy lifting.

### Color Token Definitions

```python
# src/web/theme/colors.py

@dataclass(frozen=True)
class ColorTokens:
    # Surfaces
    bg_primary: str           # Main background
    bg_secondary: str         # Card backgrounds
    bg_tertiary: str          # Elevated surfaces (modals)
    surface_border: str       # Card/zone borders

    # Zones (semantic)
    environment_zone_bg: str  # Left zone background tint
    agent_zone_bg: str        # Right zone background tint
    interaction_zone_bg: str  # Center zone background

    # Accents
    environment_accent: str   # Teal (used in environment zone headers, icons)
    agent_accent: str         # Amber (used in agent zone headers, icons)
    success_feedback: str     # Green pulse for positive reward
    regret_feedback: str      # Red pulse for negative regret
    selected_glow: str        # Highlight color for selected arm

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str
    text_on_accent: str       # Text on colored backgrounds

    # Charts
    chart_bg: str             # Transparent in both modes
    chart_grid: str           # Subtle grid lines
    chart_line_primary: str   # Main data series
    chart_line_secondary: str # Comparison series
    chart_bar_fill: str       # Histogram bars

    # Controls
    control_bg: str
    control_border: str
    control_fg: str           # Button text
    slider_track: str
    slider_thumb: str
```

### Light Theme Values

```python
LIGHT_TOKENS = ColorTokens(
    bg_primary="#FAFAFA",
    bg_secondary="#FFFFFF",
    bg_tertiary="#F5F5F5",
    surface_border="#E0E0E0",
    environment_zone_bg="#F0F7FA",     # Very subtle teal tint
    agent_zone_bg="#FFF8F0",          # Very subtle amber tint
    interaction_zone_bg="#FFFFFF",
    environment_accent="#00796B",      # Teal 700
    agent_accent="#E65100",           # Amber 900
    success_feedback="#2E7D32",       # Green 800
    regret_feedback="#C62828",        # Red 800
    selected_glow="#FFB74D",          # Amber 300
    text_primary="#212121",
    text_secondary="#616161",
    text_muted="#9E9E9E",
    text_on_accent="#FFFFFF",
    chart_bg="#00000000",            # Transparent
    chart_grid="#E0E0E0",
    chart_line_primary="#00796B",
    chart_line_secondary="#E65100",
    chart_bar_fill="#90A4AE",        # Blue Grey 300
    control_bg="#FFFFFF",
    control_border="#BDBDBD",
    control_fg="#212121",
    slider_track="#BDBDBD",
    slider_thumb="#00796B",
)
```

### Dark Theme Values

```python
DARK_TOKENS = ColorTokens(
    bg_primary="#121212",
    bg_secondary="#1E1E1E",
    bg_tertiary="#2C2C2C",
    surface_border="#333333",
    environment_zone_bg="#0D2028",     # Very subtle teal tint on dark
    agent_zone_bg="#281A0A",          # Very subtle amber tint on dark
    interaction_zone_bg="#1E1E1E",
    environment_accent="#4DB6AC",      # Teal 300
    agent_accent="#FFB74D",           # Amber 300
    success_feedback="#66BB6A",       # Green 400
    regret_feedback="#EF5350",        # Red 400
    selected_glow="#FF8F00",          # Amber 800
    text_primary="#E0E0E0",
    text_secondary="#9E9E9E",
    text_muted="#616161",
    text_on_accent="#121212",
    chart_bg="#00000000",
    chart_grid="#333333",
    chart_line_primary="#4DB6AC",
    chart_line_secondary="#FFB74D",
    chart_bar_fill="#546E7A",        # Blue Grey 600
    control_bg="#2C2C2C",
    control_border="#444444",
    control_fg="#E0E0E0",
    slider_track="#444444",
    slider_thumb="#4DB6AC",
)
```

### Implementation: ThemeManager

```python
# src/web/theme/theme_manager.py

class ThemeManager:
    """Applies color tokens to Flet page.theme and page.dark_theme."""

    @staticmethod
    def apply_theme(page: ft.Page, mode: str) -> None:
        tokens = LIGHT_TOKENS if mode == "light" else DARK_TOKENS
        page.theme_mode = ft.ThemeMode.LIGHT if mode == "light" else ft.ThemeMode.DARK

        page.theme = ft.Theme(
            color_scheme_seed=tokens.environment_accent,
            font_family="system-ui, -apple-system, sans-serif",
        )
        page.dark_theme = ft.Theme(
            color_scheme_seed=DARK_TOKENS.environment_accent,
            font_family="system-ui, -apple-system, sans-serif",
        )

        # Store tokens in page.session for component access
        page.session.set("color_tokens", tokens)

    @staticmethod
    def get_tokens(page: ft.Page) -> ColorTokens:
        return page.session.get("color_tokens")
```

### Theme Toggle Component

```python
# src/web/components/controls/theme_toggle.py

class ThemeToggle:
    """Dark/light mode switch using ft.Switch with sun/moon icons."""

    @staticmethod
    def build(page: ft.Page) -> ft.Control:
        current = page.session.get("color_tokens")
        is_dark = page.theme_mode == ft.ThemeMode.DARK

        return ft.IconButton(
            icon=ft.Icons.DARK_MODE if is_dark else ft.Icons.LIGHT_MODE,
            tooltip="Toggle dark mode",
            on_click=lambda e: _toggle_theme(page),
        )
```

### Key Principle: Theme Inheritance

All child components inherit theme colors by reading from `ThemeManager.get_tokens(page)` — never by hardcoding hex values. This ensures:
- Sub-components automatically adapt to dark/light mode.
- Chart colors sync with the ambient theme.
- Future theme variants (high-contrast, colorblind-friendly) only require a new `ColorTokens` instance.

---

## 9. Charting & Data Visualization

### Library Selection: Flet Native Charts (`ft.LineChart`, `ft.BarChart`)

**Decision:** Use Flet's built-in chart controls. No external charting library.

**Why not Plotly/Matplotlib:**
- Plotly requires HTML/JS interop or static image generation — breaks real-time animation.
- Matplotlib requires `ft.MatplotlibChart` wrapper — raster images, no smooth animation, heavy dependencies.
- Flet's native charts are GPU-accelerated (via Flutter's Skia engine), support smooth animations on data change, and inherit the Flet theme system.

### Chart Components

#### 1. Cumulative Regret Chart (`RegretChart`)
```python
ft.LineChart(
    data_series=[
        ft.LineChartData(
            data_points=[ft.LineChartDataPoint(x, y) for x, y in regret_points],
            color=tokens.chart_line_primary,
            stroke_width=2,
            curved=True,
            prevent_curve_over_shooting=True,
        )
    ],
    border=ft.Border(bottom=ft.BorderSide(1, tokens.chart_grid)),
    grid_lines=ft.GridLines(interval=10, color=tokens.chart_grid),
    tooltip_bgcolor=tokens.bg_tertiary,
    bgcolor=tokens.chart_bg,  # Transparent — inherits parent bg
    min_y=0,
    animated=True,            # Smooth transition on new data point
    animation_duration=300,
)
```

#### 2. Arm Selection Histogram (`ArmHistogram`)
```python
ft.BarChart(
    bar_groups=[
        ft.BarChartGroup(
            x=i,
            bar_rosters=[
                ft.BarChartRod(
                    from_y=0,
                    to_y=count,
                    color=tokens.agent_accent,
                    tooltip=f"{arm_label}: {count} pulls",
                )
            ],
        )
        for i, (arm_label, count) in enumerate(pull_counts)
    ],
    border=ft.Border(bottom=ft.BorderSide(1, tokens.chart_grid)),
    grid_lines=ft.GridLines(interval=1, color=tokens.chart_grid),
    bgcolor=tokens.chart_bg,
    animated=True,
    animation_duration=300,
)
```

#### 3. Reward Timeline Sparkline (`RewardTimeline`)
A compact LineChart showing per-step reward (0 or 1) as a step function — useful for spotting streaks.

#### 4. Knowledge Heatmap (`KnowledgeHeatmap`)
For contextual bandits: a grid of `ft.Container` cells where each cell represents an (arm × context_bucket) pair, colored by the agent's estimated mean reward. Built with a `ft.Column` of `ft.Row` of `ft.Container` widgets (no native heatmap in Flet).

### Chart Update Pattern

```python
# Charts don't rebuild from scratch — they mutate their data series in place.
# This enables Flet's built-in animation on data change.

def update_regret_chart(chart: ft.LineChart, new_point: tuple[int, float]):
    series = chart.data_series[0]
    series.data_points.append(ft.LineChartDataPoint(*new_point))

    # Trim to last 100 points for performance
    if len(series.data_points) > 100:
        series.data_points = series.data_points[-100:]

    chart.update()  # Flet diffing engine applies minimal DOM updates
```

### Chart Styling Principles
- **Transparent backgrounds**: Charts inherit the zone's background color.
- **No chart junk**: Remove default axes titles, reduce grid lines to minimum, hide legend when there's only one series.
- **Color syncing**: Chart colors come from `ColorTokens` — they switch automatically with dark/light mode.
- **Smooth animations**: `animated=True` on all chart controls with `animation_duration=300ms`.

---

## 10. Core Mechanics: Environment vs Agent Visual Separation

### The Pedagogical Challenge

Beginners conflate "what the world actually does" with "what the agent thinks." The UI must constantly reinforce that these are two separate realities:

| Reality | Questions |
|---|---|
| **Environment (Ground Truth)** | What are the actual probabilities? What context is the world generating? What reward would each arm give? |
| **Agent (Learned Model)** | What does the algorithm think the probabilities are? Which arm does it believe is best? How confident is it? |

### Visual Separation Strategy

#### Zone Background Tinting
- **Left zone (Environment)**: Subtle teal tint (`environment_zone_bg`) with teal accent headers.
- **Right zone (Agent)**: Subtle amber tint (`agent_zone_bg`) with amber accent headers.
- **Center zone (Interaction)**: Neutral white/gray — the bridge between the two realities.

#### Color-Coded Data Flow Arrows
Inside the `LoopVisualizer`, use colored arrows:
- **Teal arrows** (Environment → Agent): Context flowing to the agent, reward flowing back.
- **Amber arrows** (Agent → Environment): Arm selection sent to the world.

#### The "Hidden Truth" Toggle
In the Environment zone, a collapsible panel labeled "Hidden Ground Truth 🔒" shows the true arm probabilities. When collapsed (default in Lesson mode), a lock icon reminds users "the agent can't see this." When expanded (Sandbox mode), editable sliders let users tweak the hidden truth and observe how the agent adapts.

#### Knowledge State Visualization
In the Agent zone, the knowledge table displays:
- **Estimated mean reward** per arm (what the agent believes)
- **Confidence interval** (for UCB: upper confidence bound)
- **Pull count** (how many times tried)
- **Uncertainty indicator** (gradient bar: narrow = confident, wide = uncertain)

The key visual cue: The Agent zone's numbers are **approximations** (displayed with ~ prefix, e.g., "~0.72") while the Environment zone's numbers are **ground truth** (exact, e.g., "0.737").

---

## 11. Interaction Loop Visualization

### The 4-Phase Animated Bridge

The `LoopVisualizer` component in the center zone visually narrates the interaction cycle:

```
Phase 0: IDLE
┌──────────────┐                    ┌──────────────┐
│ ENVIRONMENT  │                    │    AGENT     │
│  (waiting)   │                    │  (waiting)   │
└──────────────┘                    └──────────────┘

Phase 1: CONTEXT GENERATED (teal pulse in Environment zone)
┌──────────────┐                    ┌──────────────┐
│ ENVIRONMENT  │  ──Context────▶   │    AGENT     │
│ Time: 5pm    │  (teal arrow)     │  Receiving   │
│ Rain: Yes    │                    │  context...  │
└──────────────┘                    └──────────────┘
Animation: ContextDisplay slides new values in from left.

Phase 2: ARM SELECTED (amber glow on selected arm card)
┌──────────────┐                    ┌──────────────┐
│ ENVIRONMENT  │  ◀──Arm "Priority"─│    AGENT     │
│  (waiting)   │  (amber arrow)    │  Selected:   │
│              │                    │  ▸ Priority  │
└──────────────┘                    └──────────────┘
Animation: Selected arm card scales up briefly (105% → 100%),
          amber border glow fades in.

Phase 3: REWARD RECEIVED (green/red pulse feedback)
┌──────────────┐                    ┌──────────────┐
│ ENVIRONMENT  │  ──Reward: +1──▶  │    AGENT     │
│ Actual: 0.72 │  (green arrow)    │  Received:   │
│ Result: WIN  │                    │  ✓ Success!  │
└──────────────┘                    └──────────────┘
Animation: RewardFeedback component shows:
  - Success: Green circular pulse expanding from center
  - Failure: Red shake on the arm card
  - Duration: 600ms, then fades

Phase 4: KNOWLEDGE UPDATED (Agent zone highlights changed values)
┌──────────────┐                    ┌──────────────┐
│ ENVIRONMENT  │                    │    AGENT     │
│  (waiting)   │                    │  Updated:    │
│              │                    │  Priority    │
│              │                    │  ~0.68→~0.72 │
│              │                    │  ↑ Highlight │
└──────────────┘                    └──────────────┘
Animation: Changed cells in KnowledgeTable flash amber for 400ms,
          then settle to new values.
```

### Implementation Approach

Use `asyncio.sleep()` for timed phase transitions within the `SimulationController.step_once()` method. The animation is a state machine:

```python
class InteractionPhase(Enum):
    IDLE = 0
    CONTEXT_GENERATED = 1
    ARM_SELECTED = 2
    REWARD_RECEIVED = 3
    KNOWLEDGE_UPDATED = 4

# In SimulationController:
async def step_once(self):
    # Phase 1
    self._set_phase(InteractionPhase.CONTEXT_GENERATED)
    self._emit(Event.CONTEXT_GENERATED, context)
    self._refresh_ui()
    await asyncio.sleep(0.3)

    # Phase 2
    result = self.simulator.step()  # Actually runs the simulation
    self._set_phase(InteractionPhase.ARM_SELECTED)
    self._emit(Event.ARM_SELECTED, result.chosen_arm)
    self._refresh_ui()
    await asyncio.sleep(0.4)

    # Phase 3
    self._set_phase(InteractionPhase.REWARD_RECEIVED)
    self._emit(Event.REWARD_RECEIVED, result.reward)
    self._refresh_ui()
    await asyncio.sleep(0.6)  # Longer pause for feedback absorption

    # Phase 4
    self._set_phase(InteractionPhase.KNOWLEDGE_UPDATED)
    self._emit(Event.KNOWLEDGE_UPDATED, policy_state)
    self._refresh_ui()
    await asyncio.sleep(0.3)

    self._set_phase(InteractionPhase.IDLE)
    self._refresh_ui()
```

For "Run N Steps" (fast mode), skip the animation delays and jump directly to the final state after N steps.

### Speed Control

| Mode | Behavior |
|---|---|
| **Step-by-Step** | Full 4-phase animation with delays (default speed: 1.6s per step) |
| **Play (Continuous)** | Reduced delays (0.1s per phase, total ~0.4s per step) |
| **Run N Steps** | No animation — jump to state after N steps, animate only the chart updates |
| **Speed Slider** | Multiplier from 0.25× to 4× applied to all delays |

---

## 12. Algorithm Curriculum & Progression

### Phase 1: Context-Free Bandits (Build Intuition)

| Lesson | Algorithm | Real-World Metaphor | Key Concept |
|---|---|---|---|
| 1.1 | Random | "Flipping a coin to choose a treatment" | Exploration baseline |
| 1.2 | Epsilon-Greedy | "Mostly stick with what works, occasionally try something new" | Exploration vs Exploitation trade-off |
| 1.3 | UCB1 | "Be optimistic about things you haven't tried enough" | Optimism in the face of uncertainty |
| 1.4 | Thompson Sampling | "If I think Treatment A works 70% of the time, I'll pick it 70% of the time" | Probability matching |

**Visual emphasis:** No context features — just arms with fixed, hidden success rates. The agent learns purely from trial and error.

### Phase 2: Introducing Context (The Aha Moment)

| Lesson | Algorithm | Real-World Metaphor | Key Concept |
|---|---|---|---|
| 2.1 | Why Context Matters | "A raincoat is great when it rains, useless when it's sunny" | Context changes what's optimal |
| 2.2 | LinUCB (Linear UCB) | "This patient is elderly with high blood pressure — Treatment B is probably better" | Linear relationship between features and rewards |
| 2.3 | LinTS (Linear Thompson) | "I'm 80% sure this user prefers action movies based on their history" | Contextual probability matching |
| 2.4 | Comparison Mode | Side-by-side: Epsilon-Greedy vs LinUCB on the same world | Context-free vs contextual performance gap |

**Visual transition:** When the user moves from Phase 1 to Phase 2, the `ContextDisplay` component transforms from "No context available" to showing rich feature vectors. The `KnowledgeHeatmap` chart appears, showing how the agent's understanding now depends on context.

### Lesson Progression UI

```
┌──────────────────────────────────────────────┐
│  Lesson: "The Optimistic Doctor" (UCB1)      │
│  ● Stage 1 ○ Stage 2 ○ Stage 3 ○ Stage 4 ○  │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │ 🎯 Current Objective:                   │  │
│  │ "Run 10 steps and observe which arm     │  │
│  │  the algorithm explores most."          │  │
│  │                                        │  │
│  │ ✓ Run 10 steps (10/10)                 │  │
│  │ ○ Identify the most-pulled arm          │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  [◀ Previous Lesson]    [Next Lesson ▶]      │
└──────────────────────────────────────────────┘
```

---

## 13. Wireframe Descriptions

### Wireframe 1: Home Screen
- Hero area with large COBA logo and tagline: "Learn how machines make decisions under uncertainty"
- Two large cards:
  - "Start Learning" → navigates to Lesson 1.1 (guided path)
  - "Free Play" → navigates to Arena (unguided exploration)
- Dark mode toggle in top-right corner
- Footer: "Built with Flet · 17 algorithms · 7 real-world scenarios"

### Wireframe 2: Lesson Mode (Phase 1 — Epsilon-Greedy, Rural Clinic)
- **Top bar**: COBA logo, lesson title, dark mode toggle, step counter (3/100)
- **Left zone (Teal)**: "Rural Clinic" world card with doctor icon. ContextDisplay shows "No context features in Phase 1." HiddenTruthPanel is locked (padlock icon) with tooltip "The doctor doesn't know the true treatment effectiveness — she must learn by trying."
- **Center zone**: LoopVisualizer showing Phases 2-4 (arm selected, reward received, knowledge updated). Three ArmCards: "Standard Care" (selected, amber glow), "Targeted Follow-up", "Remote Monitoring". RewardFeedback: green ✓ pulse. StepControls: [Step ▸] [▶ Play] [↺ Reset].
- **Right zone (Amber)**: KnowledgeTable showing running means: Standard Care ~0.72, Follow-up ~0.45, Remote ~0.38. PullCounter: Standard: 5 pulls, Follow-up: 2, Remote: 1. PolicyStateCard: "Epsilon: 0.1 (10% exploration)"
- **Bottom zone**: RegretChart (accumulating over 3 steps so far) and ArmHistogram. Both collapsed by default in Lesson mode — expandable via chevron.
- **Overlay**: LessonObjectiveCard at bottom-center: "Stage 1/4: Run 5 steps and observe."

### Wireframe 3: Arena Mode (Phase 2 — LinUCB, RidePilot)
- **Left zone**: World selector dropdown (RidePilot selected). ContextDisplay shows: Time: 5pm, Weather: Rain, Demand: High, Surge: 1.8×. HiddenTruthPanel is expanded (no lock): three sliders for Priority/Standard/Pool Match success probabilities.
- **Center zone**: LoopVisualizer with all 4 phases active. Four ArmCards for RidePilot world. RewardFeedback active.
- **Right zone**: KnowledgeTable with context-dependent estimates. KnowledgeHeatmap visible (2×3 grid of arm × time-of-day buckets). UncertaintyDisplay showing confidence bounds. PolicyStateCard: "LinUCB · alpha=1.0"
- **Bottom zone**: Three charts: RegretChart (animated, 47 data points), ArmHistogram (animated), RewardTimeline (sparkline showing recent reward streak).
- **Top controls**: Speed slider, Run-N input field with "Run 50" button, World/Policy selectors.

### Wireframe 4: Sandbox Mode (Editable World)
- Same as Arena but the HiddenTruthPanel is the primary focus.
- Each arm has a probability slider with real-time preview: "Priority Dispatch: ████████░░ 72%" with a small inline sparkline showing how changing this affects the agent's performance.
- "Generate Custom World" button opens a dialog for creating new feature/arm definitions.

### Wireframe 5: Comparison Mode (Epsilon-Greedy vs LinUCB vs Thompson)
- Top: Multi-select dropdown for policies (2-3 max).
- Center: Overlaid regret chart with three colored lines + legend. Arm distribution grouped bar chart (side-by-side bars for each policy).
- Bottom: Metrics table showing cumulative reward, final regret, best-arm identification rate for each policy.
- "Run Comparison" button runs all policies for N steps and shows results.

---

## 14. Implementation Roadmap

### Phase 1: Foundation — Theme & Layout (Week 1)
**Goal:** Dark/light mode works, new folder structure exists, SplitWorkspaceLayout renders with placeholder content.

1. **Create `src/web/theme/` module**
   - `colors.py`: `ColorTokens` dataclass, `LIGHT_TOKENS`, `DARK_TOKENS`
   - `themes.py`: `ft.Theme` construction from tokens
   - `theme_manager.py`: `ThemeManager.apply_theme(page, mode)`
   - `typography.py`, `spacing.py`: Typographic scale and spacing constants

2. **Create `src/assets/` directory**
   - Add SVG icons for: environment, agent, step, play, reset, sun, moon
   - Add 7 world-themed illustrations (simple SVG line art)

3. **Create `src/web/layouts/` module**
   - `base.py`: `BaseLayout` protocol
   - `split_workspace.py`: Render the 3-zone + bottom-charts layout
   - Migrate existing `_build_three_pane_body()` logic into `split_workspace.py`, refactored to use color tokens

4. **Create `src/web/components/shared/` module**
   - `section_header.py`: Zone header with icon + title + accent color strip
   - `metric_badge.py`: Value + label compact display
   - `empty_state.py`: "Nothing to show yet" placeholder

5. **Refactor `main.py` → `app.py`**
   - Extract `AppShell` class from `main.py`
   - Wire up `ThemeManager` to `page.theme_mode` change handler
   - `main.py` becomes thin: just `ft.app(target=app.main)`

**Commit:** `feat(web): add theme system, split-workspace layout, and shared components`

---

### Phase 2: Environment & Agent Zones (Week 2)
**Goal:** Left and right zones render real data from the simulator.

1. **Create `src/web/components/environment/` module**
   - `world_card.py`: World name, description, icon — reads from `WorldConfig`
   - `context_display.py`: Renders feature vector as key-value cards with teal accent
   - `hidden_truth_panel.py`: Collapsible panel with probability sliders per arm

2. **Create `src/web/components/agent/` module**
   - `knowledge_table.py`: DataTable of arms × mean reward × pulls, amber accent
   - `pull_counter.py`: Compact horizontal bar display of arm selection counts
   - `uncertainty_display.py`: Confidence interval visualization (for UCB)
   - `policy_state_card.py`: Algorithm-specific info (epsilon value, alpha, etc.)

3. **Create `src/web/state/` module**
   - `app_state.py`: Central `AppState` dataclass
   - `simulation_controller.py`: Wraps `DiscreteSimulator`, exposes `step_once()`, `run_n()`, `reset()`
   - `interaction_state.py`: `InteractionPhase` enum + phase tracking
   - `event_bus.py`: Simple pub/sub event system

4. **Integrate into SplitWorkspaceLayout**
   - Replace placeholder content with actual environment/agent components
   - Wire components to `AppState` via `page.session`

**Commit:** `feat(web): add environment and agent zone components with state management`

---

### Phase 3: Interaction Loop & Controls (Week 3)
**Goal:** The center zone shows animated 4-phase interaction. Controls are fully functional.

1. **Create `src/web/components/interaction/` module**
   - `loop_visualizer.py`: Animated bridge with directional arrows and phase indicators
   - `arm_cards.py`: Arm selection cards with glow animation on select
   - `reward_feedback.py`: Green/red pulse overlay for success/failure
   - `step_indicator.py`: Progress bar showing current step / horizon

2. **Create `src/web/components/controls/` module**
   - `step_controls.py`: Step, Play/Pause, Reset, Run-N buttons
   - `speed_slider.py`: Animation speed control
   - `probability_sliders.py`: Sliders for hidden ground-truth probabilities
   - `world_selector.py`, `policy_selector.py`: Dropdowns

3. **Implement animation controller in `SimulationController`**
   - Phase-by-phase stepping with `asyncio.sleep()` delays
   - Speed multiplier from slider
   - Run-N fast mode (skip animations)

4. **Create `src/web/components/controls/theme_toggle.py`**
   - Sun/moon icon button, toggles `page.theme_mode`

**Commit:** `feat(web): add animated interaction loop and full control panel`

---

### Phase 4: Charts & Data Visualization (Week 4)
**Goal:** Real-time animated charts render in the bottom zone.

1. **Create `src/web/components/charts/` module**
   - `chart_theme.py`: Theme-aware chart styling — reads `ColorTokens`
   - `regret_chart.py`: Cumulative regret `ft.LineChart` with animated updates
   - `arm_histogram.py`: Arm selection `ft.BarChart` with animated bar growth
   - `reward_timeline.py`: Per-step reward sparkline
   - `knowledge_heatmap.py`: Arm × Context grid using `ft.Container` cells

2. **Integrate charts into `SplitWorkspaceLayout` bottom zone**
   - Collapsible container with expand/collapse chevron
   - Charts update via `EventBus` subscription to `STEP_COMPLETED`

3. **Performance optimization**
   - Cap data points at 100 for time-series charts
   - Use `chart.update()` for incremental updates instead of rebuilds

**Commit:** `feat(web): add real-time animated charts with theme-aware styling`

---

### Phase 5: Lesson System & Narrative (Week 5)
**Goal:** Guided lesson progression with visual storytelling and real-world framing.

1. **Enhance `src/web/curriculum/`**
   - Add narrative banners per lesson stage (e.g., "You are a doctor at a rural clinic...")
   - Per-lesson illustration selection
   - Lock/unlock HiddenTruthPanel based on lesson stage

2. **Create `LessonView` as a variant of `SplitWorkspaceLayout`**
   - `LessonProgressBar` component (horizontal step indicator for lesson stages)
   - `NarrativeBanner` component (contextual storytelling header)
   - `LessonObjectiveCard` overlay component

3. **Implement lesson state machine**
   - Stage completion detection (e.g., "has run 10 steps")
   - Auto-advance with confirmation
   - Unlock new controls/components as user progresses

4. **Define Lesson 1.1-1.4 (Context-Free) and 2.1-2.4 (Contextual)**
   - Each lesson has: world assignment, policy, initial parameters, narrative text, objectives
   - Phase 1→2 transition: ContextDisplay changes from "No context" to real features, KnowledgeHeatmap appears

**Commit:** `feat(web): add guided lesson system with narrative storytelling`

---

### Phase 6: Remaining Routes & Polish (Week 6)
**Goal:** Sandbox, Comparison, and Home routes are complete. Visual polish applied.

1. **Home route** (`/`)
   - Hero section with animated COBA logo
   - Two navigation cards with illustrations
   - Quick feature highlights (3 cards: "17 Algorithms", "7 Real Worlds", "Real-Time Charts")

2. **Sandbox route** (`/sandbox`)
   - Full Arena layout with HiddenTruthPanel as primary focus
   - "Generate Custom World" dialog
   - Real-time preview of how probability changes affect agent performance

3. **Comparison route** (`/comparison`)
   - Multi-policy selector
   - Overlaid regret chart (colored lines)
   - Side-by-side arm distribution grouped bar chart
   - Summary metrics table

4. **Visual polish**
   - Consistent spacing using `spacing.py` constants
   - Typography scale applied everywhere
   - Hover states on all interactive elements
   - Loading skeletons for async operations
   - Smooth page transitions

**Commit:** `feat(web): complete sandbox, comparison, and home routes with visual polish`

---

### Phase 7: Responsive Layout & Mobile (Week 7)
**Goal:** Layout adapts gracefully to different screen sizes.

1. **Responsive `SplitWorkspaceLayout`**
   - **Wide (≥1200px):** 3-column horizontal + bottom charts
   - **Medium (768-1199px):** Environment + Agent stacked, Interaction between them, charts below
   - **Narrow (<768px):** Single-column vertical scroll with all zones stacked

2. **Use `page.width` and `page.on_resize`**
   - Switch layout mode based on breakpoints
   - Chart sizes scale proportionally
   - NavigationRail → BottomNavigationBar on narrow screens

3. **Touch-friendly controls**
   - Larger tap targets on mobile
   - Slider thumb size increased for touch

**Commit:** `feat(web): add responsive layout with mobile support`

---

## 15. Verification & Testing Strategy

### Unit Tests (`tests/web/`)

| Module | Test File | What to Test |
|---|---|---|
| `theme/colors.py` | `test_theme/test_colors.py` | Token immutability, light/dark contrast ratios ≥ 4.5:1 |
| `theme/theme_manager.py` | `test_theme/test_theme_manager.py` | Apply/switch modes, token retrieval from session |
| `state/app_state.py` | `test_state/test_app_state.py` | State transitions, immutability of frozen fields |
| `state/simulation_controller.py` | `test_state/test_simulation_controller.py` | Step sequencing, phase transitions, event emission |
| `state/event_bus.py` | `test_state/test_event_bus.py` | Subscribe, emit, unsubscribe, error isolation |
| `layouts/split_workspace.py` | `test_layouts/test_split_workspace.py` | Zone ratios, collapse behavior, responsive breakpoints |
| `components/charts/*` | `test_components/test_charts.py` | Data point appending, animation flag, theme color application |
| `components/controls/*` | `test_components/test_controls.py` | Button callbacks, slider ranges, Run-N input validation |
| `components/interaction/*` | `test_components/test_interaction.py` | Phase display, arm card glow, reward feedback state |

### Integration Tests

| Scenario | What to Verify |
|---|---|
| Theme toggle → All zones update | Zone background colors, chart colors, text colors all switch |
| Step button → Full interaction cycle | All 4 phases display in sequence, correct data flows to agent zone |
| Run 50 → Charts animate | Chart data points accumulate correctly, no visual glitches |
| Probability slider change → Agent adapts | After slider change and 20+ steps, agent's mean estimates shift toward new truth |
| Lesson stage completion → Auto-advance | Stage counter increments, new objective appears, controls unlock |
| World change → Simulator resets | All state cleared, new world context displays, new arm cards appear |

### Visual Regression
- Screenshot comparisons for each route in both light and dark mode.
- Verify consistent spacing, font sizes, and alignment across all components.

### Performance
- Frame timing: chart updates under 16ms (60fps target)
- Step animation: no jank during phase transitions
- Run-N (1000 steps): completes without blocking the UI thread

---

## Appendix: Dependency Changes

### `pyproject.toml` Additions
```toml
[project]
dependencies = [
    # ... existing deps ...
    "flet>=0.25.0",     # Already present — ensure latest for native chart animation support
]

[project.optional-dependencies]
dev = [
    # ... existing dev deps ...
    "pytest-asyncio",   # For async simulation controller tests
]
```

### No New External Dependencies
All charting uses Flet's built-in `ft.LineChart`, `ft.BarChart`, and `ft.PieChart`. No Plotly, Matplotlib, or third-party chart libraries needed.

---

## Summary of Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Layout | Split-Workspace (Option C) | Best narrative flow, natural Environment/Agent separation, full-width charts |
| Chart library | Flet native charts | Smooth animation, theme inheritance, no extra dependencies |
| State management | Centralized AppState + EventBus | Predictable, testable, avoids global mutable state |
| Theme system | Semantic color tokens + Flet Theme | Dark/light mode with zero per-component code changes |
| Animation | Async phase-by-phase stepping | Beginner-friendly pacing, skippable for fast mode |
| Component pattern | Keep VM+P, add EventBus subscription | Preserves existing clean pattern, adds reactivity |
| Responsive | Breakpoint-based layout switching | 3-column → 2-column → single-column stack |
| Lesson progression | Stage-based with auto-advance | Guided path prevents overwhelm, unlocks complexity gradually |
