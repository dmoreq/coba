# COBA: Roadmap & Future Enhancements

**Current Status:** v0.2 — Flet redesign in progress
**Last Updated:** 2025-05-25

---

## Overview

Planned enhancements for the COBA educational platform, ordered by priority.

---

## Phase 2: Full Flet UI Integration (Current)

### 2.1 Component Tests
Add mock-page tests for all 6 `components/*.py` files (environment, agent, interaction, charts, shared, theme_toggle), `layouts/split_workspace.py`, `app.py`, and `theme/theme_manager.py`.

### 2.2 Selector Components
Extract world/policy/speed selectors from `main.py` into dedicated components in `components/controls/`.

### 2.3 Policy Base Classes
Extract `ContextFreePolicyBase`, `LinearContextualPolicyBase`, `BucketPolicyBase` to eliminate ~360 lines of duplication across the 14 policy wrappers.

### 2.4 Shared Factory
Extract `build_simulator()` from the 3 duplicated sites (`main.py`, `view_models.py`, `sandbox.py`) into `simulator_factory.py`.

### 2.5 Hidden Truth Panel
Add collapsible `HiddenTruthPanel` with probability sliders in the environment zone.

---

## Phase 3: User Experience (Next)

### 3.1 Lesson Narrative
Add narrative banners, lesson progress bars, and visual objective cards to the lesson route.

### 3.2 Sandbox Mode
Full world-editing UI (add/remove arms, adjust probabilities, generate custom worlds).

### 3.3 Comparison Mode
Side-by-side policy comparison with overlaid regret charts and metrics tables.

### 3.4 Animation
Reintroduce smooth phase transitions (context→arm→reward→learn) using `ft.animate_*` properties once Flet supports them.

---

## Phase 4: Polish & Scale

### 4.1 Responsive Layout
Breakpoint-based layout switching: 3-column → 2-column → single-column stack.

### 4.2 Internationalization
Extract content strings — initial targets: Vietnamese, Spanish.

### 4.3 Progress Persistence
Save lesson progress and preferences across sessions (currently `PreferencesStore` handles this, but lesson progress is in-memory only).

### 4.4 Keyboard Shortcuts
S=Step, Space=Play/Pause, R=Reset, D=Dark mode toggle, Arrow Left/Right=Navigate.
