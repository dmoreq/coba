# COBA Web: System Architecture & Design Decisions

**Status:** Production-ready (v1.0)
**Last Updated:** 2026-05-18

This document outlines the architectural decisions and system design for COBA Web, an interactive educational platform for teaching contextual bandit algorithms.

## Quick Navigation

1. [System Overview](#system-overview)
2. [Frontend Architecture](#frontend-architecture)
3. [Backend Architecture](#backend-architecture)
4. [Component Design](#component-design)
5. [State Management](#state-management)
6. [Performance & Optimization](#performance--optimization)
7. [Testing Strategy](#testing-strategy)
8. [Key Design Decisions](#key-design-decisions)

---

## System Overview

### Tech Stack

**Frontend**: Next.js 16 + React 19 + TypeScript (strict) + TailwindCSS
**Backend**: FastAPI (Python 3.10+) + Pydantic
**Data**: In-memory session state (can add persistence layer)
**Testing**: Frontend (Vitest + React Testing Library), Backend (Pytest)

### Project Structure

```
coba/
├── web/
│   ├── frontend/              # Next.js 16 app
│   │   ├── app/               # Pages + layouts
│   │   ├── components/        # 50+ reusable components
│   │   ├── lib/
│   │   │   ├── api.ts         # Typed API client
│   │   │   ├── lessons.ts     # Lesson registry (17 configs)
│   │   │   └── hooks.ts       # Custom hooks (useSession, useSimulator)
│   │   └── tests/             # 103 unit tests
│   │
│   └── backend/               # FastAPI app
│       ├── app/
│       │   ├── models/        # Pydantic schemas
│       │   ├── routers/       # Endpoints (6 routes)
│       │   └── services/      # Business logic
│       └── tests/             # 63 tests (90% coverage)
│
└── docs/                       # Documentation
```

### Key Metrics

| Metric | Value |
|--------|-------|
| Frontend Tests | 103/103 ✅ |
| Backend Tests | 63/63 (90% coverage) ✅ |
| TypeScript Errors | 0 ✅ |
| Build Time | ~1.2s ✅ |
| Bundle Size (gzipped) | ~45KB |
| Components | 50+ |
| Lessons | 17 |
| Lines of Code | ~7,500 |

---

## Frontend Architecture

### Core Concepts

---

### Lesson Structure: Progressive Curriculum with 17 Algorithms

**Status:** Complete. All 17 lessons implemented and interactive.

**Lessons by Difficulty:**

**Beginner (3):** Epsilon-Greedy, UCB1, Thompson Sampling
**Intermediate (5):** LinUCB, LinTS, Logistic, Cluster Routing, LinUCB-Hybrid
**Advanced (5):** Neural Linear, Random Forest, GP-UCB, Softmax, Sliding-Window LinUCB
**Specialist (4):** Drift Detection, Offline Evaluation, CATS, Production Features

### Lesson UI Pattern: 2-Column LessonShell

All lessons use consistent layout:
- **Left Column:** Theory cards + interactive controls
- **Right Column:** Live visualizations (arm scores, rewards, regret)
- **Bottom:** Trace explorer (detailed state inspection)
- **Responsive:** Stack to single column on mobile

**Benefits:**
- Theory and practice side-by-side (visual learning)
- Consistent UX across all 17 lessons
- Controls immediately show results on charts
- Trace panel allows deep inspection of algorithm decisions

### Terminology & Pedagogy

Platform uses domain-agnostic language:
- "arm" → specific choice/action
- "reward" → outcome value
- "context" → input features
- "policy" → decision algorithm

Glossary integrated into each lesson for inline term explanations.

---

## Component Design

### Component Hierarchy

**UI Primitives (7):** Button, Card, Slider, Toggle, Badge, Kbd, Tooltip
**Layout Components (4):** TopBar, Sidebar, LessonLayout, MobileNav
**Chart Components (8):** ArmBar, RewardChart, RegretChart, PullHistogram, BetaDistribution, DriftTimeline, TreeDiagram, ConfidenceEllipse
**Educational Components (15+):** TheoryCard, TracePanel, FlowAnnotation, ControlPanel, etc.
**Lesson Implementations (17):** One for each algorithm

### Reusability Strategy

**Shared Educational Components:**
- `LessonShell` — 2-column layout for all lessons
- `TheoryCard` — Collapsible algorithm explanation
- `TracePanel` — Detailed trace inspection
- `Chart*` — Live visualization (rewards, regret, scores, distributions)

**Benefits:**
- Single source of truth for lesson layout
- Changes propagate to all 17 lessons instantly
- Consistent testing across all lessons
- Rapid lesson development (just configure data source)

### Performance Optimization

**Code Splitting:**
- Each lesson lazy-loaded via `next/dynamic` with `ssr: false`
- Only active lesson JS is loaded
- ~40% reduction in initial bundle

**Memoization:**
- `React.memo` on frequently-rendered components (Badge, Button, Chart)
- `useCallback` for event handlers in interactive controls
- `useMemo` for expensive computations (trace processing)

**Rendering:**
- Recharts with optimized tick rendering
- Virtualized lists for large traces (if needed)
- Intersection Observer for lazy chart loading

---

## State Management

### Client-Side (localStorage)

**Persisted State:**
- Progress: which lessons marked complete
- Theme preference: light/dark mode
- UI state: sidebar collapsed, trace expanded, etc.
- User preferences: speed, volume, etc.

**Implementation:** React hooks + custom `useLocalStorage` hook

**Trade-offs:**
- ✅ Fast, no network latency
- ✅ Works offline
- ✅ Simple to clear (manual reset)
- ❌ No cross-device sync
- ❌ Lost if user clears browser cache

**Future:** Backend persistence can be added with auth layer.

### Server-Side (FastAPI)

**Stateful Sessions:**
- `POST /sessions` — Create bandit session, returns session_id
- Session contains: bandit state, arm stats, trace history
- Session expires after 24 hours (configurable)
- Stored in-memory (or Redis for scaling)

**REST Pattern:**
```
POST /sessions                      # Create
GET /sessions/{id}                  # Read stats
POST /sessions/{id}/step            # Action: step bandit
POST /sessions/{id}/update          # Action: record reward
DELETE /sessions/{id}               # Cleanup
```

### Data Synchronization

**Frontend ↔ Backend Flow:**
1. User clicks "Run" on lesson
2. Frontend creates session: `POST /sessions`
3. Backend initializes bandit with lesson-specific config
4. Frontend loops: `step` → wait → `update` with reward
5. All state lives on backend; frontend is a view layer

**State Separation:**
- **Backend owns:** Bandit logic, arm stats, trace history
- **Frontend owns:** UI state (paused, speed, theme), progress tracking
- No redundant state between tiers

---

## Backend Architecture

### Session Management (FastAPI)

**BanditSessionService** (core session lifecycle):
- `create_session(policy_type, config)` — Initialize with bandit
- `get_session(id)` — Fetch current state
- `update_session(id, arm, reward)` — Record feedback
- `delete_session(id)` — Cleanup

**TraceBuilder** (immutable trace history):
- Pure function: `(state, action, reward) → trace_entry`
- Appends to session's trace list
- Enables replay and inspection

**SimulatorService** (reward generation):
- Lesson-specific reward functions (17 policies)
- Returns immediate feedback for UI
- Decoupled from backend policy (allows exploration)

### REST Endpoints (6)

**Sessions:**
```
POST   /sessions                   # Create session
GET    /sessions/{id}              # Get stats & current state
POST   /sessions/{id}/step         # Step bandit (get decision)
POST   /sessions/{id}/update       # Update with reward
DELETE /sessions/{id}              # Delete session
```

**Lesson Extras:**
```
POST   /sessions/{id}/arm          # Add/remove arm
POST   /sessions/{id}/drift        # Inject drift
POST   /sessions/{id}/offline-eval # Offline evaluation
GET    /sessions/{id}/cluster-map  # Cluster visualization
GET    /sessions/{id}/leaf-scores  # CATS decision tree scores
```

### Validation & Error Handling

**Pydantic Models:**
- All inputs validated with Pydantic schemas
- Type-safe request/response contracts
- Automatic OpenAPI docs at `/docs`

**Error Responses:**
- 400: Invalid input (malformed context, invalid reward range)
- 404: Session not found
- 422: Validation error (snake_case conversion, value ranges)
- 500: Server error (logs include trace for debugging)

## Performance & Optimization

### Frontend

**Build:**
- Turbopack for ~1.2s builds
- SWC for TypeScript transpilation
- Tree-shaking removes unused code

**Bundle:**
- ~45KB gzipped (Next.js app + Recharts)
- Chunk splitting: each lesson <50KB
- Lazy loading reduces initial JS load

**Runtime:**
- React 19 server components where possible
- Memoization of expensive components
- Debounced slider updates (trace processing)

### Backend

**Scaling:**
- FastAPI async/await for concurrency
- Session storage: in-memory (can swap for Redis)
- CORS enabled for cross-origin requests

**Optimization:**
- Sherman-Morrison for O(d²) ridge regression updates
- K-means clustering pre-computed
- Trace stored as list (not array, for JSON serialization)

---



---

## Accessibility

**WCAG 2.1 AA Compliance:**
- ✅ Keyboard navigation (Space/Arrows/Ctrl+N/P)
- ✅ Focus indicators on all interactive elements
- ✅ Color contrast ≫4.5:1
- ✅ Semantic HTML (button, link, heading, nav)
- ✅ ARIA labels on custom components
- ✅ Dark mode support

**Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| Space | Play / Pause |
| → / ← | Step forward / backward |
| 1/2/3 | Speed 1x / 10x / 100x |
| Ctrl+N | Next lesson |
| Ctrl+P | Previous lesson |
| ? | Help |

**Future:** Full screen reader testing (NVDA/VoiceOver)

### Dark Mode

All components support light/dark themes:
- CSS variables in `:root` and `.dark`
- TailwindCSS `dark:` utilities throughout
- Toggle in TopBar; persists to localStorage
- No flash on page load (blocking script in head)

---

## Testing Strategy

### Coverage

**Frontend:** 103 tests (Vitest + React Testing Library)
- Components: rendering, props, state transitions
- Hooks: useSession, useSimulator, useProgress
- Integration: lesson flow, navigation
- Target: 80%+ coverage on critical paths

**Backend:** 63 tests (Pytest, 90% coverage)
- Session lifecycle: create, step, update, delete
- Policies: all 17 algorithms tested
- Validation: Pydantic error handling
- Extras: arm management, drift, evaluation

**E2E (Manual):**
- Keyboard navigation (all shortcuts)
- Cross-browser (Chrome, Firefox, Safari)
- Mobile responsive (iOS, Android)
- Accessibility (screen reader, keyboard-only)

### Running Tests

**Frontend:**
```bash
cd web/frontend
npm test                   # Run all
npm test -- --coverage    # With coverage
```

**Backend:**
```bash
cd web/backend
pytest                     # Run all
pytest --cov=app          # With coverage (target: 90%+)
```

---

## Key Design Decisions

### 1. Lesson Registry (Single Source of Truth)

**Location:** `web/frontend/lib/lessons.ts`

```typescript
interface LessonConfig {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'specialist';
  policyType: PolicyType;
  initialConfig: any;  // Policy-specific settings
  theorySections: TheorySection[];
  chartTypes: ChartType[];
  // ...
}
```

**Benefit:** Adding a new lesson requires only adding a config object; all UI logic reuses LessonShell.

### 2. REST Session Model

**Why not WebSockets?**
- Simpler to implement and debug
- Educational (teaches REST state machine)
- Sufficient for interactive pace (100-500ms steps)
- Easier to scale (stateless backend)

**Trade-off:** Slightly more latency vs maximum simplicity

### 3. Client-Side Chart Rendering

**Why Recharts (not server-rendered)?**
- Instant interactivity (no server round-trip)
- Zooming, panning, tooltip on client
- Reduces backend compute load
- Works offline

**Trade-off:** ~45KB bundle size vs rich interactive experience

### 4. Theme Toggle (CSS Variables + localStorage)

**Why not context API?**
- CSS variables work even if JavaScript fails
- Instant toggle (no React re-render overhead)
- Survives page reload automatically
- Simple to extend (add more themes without code)

**Implementation:**
- `:root { --color-bg: white; }`
- `.dark { --color-bg: black; }`
- Blocking script in `<head>` prevents flash

---

## Future Roadmap

### Phase 2 (Planned)

- [ ] User accounts & progress sync to backend
- [ ] Internationalization (i18n) for multi-language support
- [ ] Mobile app (React Native) with offline-first sync
- [ ] Advanced analytics (time per lesson, policy comparison)
- [ ] Quiz system with scoring

### Phase 3 (Future)

- [ ] Real-time collaboration (multi-user sessions)
- [ ] Live instructor dashboard
- [ ] Video tutorials embedded
- [ ] Research data export (for educators)
- [ ] WebSocket upgrade for lower latency

---

## Summary

**Core Principles:**
1. **Progressive Pedagogy** — Beginner → Advanced, each lesson builds on prior
2. **Component Reusability** — LessonShell + shared charts serve all 17 lessons
3. **Clear Separation of Concerns** — Backend owns logic, frontend owns UI
4. **Accessibility First** — Keyboard nav, dark mode, semantic HTML from day 1
5. **Measured Performance** — Bundle optimized, tests prevent regressions
6. **Testing as Documentation** — Tests show intended behavior

This architecture balances **simplicity** (no auth, localStorage, REST) with **scalability** (component reuse, policy abstraction) and **pedagogy** (progressive complexity, consistency).

---

**Last Updated:** 2026-05-18
**Status:** Production-ready (v1.0)
**Maintainer:** COBA Team
