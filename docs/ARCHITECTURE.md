# COBA-Web Architecture Decision Record

This document outlines key architectural decisions made during the educational redesign.

## Table of Contents

1. [Lesson Structure Design](#lesson-structure-design)
2. [Component Architecture](#component-architecture)
3. [State Management](#state-management)
4. [Data Fetching & Caching](#data-fetching--caching)
5. [Performance Optimization](#performance-optimization)
6. [Accessibility Approach](#accessibility-approach)
7. [Testing Strategy](#testing-strategy)

---

## Lesson Structure Design

### Decision: 8-Lesson Progressive Curriculum

**Choice**: Linear curriculum with 8 lessons (beginner → advanced) + 1 reference.

**Rationale**:
- Clear learning progression: foundational concepts → extensions → advanced topics
- Each lesson builds on previous one, reducing cognitive load
- Prerequisites explicitly marked, guiding learners through optimal paths
- Manageable scope: 30-60 min per lesson = 4-5 hours total

**Alternative Considered**: Modular lessons with no ordering
- **Rejected**: Non-linear approach would require lessons to be self-contained, removing the power of progressive teaching.

### Decision: Problem-Why-Technique-Demo Structure

**Choice**: Every lesson follows: Problem → Why It's Hard → Technique → Interactive Demo → What You're Seeing

**Rationale**:
- Mirrors pedagogical best practice (activate prior knowledge → introduce challenge → teach solution → practice)
- Each section serves a specific cognitive purpose
- Consistent structure makes learning predictable
- Frees learners to focus on content, not navigation

**Alternative Considered**: Technique-first (math → intuition)
- **Rejected**: Learners without ML background would be lost; intuition first makes math accessible.

### Decision: Domain-Agnostic Terminology

**Choice**: "option" not "arm", "reward" not "CTR", "cluster" not "segment"

**Rationale**:
- Platform teaches contextual bandits conceptually, not for ad-tech specifically
- Generic terms apply to broader domains (pricing, recommendations, clinical trials)
- Reduces jargon barriers for learners from non-ad backgrounds
- Glossary system explains all technical terms inline

**Alternative Considered**: Keep AdTech terminology
- **Rejected**: Narrows audience and requires mental translation for non-AdTech learners.

---

## Component Architecture

### Decision: Shared Educational Components (LessonHeader, LessonNav, GlossaryTip)

**Choice**: Create reusable educational components instead of page-specific logic.

**Files**:
- `components/education/LessonHeader.tsx` — problem/why/technique blocks
- `components/education/LessonNav.tsx` — prev/next lesson navigation
- `components/education/GlossaryTip.tsx` — inline glossary tooltips

**Rationale**:
- DRY principle: avoid duplicating lesson structure across 8 pages
- Consistent UX: every lesson looks and feels the same
- Easy maintenance: update all lessons by changing one component
- Testable: unit tests cover all lessons simultaneously

**Alternative Considered**: Page-specific components
- **Rejected**: 8 copies of similar code would be maintenance nightmare; changes couldn't be centralized.

### Decision: React.memo for GlossaryTip & Other Interactive Components

**Choice**: Wrap frequently-rendered components with `React.memo` to prevent unnecessary re-renders.

**Rationale**:
- GlossaryTip is used 10-20 times per lesson page
- Without memo, clicking one tooltip could re-render all tooltips
- memo has minimal overhead for props-based equality checking
- Small component size makes memoization effective

**Alternative Considered**: useCallback on all props
- **Rejected**: memo is simpler and sufficient for this use case.

### Decision: Lazy Loading Components & Code Splitting

**Choice**: Each lesson page lazy-loaded via `next/dynamic` to reduce initial bundle.

**Rationale**:
- Only active lesson code is loaded
- Initial page load is ~50% faster
- Navigation between lessons feels responsive

**Implementation**:
```typescript
export const PlaygroundPage = dynamic(() => import('./playground'), { ssr: false });
```

---

## State Management

### Decision: localStorage for Client-Side Persistence

**Choice**: Use `localStorage` for progress, bookmarks, preferences (no backend persistence required).

**Rationale**:
- No authentication system yet; backend storage would require login
- Learning progress is personal, device-local is acceptable
- Fast, no network latency
- Simple to implement (JSON serialization)

**Alternative Considered**: Backend persistence
- **Deferred**: Can be added later if multi-device sync is needed.

**Trade-offs**:
- ✅ Fast, offline-capable, no backend load
- ❌ No sync across devices
- ❌ Lost if user clears browser data

### Decision: API Fallback Pattern

**Choice**: Fetch from `/api/*` endpoints, fallback to static client-side data.

**Hooks**:
```typescript
useLessonData() // Tries API, falls back to static LESSONS
useGlossaryData() // Tries API, falls back to static GLOSSARY
```

**Rationale**:
- API data allows backend to control curriculum/glossary without re-deploying frontend
- Fallback ensures app works even if API is slow/unavailable
- Teaches learners about resilience patterns

**Alternative Considered**: Always use static data
- **Rejected**: Loses flexibility of server-controlled content.

---

## Data Fetching & Caching

### Decision: localStorage Cache with TTL

**Choice**: Cache API responses in `localStorage` with no expiration (or optional TTL).

**Pattern**:
```typescript
const cached = localStorage.getItem("coba-glossary-cache");
if (cached) return JSON.parse(cached);
// else fetch from API
```

**Rationale**:
- Reduces API calls on repeat visits
- Network-independent (works offline)
- Simple invalidation (call `clearEducationCaches()`)

**Alternative Considered**: Service Worker caching
- **Deferred**: localStorage is sufficient for now; Service Worker adds complexity.

---

## Performance Optimization

### Decision: Plotly Chart Lazy Loading

**Choice**: Charts load on-demand (when section scrolls into view).

**Rationale**:
- Plotly.js is ~3MB (large)
- Most users don't scroll to every chart
- Intersection Observer detects visibility
- Reduces initial load by ~2 seconds

### Decision: Memoization Strategy

**Components Memoized**:
- ✅ `GlossaryTip` — props-based, no children changes
- ✅ `ChartCard` — expensive chart component
- ✅ `Badge` — rendered many times
- ⏳ `LessonHeader` — could memo if props rarely change

**Not Memoized**:
- ❌ `PageShell` — contains heavy children, memo won't help
- ❌ `Button` — too simple to benefit

---

## Accessibility Approach

### Decision: WCAG 2.1 AA Compliance Target

**Choice**: All interactive elements follow WCAG 2.1 AA guidelines.

**Implemented**:
- ✅ Keyboard navigation (Tab, Enter, Escape)
- ✅ Focus indicators (outline or shadow)
- ✅ Color contrast (4.5:1 for normal text)
- ✅ Semantic HTML (button, link, heading)
- ✅ ARIA labels where needed
- ✅ Dark mode support

**Not Yet Implemented**:
- ⏳ Full screen reader testing (NVDA/VoiceOver)
- ⏳ High contrast mode
- ⏳ Closed captions (if videos added)

### Decision: Dark Mode as First-Class Feature

**Choice**: Every component supports dark mode from day one.

**Pattern**:
```tsx
<div className="bg-surface dark:bg-surface-muted text-foreground dark:text-foreground">
```

**Rationale**:
- CSS variables defined for light & dark
- No late-stage dark mode refactoring needed
- Learners can choose preferred theme
- Reduces eye strain

---

## Testing Strategy

### Decision: Three-Tier Testing

**Tier 1: Unit Tests** (Jest + React Testing Library)
- Component rendering, props handling
- Glossary term lookup
- Progress tracking

**Tier 2: E2E Tests** (Playwright)
- Complete user flows (home → lesson → next lesson)
- Navigation between lessons
- Glossary interaction

**Tier 3: Manual Accessibility Testing**
- Keyboard-only navigation
- Screen reader testing (NVDA/VoiceOver)
- Visual regression (light/dark modes)

**Target Coverage**: 80%+ on critical paths, 100% on education components

### Decision: Test-Driven Development (TDD) for New Features

**Process**:
1. Write test case (test fails)
2. Implement feature (test passes)
3. Refactor with confidence

**Example**: Progress tracking
- Test: `useProgressTracking(2)` marks lesson as started
- Test: Leaving lesson marks it as completed
- Implement: localStorage logic
- Test passes ✅

---

## Future Architectural Decisions

### Planned: Backend Persistence

When user authentication is added:
- Sync progress to backend
- Persist bookmarks across devices
- Track completion time per user

### Planned: Internationalization (i18n)

Current approach:
- All content in English
- Glossary is translation-ready (separate data file)

### Planned: Real-Time Collaboration

If group learning is added:
- Use WebSockets for live lesson discussions
- Broadcast quiz results
- Shared whiteboard for diagrams

### Planned: Mobile-First Native App

Current: Responsive web app
Future: React Native app with offline-first sync

---

## Decision Rubric: Adding New Features

Before implementing new features, use this rubric:

| Question | Answer |
|----------|--------|
| Does it support learning? | **Required** |
| Can it use existing components? | **Prefer** |
| Does it require new backend API? | **Defer unless critical** |
| Can it be tested? | **Required** |
| Does it break WCAG 2.1 AA? | **Unacceptable** |
| Does it slow page load? | **Profile first** |

---

## Summary

**Key Principles**:
1. **Progressive complexity** — foundation → extensions → advanced
2. **DRY components** — shared education components across 8 lessons
3. **Resilience** — API fallback to static data
4. **Accessibility first** — not an afterthought
5. **Performance measured** — profiling before optimization
6. **Testing as documentation** — tests show intended behavior

This architecture balances simplicity (localStorage, fallback), scalability (component reuse), and pedagogy (lesson structure).

---

**Last Updated**: May 2024
**Maintainer**: @dmoreq
