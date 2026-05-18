# COBA Web: Roadmap & Future Enhancements

**Current Status:** v1.0 — Production-ready
**Last Updated:** 2026-05-18

---

## Overview

This roadmap outlines planned enhancements, improvements, and experimental features for COBA Web. Items are grouped by phase and priority.

---

## Phase 2: User Experience & Persistence (Q3 2026)

### 2.1 User Accounts & Progress Sync

**Goal:** Allow users to sign in and sync progress across devices.

**Implementation:**
- Add authentication (Auth0, Clerk, or Firebase)
- Move localStorage progress to backend database
- Track completion time, attempts per lesson
- Leaderboard (optional)

**Effort:** Medium (2-3 weeks)
**Priority:** Medium (nice-to-have, not essential)

### 2.2 Internationalization (i18n)

**Goal:** Support multiple languages (Vietnamese, Spanish, Mandarin initial targets).

**Current State:**
- English content in `web/frontend/lib/lessons.ts`
- Glossary already data-driven
- Algorithm descriptions as JSX (need extraction)

**Implementation:**
- Extract all content strings to i18n files
- Use `next-intl` middleware for routing (`/en/lesson/ucb1`, `/vi/lesson/ucb1`)
- Translate glossary, theory cards, UI labels
- Backend already language-agnostic (just numbers)

**Effort:** Medium (1-2 weeks per language)
**Priority:** Medium (expand audience globally)

### 2.3 Advanced Analytics Dashboard

**Goal:** Help educators understand how students learn.

**Metrics to Track:**
- Time spent per lesson
- Number of resets
- Parameter exploration patterns
- Most-paused lessons
- Comparison: which policies students choose

**Implementation:**
- Backend: log events (lesson_start, policy_changed, etc.)
- Frontend: page context + event tracking
- Dashboard: `/admin/analytics` with charts

**Effort:** Medium (1-2 weeks)
**Priority:** Low (research feature)

---

## Phase 3: Interactive Learning (Q4 2026)

### 3.1 Quiz System

**Goal:** Test understanding after each lesson.

**Types:**
- Multiple choice (conceptual)
- Parameter optimization (simulation-based)
- Math derivations (fill-in-the-blank)

**Implementation:**
- Lesson config includes `quizzes: Quiz[]`
- QuizPanel component
- Score tracking + hints system
- Backend validates answers

**Effort:** High (2-3 weeks)
**Priority:** Medium (improves learning outcomes)

### 3.2 Comparison Mode (A/B across lessons)

**Goal:** Let users run two policies side-by-side.

**UI:**
- Select lesson 1 (policy A)
- Select lesson 2 (policy B)
- Run both with identical contexts
- Compare rewards, regret, arm selection

**Implementation:**
- Two parallel sessions (backend)
- Split-screen layout (frontend)
- Diff visualization

**Effort:** Medium (1-2 weeks)
**Priority:** Low (nice-to-have)

### 3.3 Embedded Video Tutorials

**Goal:** Visual explanations for complex algorithms.

**Content:**
- One ~5min video per lesson
- Recorded screen captures + voiceover
- Hosted on Vercel (or S3)

**Implementation:**
- Video component in TheoryCard
- HLS streaming for mobile
- Subtitles/captions (auto-generated)

**Effort:** Medium (video production, encoding)
**Priority:** Low (optional enhancement)

---

## Phase 4: Native & Offline (2027)

### 4.1 React Native Mobile App

**Goal:** iOS/Android apps with offline-first sync.

**Tech Stack:**
- React Native (Expo)
- SQLite for offline persistence
- Redux for state sync

**Implementation:**
- Port lessons to React Native (most can reuse logic)
- Charts: use `react-native-svg` for Recharts compatibility
- Offline mode: cached lessons + local bandit state
- Sync: when online, push progress to backend

**Effort:** High (4-6 weeks)
**Priority:** Low (web-first approach sufficient)

### 4.2 Offline-First Web (Service Worker)

**Goal:** Full offline support for existing web app.

**Implementation:**
- Service Worker caches lesson configs + bundles
- IndexedDB for session state
- Background sync API for offline updates
- Works even without internet

**Effort:** Medium (1-2 weeks)
**Priority:** Medium (improves resilience)

---

## Phase 5: Collaboration & Research (2027+)

### 5.1 Real-Time Multi-User Sessions

**Goal:** Classroom scenario — students learn together.

**Features:**
- Create a "class session"
- Students join same bandit instance
- See each other's policy choices
- Instructor can reset/freeze/replay

**Implementation:**
- WebSocket upgrade (for real-time updates)
- Room management (Redis)
- Role-based permissions (student, instructor, admin)

**Effort:** High (2-3 weeks)
**Priority:** Low (niche use case)

### 5.2 Instructor Dashboard

**Goal:** Classroom management interface.

**Features:**
- Create lessons for class
- Monitor all students in real-time
- Post announcements
- Review submissions (quizzes, comparisons)
- Export class data (for research)

**Implementation:**
- New backend routers (`/instructors/classes`, `/instructors/students`)
- Frontend dashboard (`/instructor/class/{id}`)
- Pydantic models for class + enrollment

**Effort:** High (3-4 weeks)
**Priority:** Low (educational institution feature)

### 5.3 Research Data Export

**Goal:** Educators & researchers can analyze student data.

**Formats:**
- CSV: (student_id, lesson, policy_chosen, reward_mean, regret, duration)
- JSON: Full trace dump
- SQL: Direct database query

**Implementation:**
- Backend endpoint: `GET /data/export?format=csv&class_id={id}`
- Privacy: hashed user IDs, configurable retention
- Terms of Service covering data usage

**Effort:** Low-Medium (1 week)
**Priority:** Low (research-oriented)

---

## Infrastructure & DevOps

### Database Persistence (When Needed)

**Current:** In-memory session storage
**Future:** PostgreSQL or MongoDB

**When:** When user accounts / analytics phase begins
**Options:**
- PostgreSQL + SQLAlchemy (ORM)
- MongoDB + Pydantic (JSON-native)

### Monitoring & Observability

**Current:** None
**Future:** Add when deployed to production

**Tools:**
- Sentry (error tracking)
- PostHog (analytics)
- Vercel Analytics (frontend performance)
- Datadog (backend metrics)

### API Versioning

**Current:** Single version (`/sessions`)
**Future:** When breaking changes needed: `/v1/sessions`, `/v2/sessions`

---

## Experimentation & Research Ideas

### A. Adaptive Difficulty

Automatically adjust lesson parameters based on student performance:
- High regret? Simplify policy or increase exploration
- Low engagement? Add more interactive elements
- Quick learner? Jump to advanced lessons

### B. Gamification

- Achievement badges (e.g., "Beat Thompson Sampling")
- Leaderboards (optional, privacy-aware)
- Streak counter (consecutive lesson completions)
- XP system (unlocks advanced content)

### C. Socratic Method (AI-Powered Hints)

- Student asks question → Claude/GPT generates hint
- Avoids spoiling answer
- Teaches problem-solving, not just facts

### D. Custom Policy Creation

Allow advanced learners to implement their own policies:
- Web IDE (Monaco Editor or similar)
- Sandboxed Python execution
- Run against platform's bandits
- Compare to pre-built policies

### E. Real-World Datasets

Move beyond synthetic data:
- Real A/B test logs (anonymized)
- Real-time streaming data (stock prices, sports)
- Challenges: "beat this regret baseline with your policy"

---

## Breaking Changes Planned

None in v1.0. Future breaking changes will bump major version:

**v2.0:** User accounts + database (backend schema changes)
**v3.0:** WebSocket upgrade (API protocol change)

---

## Priority Matrix

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| User Accounts | Medium | High | 🟡 Medium |
| i18n | Medium | High | 🟡 Medium |
| Analytics | Medium | Medium | 🟡 Medium |
| Quiz System | High | High | 🟡 Medium |
| Video Tutorials | High | Medium | 🔴 Low |
| Mobile App | High | Medium | 🔴 Low |
| Multi-User | High | Low | 🔴 Low |
| Comparison Mode | Medium | Medium | 🔴 Low |
| Offline PWA | Medium | Medium | 🟡 Medium |

---

## How to Contribute

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details on:
- Proposing new features
- Opening issues
- Submitting PRs
- Code review process

---

## Questions?

Reach out or open an issue on GitHub!
