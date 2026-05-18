# Adding New Lessons to COBA Web

This guide explains how to add a new interactive lesson to the platform.

## Overview

A lesson teaches one algorithm through an interactive simulation:
1. **Theory** — problem, intuition, technique (collapsible card)
2. **Simulation** — real-time stepping with configurable speed (1x/10x/100x)
3. **Visualization** — reward/regret curves, arm scores, trace log
4. **Learning by doing** — adjust parameters, watch behavior change

All 17 lessons are registered in a single file (`lib/lessons.ts`) and rendered via a dynamic route.

---

## Step 1: Add Lesson Metadata to Registry

**File:** `web/frontend/lib/lessons.ts`

The `LESSONS` array defines all 17 lessons. Add a new `LessonMeta` object:

```typescript
{
  slug: "my-new-algorithm",           // URL slug (unique, kebab-case)
  index: 17,                          // Position in curriculum (0-16 existing)
  title: "My New Algorithm",          // Display title
  scenario: "Real-world use case...", // Problem description (1 sentence)
  difficulty: "advanced",             // "beginner" | "intermediate" | "advanced"
  policy: "my_policy",                // Policy type (must exist in backend)
  arms: ["Option A", "Option B"],     // Arm names (or [] for continuous)
  nFeatures: 5,                       // Context feature count
  defaultConfig: {                    // BanditConfig defaults
    policy: "my_policy",
    alpha: 1.0,
    nClusters: 3,
  },
  prerequisites: [15],                // Lesson indices that must come first
  rewardFn: (arm: string, context: number[]) => {
    // Client-side reward function
    // Maps (arm, context) → reward (0-1)
    return Math.random(); // Replace with actual logic
  },
}
```

### Key Fields Explained

- **`slug`**: Used in URL (`/lesson/my-new-algorithm`). Must be unique.
- **`index`**: Position in curriculum. For 18th lesson, use 17 (0-indexed).
- **`policy`**: Must match a `PolicyType` in `web/frontend/lib/types.ts` AND be implemented in the backend (`web/backend/app/simulators/`).
- **`arms`**: List of arm names. Empty for continuous policies like CATS.
- **`nFeatures`**: Context dimension. Backend validates: `len(context) == nFeatures`.
- **`defaultConfig`**: Initial `BanditConfig` for the lesson. Users can adjust in UI.
- **`prerequisites`**: Array of lesson indices that should be completed first. Blocks progression in the UI.
- **`rewardFn`**: **Important** — rewards are computed **client-side**, not server-side. This function observes the chosen arm and context, returns a reward in `[0, 1]`.

---

## Step 2: Implement Backend Reward Simulator

**File:** `web/backend/app/simulators/`

The backend doesn't compute rewards for the bandit; it only runs the decision logic. The frontend calls `rewardFn` to generate the reward, then sends it via `POST /api/sessions/{id}/update`.

However, for **offline evaluation** and **drift injection** tests, the backend needs to know the reward function.

Add your simulator to the appropriate file:

- **Context-free policies** (UCB1, Thompson) → `context_free.py`
- **Linear contextual** (LinUCB, LinTS, Logistic) → `contextual_linear.py`
- **Advanced** (Neural, Forest, GP, CATS) → `advanced.py`

```python
# web/backend/app/simulators/advanced.py

def reward_my_new_algorithm(arm: str, context: np.ndarray) -> float:
    """Return reward in [0, 1] for (arm, context) pair."""
    # Example: sigmoid of dot product
    weights = {"Option A": [0.3, 0.5, -0.1], ...}
    w = weights.get(arm, [0] * len(context))
    logit = sum(w[i] * context[i] for i in range(len(context)))
    return 1 / (1 + np.exp(-logit))  # sigmoid
```

---

## Step 3: Create Lesson Component

**File:** `web/frontend/components/lesson/MyNewAlgorithmLesson.tsx`

All lessons follow the same structure:

```tsx
"use client";

import { useState } from "react";
import { useSession } from "@/lib/hooks/useSession";
import { useSimulator } from "@/lib/hooks/useSimulator";
import { getLessonBySlug } from "@/lib/lessons";
import { TheoryCard } from "@/components/lesson/TheoryCard";
import { TracePanel } from "@/components/lesson/TracePanel";
import { LessonShell } from "@/components/lesson/LessonShell";
import { LessonControls } from "@/components/lesson/LessonControls";
import { RewardRegretSection } from "@/components/lesson/RewardRegretSection";
import { ArmBar } from "@/components/charts/ArmBar";
import { RewardChart } from "@/components/charts/RewardChart";
import { RegretChart } from "@/components/charts/RegretChart";
import { PullHistogram } from "@/components/charts/PullHistogram";

export default function MyNewAlgorithmLesson() {
  const lesson = getLessonBySlug("my-new-algorithm")!;
  const session = useSession(lesson);
  const simulator = useSimulator(session, lesson.rewardFn);

  const [showTheory, setShowTheory] = useState(false);

  if (!session.isReady) return <div>Loading...</div>;

  return (
    <LessonShell
      topSection={
        <TheoryCard
          title="Algorithm Name"
          sections={[
            {
              heading: "The Problem",
              content: "Plain English explanation of the challenge...",
            },
            {
              heading: "The Intuition",
              content: "How the algorithm solves it...",
            },
            {
              heading: "The Math (Optional)",
              content: "Formula and derivation (can be collapsed)...",
            },
          ]}
          isOpen={showTheory}
          onToggle={setShowTheory}
        />
      }
      scoreCard={<ArmBar arms={simulator.trace[0]?.armScores || []} />}
      pullsCard={<PullHistogram stats={session.stats} />}
      rewardCard={
        <RewardChart
          data={simulator.trace.map((t, i) => ({
            x: i,
            y: t.reward,
          }))}
          label="Reward per Step"
        />
      }
      regretCard={
        <RegretChart
          data={simulator.trace.map((t, i) => ({
            x: i,
            y: t.regret,
          }))}
          label="Cumulative Regret"
        />
      }
      traceSection={<TracePanel entries={simulator.trace} compact />}
      vizSection={
        // Optional: lesson-specific visualization
        // E.g., BetaDistribution, TreeDiagram, ConfidenceEllipse, etc.
        undefined
      }
    />
  );
}
```

### Component Imports Explained

- **`useSession(lesson)`** — Manages bandit session lifecycle (create, step, update, reset). Returns: `sessionId`, `nFeatures`, `step()`, `update()`, `stats`, `reset()`.
- **`useSimulator(session, rewardFn)`** — Auto-stepping loop at configurable speed. Returns: `isRunning`, `speed`, `totalSteps`, `trace`, `start()`, `pause()`, `step()`, `setSpeed()`.
- **`LessonShell`** — Layout wrapper. Props: `topSection`, `scoreCard`, `pullsCard`, `rewardCard`, `regretCard`, `traceSection`, `vizSection`.
- **`TheoryCard`** — Collapsible theory explanation. Props: `title`, `sections[]`, `isOpen`, `onToggle`.
- **`TracePanel`** — Decision log showing each step's context, arm, reward. Props: `entries`, `compact`.
- **`LessonControls`** — Play/pause, speed buttons. (Auto-included in most lessons.)
- **Charts** — `ArmBar`, `RewardChart`, `RegretChart`, `PullHistogram`. All are custom SVG components, not Recharts.

---

## Step 4: Register Component in Router

**File:** `web/frontend/components/lesson/registry.tsx`

Import your lesson and add to the registry:

```tsx
import MyNewAlgorithmLesson from "./MyNewAlgorithmLesson";

export const LESSON_COMPONENTS: Record<string, React.ComponentType<{}>> = {
  // ... existing lessons
  "my-new-algorithm": MyNewAlgorithmLesson,
};
```

The dynamic route `app/lesson/[slug]/page.tsx` uses this registry to render the correct component.

---

## Step 5: Write Tests

**Path:** `web/frontend/tests/unit/components/lesson/`

```typescript
// MyNewAlgorithmLesson.test.tsx
import { render, screen } from "@testing-library/react";
import MyNewAlgorithmLesson from "@/components/lesson/MyNewAlgorithmLesson";

// Mock useSession and useSimulator
jest.mock("@/lib/hooks/useSession");
jest.mock("@/lib/hooks/useSimulator");

describe("MyNewAlgorithmLesson", () => {
  it("renders the theory card", () => {
    render(<MyNewAlgorithmLesson />);
    expect(screen.getByText(/Algorithm Name/i)).toBeInTheDocument();
  });

  it("renders the shell with all 4 chart cards", () => {
    render(<MyNewAlgorithmLesson />);
    // Check for card titles or key elements
    expect(screen.getByText(/Reward/i)).toBeInTheDocument();
    expect(screen.getByText(/Regret/i)).toBeInTheDocument();
  });
});
```

Run tests:
```bash
npm test
```

---

## Step 6: Test Locally

1. **Start both servers:**
   ```bash
   # Terminal 1: Backend
   cd web/backend
   source venv/bin/activate
   uvicorn app.main:app --reload

   # Terminal 2: Frontend
   cd web/frontend
   npm run dev
   ```

2. **Open browser:** `http://localhost:3000`

3. **Navigate to your lesson:** `/lesson/my-new-algorithm`

4. **Test interactions:**
   - Click Play → should step and show trace
   - Adjust speed (1x/10x/100x) → should change stepping rate
   - Observe reward/regret curves updating
   - Try keyboard shortcuts (Space for play/pause, ← / → for step, 1/2/3 for speed)

---

## Step 7: Complete Checklist Before PR

- [ ] Lesson metadata added to `lib/lessons.ts` with unique slug
- [ ] Component created in `components/lesson/` and registered in `registry.tsx`
- [ ] Component imports `LessonShell` and uses correct layout props
- [ ] `rewardFn` is implemented and returns values in `[0, 1]`
- [ ] Backend simulator added to appropriate `simulators/` file
- [ ] Unit tests written for component (80%+ coverage)
- [ ] Mobile responsive tested (375px, 768px, 1024px widths)
- [ ] Dark mode works (toggle theme in top bar)
- [ ] Keyboard shortcuts work (Space, ←/→, 1/2/3, ?)
- [ ] All existing tests pass: `npm test`
- [ ] Build succeeds: `npm run build`

---

## Example: Complete Minimal Lesson

Here's a complete minimal lesson to copy from:

**Lesson: `intro` (Explore vs Exploit)**
- **File:** `web/frontend/components/lesson/ExploreExploitLesson.tsx`
- **Slug:** `intro`
- **Policy:** `epsilon_greedy`
- **Args:** 3 arms, 1 feature (context-free), epsilon=0.1

Check this lesson for:
- Actual component structure
- Real `useSession` + `useSimulator` usage
- How to build charts from trace data
- Theory card format

---

## Troubleshooting

### "Session creation failed: 422 Validation Error"
- Check `nFeatures` matches backend validation
- Check `policy` exists in `PolicyType` enum
- Check config fields are spelled correctly (snake_case in backend request)

### "Cannot GET /lesson/my-new-algorithm"
- Slug must match exactly in `lib/lessons.ts` and `registry.tsx`
- Slug must be kebab-case (no underscores)

### "Charts not updating"
- Check `simulator.trace` is populating (breakpoint in console)
- Check `rewardFn` is returning finite numbers (not NaN, not Infinity)
- Check component is rendering with latest trace data

### "Tests failing: useSession mock issues"
- Ensure you're mocking with proper return type
- Use the test helpers in `tests/unit/testUtils/sessionMocks.ts`

---

## API Contract Reference

When you implement `rewardFn`, remember:
- **Input:** `(arm: string, context: number[])` — the chosen arm and context vector
- **Output:** `number` in `[0, 1]` — reward signal
- **Timing:** Called immediately after `step()`, before `update()`

Example from lesson registry:
```typescript
rewardFn: (arm: string, context: number[]) => {
  const weights = {
    "Option A": [0.3, 0.5, 0.1],
    "Option B": [0.1, -0.2, 0.4],
  };
  const w = weights[arm] || [0, 0, 0];
  const logit = w.reduce((sum, wi, i) => sum + wi * context[i], 0);
  return 1 / (1 + Math.exp(-logit)); // sigmoid
};
```

---

## Questions?

1. Check existing lessons in `components/lesson/` for working examples
2. Review `lib/lessons.ts` for lesson metadata patterns
3. See `web/backend/app/simulators/` for reward function examples
4. Check `CONTRIBUTING.md` for code style guidelines
5. Open a GitHub Discussion if something is unclear

---

**Happy building! Your lesson will teach thousands of learners about contextual bandits. 🚀**
