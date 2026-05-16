# Adding New Lessons to COBA-Web

This guide explains how to extend the curriculum by adding new lessons.

## Overview

The curriculum is structured as 8 progressive lessons + 1 reference page. Each lesson teaches one core concept through:
1. **Problem** — plain-English statement of the challenge
2. **Why It's Hard** — analysis of the difficulty
3. **The Technique** — how the algorithm solves it
4. **Interactive Demo** — controls, charts, step-by-step narrative
5. **What You're Seeing** — chart interpretation guide

## Lesson Structure

### 1. Add Lesson Metadata

Update `lib/lessons.ts`:

```typescript
// lib/lessons.ts
export const LESSONS = [
  // ... existing lessons
  {
    number: 9,
    href: "/my-new-lesson",
    icon: "🎯",
    label: "Your Lesson Title",
    level: "advanced", // "beginner", "intermediate", or "advanced"
    problem: "Clear one-sentence problem statement",
    prereqs: [2, 3], // Lessons that must come first
  },
];
```

Also add to backend `routers/curriculum.py`:

```python
# routers/curriculum.py
LESSONS = [
    # ... existing lessons
    {
        "number": 9,
        "href": "/my-new-lesson",
        "icon": "🎯",
        "label": "Your Lesson Title",
        "level": "advanced",
        "problem": "...",
        "prereqs": [2, 3],
    },
]
```

### 2. Create Lesson Page

Create `app/my-new-lesson/page.tsx`:

```typescript
"use client";

import { useState } from "react";
import { LessonHeader } from "@/components/education/LessonHeader";
import { LessonNav } from "@/components/education/LessonNav";
import { PageShell } from "@/components/layout/PageShell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function MyNewLessonPage() {
  const [isRunning, setIsRunning] = useState(false);

  return (
    <PageShell>
      <LessonHeader lessonNumber={9} />

      {/* Your lesson content here */}
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Interactive Demo</CardTitle>
          </CardHeader>
          <CardContent>
            <Button onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? "Stop" : "Start"} Demo
            </Button>
          </CardContent>
        </Card>
      </div>

      <LessonNav lessonNumber={9} />
    </PageShell>
  );
}
```

### 3. Add Glossary Tooltips

Wrap technical terms with `<GlossaryTip>`:

```typescript
import { GlossaryTip } from "@/components/education/GlossaryTip";

// In your JSX:
<GlossaryTip term="Exploration">
  exploring new options
</GlossaryTip>
```

Available terms (from `lib/glossary.ts`):
- Reward, Regret, Exploration, Exploitation, Context, Cluster
- Policy, UCB, Thompson Sampling, LinUCB, Propensity, IPS
- Doubly Robust, Drift, Cold-Start

### 4. Create Chart Insights

Add descriptions to `lib/chart-insights.ts`:

```typescript
// lib/chart-insights.ts
export const chartInsights = {
  // ... existing insights
  myNewLessonChart: "Your chart explanation here...",
};
```

### 5. Add Backend Demo Scenario (Optional)

Update `routers/scenarios.py`:

```python
# routers/scenarios.py
DEMO_SCENARIOS = {
    # ... existing scenarios
    9: {
        "name": "Your Scenario Name",
        "description": "What the learner will see...",
        "n_features": 3,
        "n_arms": 4,
        # ... other parameters
    },
}
```

### 6. Write Tests

Create `app/__tests__/my-new-lesson.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import MyNewLessonPage from "@/app/my-new-lesson/page";

describe("Lesson 9: My New Lesson", () => {
  it("renders the lesson header", () => {
    render(<MyNewLessonPage />);
    expect(screen.getByText(/lesson.*9/i)).toBeInTheDocument();
  });

  it("renders the interactive demo", () => {
    render(<MyNewLessonPage />);
    expect(screen.getByText(/start.*demo/i)).toBeInTheDocument();
  });

  it("renders the lesson navigation", () => {
    render(<MyNewLessonPage />);
    expect(screen.getByText(/lesson of 8/i)).toBeInTheDocument();
  });
});
```

Run tests:
```bash
npm run test
```

### 7. Update Navigation Sidebar

The sidebar is automatically updated by `NavSidebar.tsx` reading from `lib/lessons.ts`.

No manual changes needed!

## Best Practices

### Problem Statement

✅ **Good**: "How do you choose between options when rewards are random?"
❌ **Bad**: "Implement contextual bandits"

### Why It's Hard

Explain the conflict or challenge:
- "You can't learn everything instantly"
- "Exploration costs immediate reward"
- "The world changes over time"

### Technique Explanation

Start simple, then go deeper:
1. **Plain English**: "The algorithm picks options with high uncertainty"
2. **Example**: "A new ad gets tried even if old ads look better"
3. **Math**: "Q(a) + c√(ln(t)/n_a)" (optional, expandable)

### Charts

- ✅ Use domain-agnostic labels ("Reward", not "CTR")
- ✅ Include axis labels and units
- ✅ Add insight text explaining what to look for
- ✅ Show good/bad patterns (e.g., "flat line = bad learning")

### Glossary Terms

- Define every jargon word
- Provide concrete examples
- Link to related concepts
- Use consistent terminology across all lessons

## Example: Complete Lesson

See `app/playground/page.tsx` for a full example of:
- LessonHeader integration
- Interactive demo with controls
- Charts with insights
- Step narrative
- LessonNav integration
- GlossaryTips on controls

## Common Patterns

### Running a Simulation

```typescript
const [steps, setSteps] = useState<number[]>([]);
const [running, setRunning] = useState(false);

async function runSimulation() {
  setRunning(true);
  const { session_id } = await api.simulate.start({
    policy: "linucb",
    n_features: 3,
    n_arms: 4,
  });
  const source = new EventSource(api.simulate.streamUrl(sessionId));
  source.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    setSteps((prev) => [...prev, event.step]);
    // ... update other state
    if (event.done) source.close();
  };
}
```

### Displaying Charts

```typescript
<ChartCard title="Your Chart Title" insight={chartInsights.yourKey}>
  <PlotlyChart
    data={[{ x: steps, y: rewards, type: "scatter", mode: "lines" }]}
    layout={{ yaxis: { title: "Reward" } }}
  />
</ChartCard>
```

### Showing Step Narrative

```typescript
<StepNarrative step={currentStep} />
```

The narrative automatically explains each decision based on `lib/narrative.ts`.

## Checklist for New Lessons

- [ ] Add lesson metadata to `lib/lessons.ts` and backend
- [ ] Create page at `app/my-lesson/page.tsx`
- [ ] Add LessonHeader and LessonNav
- [ ] Add GlossaryTips to all technical terms
- [ ] Create chart insights in `lib/chart-insights.ts`
- [ ] Add demo scenario to `routers/scenarios.py` (optional)
- [ ] Write unit tests
- [ ] Test on mobile (responsive)
- [ ] Test in dark mode
- [ ] Test keyboard navigation (Tab, Enter, Escape)
- [ ] Add PR description explaining the lesson concept
- [ ] Request review from @dmoreq

## Curriculum Progression

Lessons should build on each other:

```
Lesson 1: What are bandits?
  ↓
Lesson 2: Explore vs Exploit
  ├→ Lesson 3: Compare strategies
  ├→ Lesson 4: Learn from history
  └→ Lesson 5: Adapt to changes
  ├→ Lesson 6: Detect drift
  ├→ Lesson 7: Continuous actions
  └→ Lesson 8: Real-world constraints
```

Lessons with same prerequisites can be done in any order.

## Questions?

1. Check existing lessons (`app/playground/`, `app/policy-lab/`, etc.)
2. Review `lib/glossary.ts` for terminology
3. Ask in GitHub Discussions
4. File an issue with your question

---

**Thank you for extending COBA-Web's curriculum!**
