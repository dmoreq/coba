# COBA Web: Quick Start for Learners

Welcome! This guide explains how to use COBA Web to learn about contextual bandits.

---

## What Is a Contextual Bandit?

In one sentence: **A system that learns which options are best by trying them and observing results, using context (the situation) to make smarter choices.**

Real-world examples:
- **Recommendation:** Which movie should Netflix suggest to you right now?
- **Pricing:** What price should this product have based on demand, time, customer?
- **Bidding:** How much should we bid in this real-time auction?
- **Clinical trials:** What drug dose for this patient's characteristics?

---

## Your Learning Journey

COBA Web teaches you through **17 interactive lessons** spanning beginner to advanced topics. You don't need a machine learning degree — just curiosity and 30–60 minutes per lesson.

**Curriculum Structure:**
- **Beginner (3 lessons):** Foundation: exploration vs exploitation
- **Intermediate (6 lessons):** Contextual learning (use features)
- **Advanced (8 lessons):** Non-linear, adaptive, specialized algorithms

**Estimated time:**
- **Beginner:** ~1.5 hours
- **Intermediate:** ~4 hours
- **Advanced:** ~5–6 hours
- **Total:** 8–12 hours for complete mastery

Do one per day or multiple in a session — whatever works for you.

---

## Getting Started

### 1. Open the Platform

Go to **http://localhost:3000** (local) or your deployed URL.

You'll see the **Curriculum Home Page** with a map of all 17 lessons, organized by difficulty.

### 2. Pick Your Starting Point

Each lesson card shows:
- **Icon** — Visual identifier
- **Title** — What you'll learn
- **Level** — Beginner, Intermediate, or Advanced
- **Problem** — The question you'll answer
- **Prerequisites** — Lessons to do first (if any)

**Start with Lesson 0: Explore vs Exploit** (marked 🎲 Beginner).

### 3. Work Through a Lesson

When you open a lesson, you'll see:

```
┌─────────────────────────────────────┐
│  Algorithm Name           [Beginner] │
├─────────────────────────────────────┤
│ [Theory Card — collapsible]         │
│ - Problem: What's the challenge?    │
│ - Intuition: How does it work?      │
│ - Optional: Mathematical formula    │
│                                     │
│ [Interactive Controls]              │
│ - Play / Pause button               │
│ - Speed selector (1x / 10x / 100x)  │
│                                     │
│ ┌─ Charts (4 columns) ────────────┐ │
│ │ Arm Scores | Pull Counts         │ │
│ │ Rewards    | Regret              │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Decision Trace ────────────────┐ │
│ │ Step-by-step what happened      │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Lesson-Specific Visualization ┐ │
│ │ (Beta distribution, tree, etc)  │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

---

## How to Learn

### 📖 Theory Card

Expand the theory card to understand:
1. **The Problem** — Plain English description of the challenge
2. **The Intuition** — How the algorithm tackles it
3. **The Math (optional)** — Formulas (can stay collapsed)

No math degree needed — the intuition is sufficient.

### ▶️ Run the Simulation

Click **Play** to run the algorithm in real-time:
- The algorithm **decides** which option to try
- You **observe** the reward
- The algorithm **learns** and improves
- Repeat

**Speed controls:**
- **1x** — Watch every step carefully
- **10x** — See patterns emerge
- **100x** — Fast-forward to steady state

### 📊 Read the Charts

Four live charts show you what's happening:

| Chart | Shows |
|-------|-------|
| **Arm Scores (bar)** | How well the algorithm rates each option |
| **Pull Count (histogram)** | How many times each option was tried |
| **Reward Over Time (line)** | Cumulative reward per step — should go up! |
| **Cumulative Regret (line)** | Cost of learning — should go up slowly |

**What to look for:**
- **Reward curve:** Should increase (algorithm learning)
- **Regret curve:** Should flatten (optimal choices dominating)
- **Pulls:** Should concentrate on the best options

### 🔍 Inspect the Trace

The **Decision Trace** panel shows each step:

```
Step 42:  Chose "Option A"
Reward: 0.85
Context: [0.5, -0.2, 1.1, ...]
Cluster: 3 of 5
```

This explains **what just happened** without needing to understand the math.

---

## Learning Tips

### ✅ Do This

1. **Start with Lesson 0** — Don't skip
2. **Read the problem statement** first (before clicking Play)
3. **Expand the theory card** — Understand the idea
4. **Run the demo multiple times** — See patterns
5. **Adjust parameters** — Change alpha, tau, window size, etc. to see effects
6. **Move to the next lesson** — Each builds on the last

### ❌ Don't Do This

1. ❌ Skip lessons — prerequisites exist for a reason
2. ❌ Assume you need a math degree — you don't
3. ❌ Rush — take time to understand
4. ❌ Ignore the trace panel — it's your window into what's happening
5. ❌ Do lessons out of order — they're sequenced pedagogically

---

## Terminology

**Core Concepts:**
- **Reward** — How good a decision was
- **Regret** — Cumulative cost of learning
- **Exploration** — Trying uncertain options
- **Exploitation** — Using what you've learned

**Contextual:**
- **Context** — Features about the current situation
- **Policy** — The decision algorithm
- **Cluster** — Group of similar contexts

**Advanced:**
- **Thompson Sampling** — Bayesian exploration via posterior sampling
- **LinUCB** — Contextual linear learning with confidence bounds
- **Drift** — Distribution change over time

Each lesson explains its specific terminology in context. No prior ML knowledge assumed.

---

## FAQ

### Q: Do I need to understand math?

**A:** No! COBA-Web teaches **concepts first, math second**.
- Every lesson explains ideas in plain English
- Charts guide your intuition
- Formulas are optional (collapsed by default)
- You'll understand *why* before seeing equations

### Q: How long does it take?

**A:** 30–60 minutes per lesson:
- **Beginner (3 lessons):** ~1.5 hours
- **Intermediate (6 lessons):** ~4 hours
- **Advanced (8 lessons):** ~5–6 hours
- **Total:** 8–12 hours for all 17 lessons

### Q: Should I follow a specific order?

**A:** Yes! Start with Lesson 0 and progress sequentially. Prerequisites are marked on the home page.

**Recommended flow:**
```
Lessons 0–2: Foundation (exploration vs exploitation)
    ↓
Lessons 3–5: Contextual basics (learn from features)
    ↓
Lessons 6+: Advanced (clusters, drift, continuous, etc.)
```

### Q: I'm stuck on a concept. What should I do?

**A:**
1. **Re-read the theory card** — Expand sections marked "Problem" or "Intuition"
2. **Run the simulation again** — Watch patterns emerge
3. **Adjust parameters** — Change settings and observe effects
4. **Watch the trace panel** — Inspect individual decisions
5. **Ask in GitHub Discussions** — The community can help

### Q: Will I understand real-world systems after?

**A:** Yes! You'll master core algorithms used in production:
- **A/B Testing** — UCB1, Thompson Sampling
- **Personalization** — LinUCB, Neural Linear
- **Real-time Optimization** — CATS, Sliding-Window LinUCB
- **Handling Change** — Drift Detection

Real systems add constraints (budgets, fairness, regulations), but you'll have the algorithmic foundation. See [Algorithm Reference](./algorithms/) for deep-dives.

### Q: Can I download my progress?

**A:** Your progress is saved locally in your browser. To back it up:
- **Bookmark lessons** you want to revisit
- **Take notes** as you go
- Your data persists across sessions (until you clear browser cache)

---

## Next Steps

1. **Open the platform:** http://localhost:3000
2. **Read Lesson 0's theory card** — Understand explore vs exploit
3. **Click Play** on Lesson 0 — Watch the algorithm learn
4. **Work through at your own pace** — One or more per session
5. **Share your feedback** — Open an issue or discussion on GitHub

---

## Resources

- 📚 **Algorithm Reference:** [docs/algorithms/](./algorithms/)
- 🏗️ **Architecture Guide:** [docs/ARCHITECTURE.md](./ARCHITECTURE.md)
- 💻 **For Developers:** [docs/ADDING_LESSONS.md](./ADDING_LESSONS.md)
- 📖 **Policies Overview:** [docs/policies.md](./policies.md)

---

**Happy learning! You're about to understand how systems learn to make smarter decisions. 🚀**
