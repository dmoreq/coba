# COBA-Web: Quick Start for Learners

Welcome! This guide explains how to use COBA-Web to learn about contextual bandits.

## What Is a Contextual Bandit?

In one sentence: **A system that learns which options are best by trying them and observing results, using context (the situation) to make smarter choices.**

## Your Learning Journey

COBA-Web teaches you through 8 interactive lessons. You don't need a machine learning degree — just curiosity and 30-60 minutes per lesson.

```
🎓 Home Page (curriculum map)
  ↓
🎲 Lesson 1: Explore vs Exploit
  • Understand the fundamental trade-off
  • Estimated time: 30 min
  ↓
⚖️ Lesson 2: Strategy Comparison
  • Compare different learning strategies
  • Build on: Lesson 1
  ↓
[... more lessons ...]
```

## Getting Started

### 1. Open the Platform

Go to **http://localhost:3000** (local) or your deployed URL.

You'll see the **Curriculum Home Page** with a map of all lessons.

### 2. Read the Curriculum Overview

Each lesson card shows:
- 🎯 **Icon** — quick visual identifier
- **Title** — what you'll learn
- **Level** — beginner, intermediate, or advanced
- **Problem** — the question you'll answer
- **Prerequisites** — lessons to do first

Start with **Lesson 1: Explore vs Exploit** (marked 🎲 beginner).

### 3. Work Through a Lesson

When you open a lesson, you'll see:

```
┌─────────────────────────────────────┐
│  🎲 Explore vs Exploit         [Beginner]  │
├─────────────────────────────────────┤
│ THE PROBLEM                              │
│ How do you balance trying new things    │
│ with sticking with what works?          │
│                                          │
│ WHY IT'S HARD                           │
│ • If you always try new options,        │
│   you never exploit the good ones       │
│ • If you never explore, you'll miss     │
│   better options hiding in plain sight  │
│                                          │
│ THE TECHNIQUE                            │
│ The algorithm uses a strategy that...   │
└─────────────────────────────────────┘

[INTERACTIVE DEMO]
[Controls, Charts, Step Narrative]

[WHAT YOU'RE SEEING]
Chart guide and interpretation
```

## Using the Glossary

Every lesson defines technical terms **inline**. When you see an underlined term like:

> "The algorithm uses <u>exploration</u> to try new options."

**Click it** to see a plain-English definition:

```
Exploration: Trying options you haven't tested much yet,
even if they currently look worse based on limited data.

Example: Showing an ad with unknown click rate to learn
if it's actually good, even though current data suggests
it's mediocre.
```

**All glossary terms in COBA-Web:**

1. **Reward** — How good a decision was (1=click, 0=ignore)
2. **Regret** — The cost of learning (how much you left on the table)
3. **Exploration** — Trying uncertain options
4. **Exploitation** — Using what you've learned
5. **Context** — Info about the current situation
6. **Cluster** — A group of similar situations
7. **Policy** — The algorithm's decision rule
8. **UCB** — Upper Confidence Bound (one exploration strategy)
9. **Thompson Sampling** — Another exploration strategy
10. **LinUCB** — Learning with context
11. **Propensity** — Probability of a choice
12. **IPS** — Inverse Propensity Scoring (unbiasing)
13. **Doubly Robust** — More stable version of IPS
14. **Drift** — When the world changes
15. **Cold-Start** — Learning from zero data

If a term confuses you, **click it**. That's what it's there for!

## The Interactive Demos

Each lesson has a **Play** button. Click it to:

1. **See the algorithm learn in real-time**
2. **Understand what's happening** via the step narrative
3. **Visualize the learning** on charts

### Reading the Charts

Every chart shows **what you should look for**:

Example: "Explore vs Exploit"
- Chart title: "Average Reward Over Time"
- What to watch: "The line should go up over time"
- Bad sign: "Flat line = algorithm isn't learning"
- Good sign: "Increasing curve = finding better options"

### The Step Narrative

Below each chart, you'll see:
> "Step 5: Chose option_B. Reward: 0. Cluster: urban_young. Average so far: 0.45"

This explains **what just happened** in plain language. You don't need to know the formula — the narrative tells you the story.

## Learning Tips

### ✅ Do This

1. **Start with Lesson 1** — don't skip ahead
2. **Read the problem statement** before running the demo
3. **Click underlined terms** — they have definitions
4. **Run the demo** multiple times — notice patterns
5. **Read the chart interpretations** — they explain what you're seeing
6. **Move to the next lesson** — each builds on the last

### ❌ Don't Do This

1. ❌ Skip lessons — prerequisites exist for a reason
2. ❌ Try to memorize formulas — you'll see the math later (optional)
3. ❌ Ignore chart explanations — they guide interpretation
4. ❌ Assume you need an ML degree — you don't!
5. ❌ Rush through — take your time

## Common Questions

### Q: Do I need to understand math?

**A:** No! COBA-Web teaches the **concepts first, math second**.
- Every lesson explains ideas in plain English
- Charts guide your intuition
- Formulas are optional (collapsed by default)
- You'll understand what's happening before seeing equations

### Q: How long does it take?

**A:** 30-60 minutes per lesson. The whole curriculum:
- 🎲 Lesson 1 — 30 min
- ⚖️ Lesson 2 — 40 min
- ... (each ~30-45 min)
- **Total: ~4-5 hours for all 8 lessons**

You can do them one per session or all at once.

### Q: Can I do lessons out of order?

**A:** Not recommended. Each lesson builds on previous ones. The curriculum order is:

```
Lesson 1 (required foundation)
  ↓
Lesson 2 & 3 (explore-exploit, strategies)
  ↓
Lesson 4, 5, 6 (extensions: history, changes, drift)
  ↓
Lesson 7 & 8 (advanced: continuous, constraints)
```

Some lessons have no prerequisites (shown on home page), but doing them in order makes learning easier.

### Q: I got stuck on a concept. What do I do?

**A:**
1. **Click glossary terms** — definitions include examples
2. **Re-read the "Why It's Hard" section** — it explains the challenge
3. **Run the demo again** — watch patterns emerge
4. **Read the chart guide** — "What You're Seeing" explains interpretation
5. **Ask in GitHub Discussions** if you still need help

### Q: Will I understand real-world systems after?

**A:** You'll understand the **core concepts** (explore-exploit, learning from data, adaptation). Real systems add complexity (multiple objectives, constraints, business logic), but you'll have the foundation.

The Algorithm Reference lesson shows more advanced techniques you can explore later.

### Q: Can I download or export my progress?

**A:** Not yet, but we're working on it! For now:
- Your progress is saved locally in your browser
- Bookmark lessons you want to revisit
- Take notes as you go

## Next Steps

1. **Open the home page**: http://localhost:3000
2. **Click "Start Lesson 1"** or read the curriculum overview
3. **Work through Lesson 1** (30 min)
4. **Complete the curriculum** (one lesson per day or all at once)
5. **Share your feedback** in GitHub Discussions

## Glossary (Quick Reference)

| Term | Meaning |
|------|---------|
| **Reward** | How good a decision was |
| **Regret** | Cost of learning |
| **Exploration** | Trying uncertain options |
| **Exploitation** | Using what you've learned |
| **Context** | Info about the situation |
| **Cluster** | Group of similar situations |
| **Policy** | The algorithm's decision rule |
| **Drift** | World changes over time |

**See lessons for full definitions with examples.**

## Getting Help

- 📚 **Lessons** — hover over/click any term
- 💬 **Discussions** — https://github.com/dmoreq/coba/discussions
- 🐛 **Issues** — Report bugs: https://github.com/dmoreq/coba/issues
- 📧 **Email** — Open an issue, we'll respond

---

**Happy learning! You're about to understand how systems learn. 🚀**
