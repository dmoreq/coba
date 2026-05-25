# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# architecture
- The web module lives at src/web (sibling to src/coba), renamed from flet_redesign. Confidence: 0.65
- Prefer the Split-Workspace Dashboard layout (Option C: 3-zone horizontal + bottom charts) for the COBA web app. Confidence: 0.65

# python
- Use uv for Python dependency management. Confidence: 0.50

# git
- Use conventional commits (e.g., "feat(web):", "fix(web):", "refactor(web):") for all commits. Confidence: 0.88

# workflow
- When implementing features, include unit tests, E2E tests, and clean up legacy/redundant code in the same changeset. Confidence: 0.75
