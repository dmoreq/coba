# Contributing to COBA-Web

Thank you for your interest in contributing to COBA-Web! This document provides guidelines for different types of contributions.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Types of Contributions](#types-of-contributions)
3. [Lesson Contributions](#lesson-contributions)
4. [Code Contributions](#code-contributions)
5. [Commit Message Guidelines](#commit-message-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Code Style and Standards](#code-style-and-standards)

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/coba.git
   cd coba
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. **Set up development environment**:
   - Follow [DEPLOYMENT.md](docs/DEPLOYMENT.md) for backend and frontend setup
   - Run tests before making changes: `npm run test` (frontend), `pytest` (backend)

---

## Types of Contributions

### 🎓 Contributing Lessons

Want to teach a new concept? See [ADDING_LESSONS.md](docs/ADDING_LESSONS.md) for complete guide.

**Process:**
1. Propose the lesson in GitHub Discussions
2. Create the lesson following the template
3. Write tests and documentation
4. Submit PR with educational content

**Checklist:**
- [ ] Lesson metadata added to `lib/lessons.ts`
- [ ] Page created at `app/{lesson-name}/page.tsx`
- [ ] LessonHeader and LessonNav integrated
- [ ] GlossaryTips on all jargon terms
- [ ] Charts with plain-English labels and insights
- [ ] Unit tests written (80%+ coverage)
- [ ] Mobile responsive (tested on 375px, 768px, 1024px)
- [ ] Dark mode works correctly
- [ ] Keyboard navigation tested (Tab, Enter, Escape)

**Example**: See [ADDING_LESSONS.md](docs/ADDING_LESSONS.md) for a step-by-step walkthrough.

### 🐛 Bug Fixes

**Process:**
1. Create an issue describing the bug
2. Create a branch: `fix/issue-name`
3. Fix the bug with tests
4. Reference the issue in your PR

**Minimum requirements:**
- [ ] Failing test case added first (TDD)
- [ ] Bug fix implemented
- [ ] All existing tests pass
- [ ] Commit message references issue (#123)

### ✨ Features

**Process:**
1. Propose feature in GitHub Discussions or Issues
2. Get feedback before starting
3. Create branch: `feat/feature-name`
4. Implement with tests
5. Submit PR

**Minimum requirements:**
- [ ] Tests written (unit + E2E if applicable)
- [ ] Documentation updated
- [ ] No breaking changes (or major version bump)
- [ ] Performance impact assessed

### 📚 Documentation

**Process:**
1. Create branch: `docs/what-you-are-documenting`
2. Add/update documentation files
3. Submit PR

**Examples:**
- Updating README
- Writing deployment guides
- Creating tutorials
- Adding API documentation

### ♿ Accessibility Improvements

**Process:**
1. Document accessibility issue
2. Fix following WCAG 2.1 AA standards
3. Test with screen readers and keyboard-only navigation
4. Submit PR

---

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature (lessons, endpoints, components)
- **fix**: Bug fix
- **refactor**: Code restructuring (no behavior change)
- **perf**: Performance optimization
- **test**: Adding or updating tests
- **docs**: Documentation changes
- **chore**: Maintenance (dependencies, config)
- **style**: Code style (formatting, linting)

### Scope

- `backend`: Backend Python code
- `frontend`: Frontend TypeScript/React
- `edu`: Educational content (lessons, glossary)
- `ui`: User interface components
- `test`: Test suite
- `perf`: Performance
- `a11y`: Accessibility

### Examples

```
feat(edu): add lesson 9 on multi-armed bandits
fix(frontend): glossary tooltip not closing on mobile
refactor(backend): consolidate event models
perf(ui): optimize chart rendering with React.memo
test(backend): add curriculum endpoint tests
docs: add lesson contribution guide
```

---

## Pull Request Process

### Before Submitting

1. **Update from main**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run tests**:
   ```bash
   # Frontend
   cd coba-web/frontend
   npm run test
   npm run build

   # Backend
   cd coba-web/backend
   pytest
   ```

3. **Lint code**:
   ```bash
   # Frontend
   npm run lint

   # Backend
   ruff check .
   ```

### PR Template

Use this template for your PR description:

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] New lesson
- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] Performance improvement
- [ ] Accessibility improvement

## Related Issue
Closes #123

## Changes Made
- Point 1
- Point 2
- Point 3

## Testing
- [ ] Unit tests added
- [ ] E2E tests added
- [ ] Manual testing on mobile
- [ ] Dark mode tested
- [ ] Keyboard navigation tested

## Checklist
- [ ] Code follows style guide
- [ ] Tests pass locally
- [ ] No breaking changes
- [ ] Documentation updated
- [ ] Commit messages follow conventions
```

### Review Process

1. **Automated checks** must pass:
   - Tests
   - Linting
   - Type checking

2. **Code review**: At least one maintainer approves

3. **Educational content**: Reviewed for:
   - Clarity and correctness
   - Jargon is explained
   - Accessibility compliance
   - Progressive difficulty

4. **Merge**: Squash and merge to keep history clean

---

## Code Style and Standards

### Frontend (TypeScript/React)

- **Prettier** for formatting (2-space indent)
- **ESLint** for linting
- **TypeScript strict mode** enabled
- **Component naming**: PascalCase (`<MyComponent />`)
- **File naming**: kebab-case (`my-component.tsx`)
- **Props interfaces**: `<ComponentName>Props`

Example:

```typescript
// ✅ Good
interface MyComponentProps {
  lesson: Lesson;
  onComplete?: () => void;
}

export function MyComponent({ lesson, onComplete }: MyComponentProps) {
  return <div>{lesson.label}</div>;
}

// ❌ Bad
export function my_component(props: any) {
  return <div>{props.lesson}</div>;
}
```

### Backend (Python)

- **Black** for formatting
- **Ruff** for linting
- **Type hints** on all functions
- **Docstrings** on classes and functions
- **Function naming**: snake_case
- **Class naming**: PascalCase

Example:

```python
# ✅ Good
def get_lesson_by_number(number: int) -> Optional[Lesson]:
    """Fetch a lesson by its number.

    Args:
        number: The lesson number (1-9)

    Returns:
        The lesson, or None if not found
    """
    return LESSONS.get(number)

# ❌ Bad
def getLessonByNumber(number):
    return LESSONS.get(number)
```

### Testing

- **Jest** + **React Testing Library** (frontend)
- **Pytest** (backend)
- **Snapshot tests** for UI components
- **Unit test** each component
- **E2E tests** for critical flows
- **80%+ code coverage** target

Example:

```typescript
// ✅ Good
describe("LessonHeader", () => {
  it("renders lesson number correctly", () => {
    render(<LessonHeader lessonNumber={2} />);
    expect(screen.getByText(/lesson 2/i)).toBeInTheDocument();
  });

  it("displays level badge for beginner lessons", () => {
    render(<LessonHeader lessonNumber={2} />);
    expect(screen.getByText(/beginner/i)).toBeInTheDocument();
  });
});
```

---

## Running Tests Locally

```bash
# Frontend
cd coba-web/frontend
npm run test                  # Run tests
npm run test:coverage         # Coverage report
npm run test:watch          # Watch mode

# Backend
cd coba-web/backend
pytest                        # Run tests
pytest --cov                  # Coverage report
pytest -v                     # Verbose output
```

---

## Questions?

- 💬 **Discussions**: https://github.com/dmoreq/coba/discussions
- 🐛 **Issues**: https://github.com/dmoreq/coba/issues
- 📧 **Email**: Open an issue and we'll connect

---

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct:

- Be respectful and inclusive
- Assume good intent
- Focus on the idea, not the person
- Help others learn

---

**Thank you for contributing! You make COBA-Web better for everyone! 🚀**
