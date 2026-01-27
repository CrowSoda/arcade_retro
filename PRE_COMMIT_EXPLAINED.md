# Pre-Commit Hooks Explained

## What Are Pre-Commit Hooks?

**Pre-commit hooks are automated checks that run BEFORE you commit code to Git.**

Think of them as a **quality gate** that catches problems before they enter your codebase.

---

## How It Works

```
You type: git commit -m "my changes"
            ↓
Pre-commit hooks run automatically
            ↓
    ┌──────────────────┐
    │ 1. Ruff linter   │ ← Checks code style
    │ 2. Trailing WS   │ ← Removes extra spaces
    │ 3. EOF fixer     │ ← Adds newline at end
    │ 4. YAML checker  │ ← Validates YAML files
    │ 5. Large files   │ ← Blocks files >500KB
    └──────────────────┘
            ↓
    All passed? ✅ Commit allowed
    Failed? ❌ Commit blocked, you fix issues
```

---

## What Does `.pre-commit-config.yaml` Do?

This file defines **which checks run automatically**:

### 1. **Ruff** (Python linter/formatter)
- **What:** Checks Python code style and formatting
- **Why:** Ensures consistent, clean code
- **Example:** Catches missing imports, unused variables, style violations
- **Result:** Auto-fixes 1,698 issues in our codebase!

### 2. **Trailing Whitespace**
- **What:** Removes spaces at end of lines
- **Why:** Keeps git diffs clean
- **Example:** `print("hello")   ` → `print("hello")`

### 3. **End-of-File Fixer**
- **What:** Ensures files end with a newline
- **Why:** POSIX standard, prevents weird git diffs
- **Example:** Adds `\n` at end of every file

### 4. **YAML Checker**
- **What:** Validates YAML syntax
- **Why:** Catches config file errors early
- **Example:** Detects missing colons, bad indentation

### 5. **Large File Checker**
- **What:** Blocks commits with files >500KB
- **Why:** Prevents bloating the repo with binaries
- **Example:** Stops you from accidentally committing a 10MB model file

---

## Installation (DONE ✅)

```bash
# 1. Install pre-commit (DONE)
pip install pre-commit

# 2. Wire it to your Git repo (DONE)
cd g20_demo
pre-commit install
# Output: pre-commit installed at .git\hooks\pre-commit

# 3. Test it works (DONE)
pre-commit run --all-files
```

---

## Daily Workflow

### Normal Commits (Hooks Run Automatically)
```bash
git add .
git commit -m "Add new feature"

# Pre-commit hooks run automatically!
# If they pass → commit succeeds
# If they fail → commit blocked, fix issues
```

### Bypass Hooks (Emergency Only)
```bash
# Use --no-verify to skip hooks (NOT RECOMMENDED)
git commit -m "Emergency hotfix" --no-verify
```

---

## Benefits

1. **Catches Errors Early** - Before they hit the repo
2. **Consistent Code Style** - Everyone follows the same rules
3. **Clean Git History** - No "fix linting" commits
4. **Automated** - No manual checking required
5. **Fast Feedback** - Know immediately if something's wrong

---

## Example: What Happens on Commit

```bash
$ git commit -m "Add new feature"

ruff.....................................................................Passed
Trim Trailing Whitespace.............................................Passed
Fix End of Files.....................................................Passed
Check Yaml...........................................................Passed
Check for added large files..........................................Passed

[main 1a2b3c4] Add new feature
 3 files changed, 42 insertions(+), 5 deletions(-)
```

✅ All checks passed! Commit succeeded.

---

## If A Check Fails

```bash
$ git commit -m "Add buggy code"

ruff.....................................................................Failed
- hook id: ruff
- exit code: 1

backend/server.py:123:5: F841 Local variable `unused` is assigned but never used

Trim Trailing Whitespace.............................................Passed
```

❌ Commit blocked! Fix the issue and try again.

---

## Configuration File Location

- **File:** `g20_demo/.pre-commit-config.yaml`
- **Edit:** Add/remove hooks as needed
- **Update:** Run `pre-commit autoupdate` to get latest versions

---

## Summary

**Pre-commit hooks = Automatic code quality checks before every commit**

- Installed ✅ (wired to `.git/hooks/pre-commit`)
- Configured ✅ (`.pre-commit-config.yaml`)
- Tested ✅ (ran on all files)

**From now on:** Every time you `git commit`, these checks run automatically!

---

**No more manual linting. No more style inconsistencies. Just commit and let the hooks do the work.**
