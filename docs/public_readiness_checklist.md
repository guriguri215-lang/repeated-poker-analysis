# Public Readiness Checklist

## Purpose

- This checklist is used before making the repository public.
- It is not a release note and not a marketing document.
- It helps prevent accidental disclosure, overclaiming, and confusing project
  scope.

## Required checks before public visibility

### Repository hygiene

- [ ] No private paths
- [ ] No personal email addresses
- [ ] No tokens, passwords, API keys, cookies, or recovery codes
- [ ] No external solver exports or private solver files
- [ ] No accidental `.claude/`, `.venv/`, cache, or build artifacts
- [ ] `.gitignore` covers local-only files

### Claims and interpretation

- [ ] README states this is experimental research / learning work.
- [ ] README states this is not a full poker solver.
- [ ] README does not guarantee profitable play.
- [ ] Assumptions and limitations are linked.
- [ ] Examples are described as abstract demonstrations, not real hand
      recommendations.

### Reproducibility

- [ ] `python scripts/check_mvp.py` passes.
- [ ] Main examples run locally.
- [ ] Tests pass.
- [ ] Docs links are present.

### Publication posture

- [ ] The repository can be public as an MVP research project.
- [ ] It should not be presented as a professional solver, commercial product,
      or real-money strategy engine.
- [ ] Longer-form articles, if written later, should link back to assumptions
      and limitations.

## Commands to run

```powershell
python scripts/check_mvp.py
python -m pytest -q
```

## Manual review notes

- Review README, MVP walkthrough, examples guide, assumptions document.
- Review GitHub repository settings before switching visibility.
- Confirm the MIT License file exists and README links to it.
- Confirm README links to the publication policy
  (`docs/publication_policy.md`).
