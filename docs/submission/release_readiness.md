# Release Readiness Guidance (Optional but Recommended)

## 1) Create Final Tag
After final verification:
```bash
git tag -a v1.0-submission -m "Final course submission snapshot"
git push origin v1.0-submission
```

## 2) Optional Release Notes
Include in GitHub release:
- selected model and rationale,
- evidence package status,
- known limitations,
- links to Jupyter Book/demo/presentation.

## 3) Optional Zip Packaging Plan for LMS
Recommended bundle content:
- `README.md`
- `docs/architecture/`
- `docs/reports/`
- `docs/submission/`
- `app/`, `ml/src/`

Exclude from zip by default (unless explicitly requested):
- `.venv/`
- `ml/datasets/`
- large model binaries not required by rubric upload limits
