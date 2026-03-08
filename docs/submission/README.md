# Submission Package Guide

This folder contains final-project submission support artifacts.

## Core Files
- `assignment_compliance_report.md` - requirement-to-evidence matrix
- `technical_synopsis.md` - 2-page-target synopsis source
- `demo_video_script.md` - 5-8 minute demo runbook
- `presentation/management_presentation_content.md` - executive deck source outline
- `final_artifacts_manifest.md` - expected final deliverable locations
- `published_links.md` - fill with final hosted links

## Jupyter Book Source
- `jupyter_book/` contains `_config.yml`, `_toc.yml`, and chapter pages.

Build command:
```bash
pip install jupyter-book
jupyter-book build docs/submission/jupyter_book
```

## Export and Publish Checklist
1. Export `technical_synopsis.pdf` from `technical_synopsis.md`.
2. Export `management_presentation.pptx` from presentation source.
3. Publish Jupyter Book and record URL in `published_links.md`.
4. Record and publish demo video and record URL in `published_links.md`.
5. Ensure figure placeholders in `docs/reports/figures/` are replaced with real evidence captures.
