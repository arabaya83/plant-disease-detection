# Submission Packaging Guide

This folder contains the missing non-code deliverables required by the final-project rubric.

## Contents
- `assignment_compliance_report.md` - requirement-by-requirement status and evidence
- `technical_synopsis.md` - two-page synopsis source
- `demo_video_script.md` - recording script and checklist
- `presentation/management_presentation_content.md` - full 12-15 slide content
- `jupyter_book/` - Jupyter Book source files

## 1) Publish Jupyter Book
From project root:

```bash
pip install jupyter-book
jupyter-book build docs/submission/jupyter_book
```

Upload the rendered `_build/html` to GitHub Pages or your institution host.

## 2) Export Technical Synopsis PDF
Recommended command (if `pandoc` is installed):

```bash
pandoc docs/submission/technical_synopsis.md -o docs/submission/technical_synopsis.pdf
```

If `pandoc` is unavailable, export from VS Code/Typora/Google Docs while preserving <=2 pages.

## 3) Produce Management Presentation File
Use `presentation/management_presentation_content.md` as the source to create:
- `management_presentation.pptx`
- optional backup `management_presentation.pdf`

## 4) Record Demo Video
Follow `demo_video_script.md`.
- Duration: 5-8 minutes
- Include required edge cases and narration
- Post final video link to Module 10 forum
