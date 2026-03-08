# Submission Completion Status

| Artifact | Current Status | Location | How To Finalize |
|---|---|---|---|
| Source code | Complete | `repo` | Keep pushing commits to `main` |
| Training scripts | Complete | `ml/src/training` | N/A |
| Evaluation pipeline | Complete | `ml/src/evaluation` | N/A |
| Model comparison | Complete | `docs/reports` | Refresh with latest metrics if retrained |
| Figures | Partial (runtime captures remain) | `docs/reports/figures` | `python docs/reports/scripts/generate_project_evidence.py` |
| Jupyter Book | Built locally (publish pending) | `docs/submission/jupyter_book/_build/html/index.html` | Publish `_build/html` via GitHub Pages or static hosting |
| Demo video | Script/checklist complete (recording pending) | `docs/submission/demo_video_script.md` | Record 5–8 min demo and publish link |
| PPTX deck | Content complete (PPTX export pending) | `docs/submission/presentation` | `bash docs/submission/scripts/export_presentation_pptx.sh` |
| Synopsis PDF | Source complete (PDF export pending) | `docs/submission/technical_synopsis.md` | `bash docs/submission/scripts/export_synopsis_pdf.sh` |

## Human Actions Remaining

1. Run evidence generation to replace any placeholder/TODO figure stubs.
2. Export `technical_synopsis.pdf` and `management_presentation.pptx`.
3. Publish Jupyter Book HTML output from `docs/submission/jupyter_book/_build/html`.
4. Record and post demo video.
5. Fill `docs/submission/published_links.md` with final URLs.
