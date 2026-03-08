# Pre-Submission Checklist

- [x] Source code is in repo
- [x] Training scripts are in `ml/src/training`
- [x] Evaluation pipeline is in `ml/src/evaluation`
- [x] Model comparison docs are in `docs/reports`
- [ ] Generate/refresh runtime evidence figures and metrics:
  `python docs/reports/scripts/generate_project_evidence.py`
- [ ] Export technical synopsis PDF:
  `bash docs/submission/scripts/export_synopsis_pdf.sh`
- [ ] Export management PPTX:
  `bash docs/submission/scripts/export_presentation_pptx.sh`
- [ ] Build/publish Jupyter Book:
  `bash docs/submission/scripts/build_jupyter_book.sh`
- [ ] Record and upload demo video using:
  `docs/submission/demo_video_script.md`
- [ ] Fill `docs/submission/published_links.md` with final URLs
- [ ] Final consistency pass on `README.md` and submission docs
