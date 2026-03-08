# Final Artifacts Manifest

This manifest lists required final-project deliverables and their expected repository locations.

| Artifact | Path | Status Type | Notes |
|---|---|---|---|
| Project entry documentation | `README.md` | In-repo | Primary reviewer entry point |
| System flow diagram | `docs/architecture/system_flow.md` | In-repo | Mermaid flow included |
| System architecture diagram | `docs/architecture/system_architecture.md` | In-repo | Layered architecture included |
| Model comparison summary | `docs/reports/model_comparison_summary.md` | In-repo | Includes selection rationale |
| Figures package guide | `docs/reports/figures/README.md` | In-repo | Figure expectations + generation |
| Figure generation script | `docs/reports/scripts/generate_submission_figures.py` | In-repo | Reproducible chart generation |
| Compliance mapping report | `docs/submission/assignment_compliance_report.md` | In-repo | Requirement-to-evidence matrix |
| Technical synopsis source | `docs/submission/technical_synopsis.md` | In-repo | 2-page-target source |
| Technical synopsis PDF | `docs/submission/technical_synopsis.pdf` | Human-export | Must be generated manually |
| Demo video script/runbook | `docs/submission/demo_video_script.md` | In-repo | Includes fallback plan |
| Management presentation source | `docs/submission/presentation/management_presentation_content.md` | In-repo | 12-15 slide executive outline |
| Final management deck | `docs/submission/presentation/management_presentation.pptx` | Human-export | Create from source outline |
| Jupyter Book source | `docs/submission/jupyter_book/` | In-repo | `_config.yml`, `_toc.yml`, chapters |
| Jupyter Book HTML build | `docs/submission/jupyter_book/_build/html/` | Generated | Build locally before publish |
| Published Jupyter Book URL | `docs/submission/published_links.md` | Human-add | Add hosted URL |
| Demo video URL | `docs/submission/published_links.md` | Human-add | Add final recorded video URL |

## Required Human-Action Checklist
1. Export `technical_synopsis.pdf`.
2. Export `management_presentation.pptx` (and optional PDF backup).
3. Build and publish Jupyter Book, then record URL.
4. Record and publish demo video, then record URL.
5. Replace figure placeholder TODO files with real demo captures.
