# Demo Video Script and Runbook (5-8 Minutes)

## Objective
Demonstrate complete CV integration using four required scenarios:
1. Healthy leaf
2. Single diseased leaf
3. Multi-leaf mixed image
4. Invalid/no-leaf image

## Timing Plan (Target: 7:00)
- 0:00-0:30 Introduction and project objective
- 0:30-1:10 Architecture + inference workflow
- 1:10-2:00 Scenario 1: Healthy leaf
- 2:00-2:50 Scenario 2: Single diseased leaf
- 2:50-4:00 Scenario 3: Multi-leaf mixed image
- 4:00-4:40 Scenario 4: Invalid/no-leaf
- 4:40-5:30 Grad-CAM explainability segment
- 5:30-6:20 Results and selected model rationale
- 6:20-7:00 Limitations and next steps

## Narration Sequence
### Opening
- Explain farmer pain point and the role of fast diagnosis.
- State integrated pipeline: validation -> segmentation -> classification -> Grad-CAM.

### Workflow overview
- Show `docs/architecture/system_flow.md`.
- Explain browser-side capture/upload and backend inference responsibilities.

## Scenario Scripts and Expected Outputs
### Scenario 1 — Healthy leaf
Action:
- Capture or upload healthy sample.
Expected:
- `status=ok`
- disease displayed as `No disease detected`
- confidence visible
- Grad-CAM shown

### Scenario 2 — Single diseased leaf
Action:
- Upload diseased sample.
Expected:
- `status=ok`
- crop + disease output
- confidence + short description
- Grad-CAM shown

### Scenario 3 — Multi-leaf mixed
Action:
- Upload image with multiple leaves.
Expected:
- `total_leaves_detected > 1`
- per-leaf outputs with possible mixed healthy/diseased status

### Scenario 4 — Invalid / no leaf
Action:
- Upload non-leaf or poor-quality sample.
Expected:
- `status=invalid`
- message: `No leaf detected. Please retake the photo.`

## Explainability Segment
- Show one Grad-CAM overlay.
- Explain highlighted regions as decision-support evidence.

## Performance Segment
- Show `docs/reports/model_comparison_summary.md`.
- State deployment default: **MobileNetV2**.
- Optional backup reference: `docs/reports/selected_model_rationale.md`.

## Live Demo Fallback Plan
If camera capture fails:
1. Use prepared upload samples.
2. Show stored outputs from `docs/reports/figures/`.
3. Continue narration using expected-output checkpoints.

## Pre-Recording Checklist
- [ ] API `/health` responds successfully
- [ ] Camera and upload fallback both tested
- [ ] Four scenario sample inputs prepared
- [ ] At least one Grad-CAM output verified
- [ ] Model comparison slide ready
- [ ] Audio quality verified
- [ ] Duration stays between 5 and 8 minutes
- [ ] Upload destination confirmed
