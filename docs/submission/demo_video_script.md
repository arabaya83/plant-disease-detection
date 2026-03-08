# Demo Video Script and Runbook (5-8 Minutes)

## Objective
Demonstrate a complete end-to-end CV workflow with required scenarios:
1. Healthy leaf
2. Single diseased leaf
3. Multi-leaf mixed image
4. Invalid/no-leaf image

## Timing Plan (Target 7:00)
- 0:00-0:30 Intro and problem
- 0:30-1:10 Architecture and workflow overview
- 1:10-2:00 Scenario 1 (healthy)
- 2:00-2:50 Scenario 2 (single diseased)
- 2:50-4:00 Scenario 3 (multi-leaf mixed)
- 4:00-4:40 Scenario 4 (invalid/no leaf)
- 4:40-5:30 Grad-CAM explanation segment
- 5:30-6:20 Metrics and selected model rationale
- 6:20-7:00 Limitations and next steps

## Narration Sequence
### Opening (0:00-0:30)
- "This system helps small farmers diagnose plant diseases from mobile photos."
- "It combines leaf validation, segmentation, classification, and Grad-CAM explainability in one API workflow."

### Workflow Overview (0:30-1:10)
- Show system flow diagram.
- Mention browser-side capture/upload and server-side inference.

## Scenario Scripts and Expected Outputs
### Scenario 1: Healthy Leaf (1:10-2:00)
Action:
- Capture/upload a healthy leaf image.
Expected output:
- status `ok`
- disease shown as `No disease detected`
- confidence >= threshold
- Grad-CAM heatmap shown
Narration:
- "Healthy predictions are rendered as 'Healthy - No disease detected'."

### Scenario 2: Single Diseased Leaf (2:00-2:50)
Action:
- Upload single diseased leaf sample.
Expected output:
- status `ok`
- crop + disease label
- confidence and short disease description
- Grad-CAM heatmap
Narration:
- "The system returns both class label and confidence, with visual explanation."

### Scenario 3: Multiple Leaves, Mixed Outcomes (2:50-4:00)
Action:
- Upload image containing multiple leaves.
Expected output:
- `total_leaves_detected` > 1
- per-leaf results may include healthy and diseased leaves together
Narration:
- "Each segmented leaf is classified independently, supporting mixed outcomes."

### Scenario 4: Invalid / No Leaf Detected (4:00-4:40)
Action:
- Upload non-leaf image or poor frame.
Expected output:
- status `invalid`
- message `No leaf detected. Please retake the photo.`
Narration:
- "The validation step prevents unreliable diagnosis on invalid images."

## Explainability Segment (4:40-5:30)
- Show one prediction with Grad-CAM.
- Narrate that highlighted regions indicate influential evidence used by model.

## Performance and Selection Segment (5:30-6:20)
- Present model comparison summary table.
- State v1 deployment choice: MobileNetV2.

## Close (6:20-7:00)
- Limitations (domain shift, segmentation sensitivity).
- Next steps (field-data adaptation, calibration, mobile optimization).

## Live Demo Fallback Plan
If live camera capture fails:
1. Switch to pre-selected upload samples immediately.
2. Use saved JSON responses and screenshots from `docs/reports/figures/`.
3. Continue narration using expected-output checkpoints.

## Pre-Recording Checklist
- [ ] API starts successfully (`/health` returns OK)
- [ ] Camera capture works on target device
- [ ] Upload fallback tested
- [ ] All four scenarios prepared as sample files
- [ ] At least one Grad-CAM output verified
- [ ] Model comparison slide ready
- [ ] Audio quality check completed
- [ ] Runtime between 5 and 8 minutes
- [ ] Final video upload destination confirmed
