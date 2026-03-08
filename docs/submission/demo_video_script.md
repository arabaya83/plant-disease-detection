# Demo Video Script (5-8 Minutes)

## 0:00-0:30 Introduction
- Introduce team and project title.
- Problem in one line: small farmers need fast in-field disease diagnosis.
- Show high-level flow: mobile capture -> backend analysis -> per-leaf result + Grad-CAM.

## 0:30-1:15 System Overview
- Briefly show architecture diagram (`docs/architecture/system_architecture.md`).
- Mention integrated CV components: validation, segmentation, classification, explainability.
- Mention deployment mode: browser + FastAPI server.

## 1:15-2:15 Test Case 1 - Healthy Leaf
- Capture/upload healthy leaf.
- Show response fields: crop, `No disease detected`, confidence, Grad-CAM.
- Explain confidence threshold behavior.

## 2:15-3:15 Test Case 2 - Single Diseased Leaf
- Run diseased sample.
- Show disease class and confidence.
- Discuss short disease description output.

## 3:15-4:30 Test Case 3 - Multi-Leaf Mixed Scenario
- Use image with multiple leaves (or composite test image).
- Show per-leaf outputs separately.
- Highlight mixed outcomes (healthy + diseased).

## 4:30-5:15 Test Case 4 - Edge Case / Invalid Input
- Upload non-leaf or very poor image.
- Show expected message: `No leaf detected. Please retake the photo.`

## 5:15-6:00 Explainability Segment
- Show one Grad-CAM overlay.
- Explain that heatmap indicates influential regions for model decision.
- Clarify it is supportive evidence, not absolute causality.

## 6:00-6:45 Performance and Results
- Present final selected model and key metrics.
- Mention response-time target and observed benchmark context.
- Summarize analytics tracking (invalid rate, confidence, common diseases, latency).

## 6:45-7:30 Limitations and Next Steps
- Domain gap warning (PlantVillage vs real-field variability).
- Next steps: field data collection, domain adaptation, mobile packaging.

## 7:30-8:00 Conclusion
- Restate impact: faster triage for farmers using mobile phones.
- Close with repository and deliverables summary.

## Recording Checklist
- [ ] Audio clear, no background noise
- [ ] Show both camera capture and upload fallback
- [ ] Include all four required test scenarios
- [ ] Show at least one Grad-CAM heatmap clearly
- [ ] Show API/analytics behavior briefly
- [ ] Keep runtime between 5 and 8 minutes
- [ ] Upload link to Module 10 Demo Video Discussion Forum
