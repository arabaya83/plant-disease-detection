# Problem and Scope

Small farmers often need immediate in-field disease assessment but lack timely expert access. This project provides a mobile-first web prototype that can:
- capture/upload leaf images,
- validate leaf presence,
- segment up to 5 leaves,
- classify each leaf as Crop + Disease,
- generate Grad-CAM overlays,
- return actionable messages when confidence is low.

Constraints in this version:
- Dataset: PlantVillage only
- English only
- Confidence threshold: 70%
