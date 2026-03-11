"""Runtime service layer for the deployed diagnosis application.

The service modules encapsulate the core steps of online inference: file
storage, basic image validation, classical segmentation, model prediction,
Grad-CAM explainability, and analytics aggregation. Keeping these concerns in
separate modules makes the API routes easier to read and safer to extend.
"""
