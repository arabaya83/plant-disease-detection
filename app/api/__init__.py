"""API routing package for the FastAPI deployment layer.

Modules in this package expose HTTP endpoints for liveness checks, analytics,
and end-to-end inference. Route handlers are intentionally thin and delegate
image processing, model inference, and persistence concerns to services in
``app.services``.
"""
