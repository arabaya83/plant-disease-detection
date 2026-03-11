"""Application package for the deployed plant-disease detection service.

This package contains the FastAPI entrypoint, HTTP routes, request/response
schemas, and runtime services used by the browser-based diagnosis workflow.
The modules under :mod:`app` form the online inference stack that consumes
trained model artifacts produced by the code in :mod:`ml`.
"""
