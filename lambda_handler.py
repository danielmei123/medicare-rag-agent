"""Lambda entry point - wraps the FastAPI app."""

from mangum import Mangum

from app import app

handler = Mangum(app, lifespan="off")