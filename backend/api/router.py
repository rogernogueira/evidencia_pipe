from fastapi import APIRouter

from backend.api.routes import files

api_router = APIRouter()
api_router.include_router(files.router, tags=["files"])
