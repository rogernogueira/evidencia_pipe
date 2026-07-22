from fastapi import APIRouter

from backend.api.routes import artifacts, files, gpu

api_router = APIRouter()
api_router.include_router(files.router, tags=["files"])
api_router.include_router(gpu.router)
api_router.include_router(artifacts.router)
