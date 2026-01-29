"""API v1 路由"""
from fastapi import APIRouter
from app.api.v1.endpoints import recognition

api_router = APIRouter()
api_router.include_router(recognition.router, tags=["recognition"])

