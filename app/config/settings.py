"""
应用配置
"""
import os
from pathlib import Path
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # 兼容旧版本
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    APP_NAME: str = "车牌识别 API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 检测器配置
    DETECTOR_MODEL_PATH: str = "models/det/yolo11n-pose-LPR-best.pt"
    DETECTOR_CONF_THRESHOLD: float = 0.25
    DETECTOR_IOU_THRESHOLD: float = 0.45
    DETECTOR_USE_GPU: bool = True
    DETECTOR_STOP_ON_FIRST_DETECTION: bool = False
    
    # OCR 配置
    OCR_LANG: str = "ch"
    OCR_USE_GPU: bool = True
    OCR_USE_DOC_ORIENTATION_CLASSIFY: bool = False
    OCR_USE_DOC_UNWARPING: bool = False
    OCR_USE_TEXTLINE_ORIENTATION: bool = False
    OCR_ENABLE_MKLDNN: bool = False
    OCR_USE_TENSORRT: bool = False
    
    # Pipeline 配置
    DEFAULT_ANGLES: List[float] = [0, 90, 180, 270]
    DEFAULT_USE_BEST_DETECTION: bool = True
    DEFAULT_TARGET_WIDTH: Optional[int] = None
    DEFAULT_TARGET_HEIGHT: Optional[int] = None
    KEYPOINT_EXPAND_RATIO: float = 0.2
    KEYPOINT_EXPAND_RATIO_FIRST: float = 0.1
    KEYPOINT_EXPAND_PIXELS: Optional[int] = None
    USE_CONCATENATED_IMAGE: bool = True
    ENABLE_FALLBACK_RECOGNITION: bool = False
    
    # 日志配置
    ENABLE_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS 配置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置实例
settings = Settings()

