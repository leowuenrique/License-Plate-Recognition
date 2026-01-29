"""
FastAPI 应用主入口
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.v1 import api_router
from app.dependencies import get_pipeline

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化 Pipeline
    logger.info("应用启动中...")
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline 初始化完成")
    except Exception as e:
        logger.error(f"Pipeline 初始化失败: {str(e)}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("应用关闭中...")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    description="基于关键点的车牌识别服务",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    debug=settings.DEBUG
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 注册路由
app.include_router(api_router)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recognize": "/api/v1/recognize",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        pipeline = get_pipeline()
        return {"status": "healthy", "message": "Service is running"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

