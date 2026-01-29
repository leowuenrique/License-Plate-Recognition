#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车牌识别 API 服务
基于 FastAPI 提供 RESTful API 接口
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import io
import os
import logging
from pathlib import Path
import uvicorn

from lpr4_kp import LicensePlatePipelineKP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(
    title="车牌识别 API",
    description="基于关键点的车牌识别服务",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量：Pipeline 实例
pipeline: Optional[LicensePlatePipelineKP] = None


def init_pipeline():
    """初始化 Pipeline"""
    global pipeline
    if pipeline is None:
        try:
            # 从环境变量读取配置，如果没有则使用默认值
            detector_model_path = os.getenv(
                'DETECTOR_MODEL_PATH',
                'models/det/yolo11n-pose-LPR-best.pt'
            )
            detector_conf_threshold = float(os.getenv('DETECTOR_CONF_THRESHOLD', '0.25'))
            detector_iou_threshold = float(os.getenv('DETECTOR_IOU_THRESHOLD', '0.45'))
            detector_use_gpu = os.getenv('DETECTOR_USE_GPU', 'true').lower() == 'true'
            ocr_use_gpu = os.getenv('OCR_USE_GPU', 'true').lower() == 'true'
            enable_logging = os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
            use_concatenated_image = os.getenv('USE_CONCATENATED_IMAGE', 'true').lower() == 'true'
            enable_fallback = os.getenv('ENABLE_FALLBACK', 'false').lower() == 'true'
            
            logger.info("正在初始化车牌识别 Pipeline...")
            logger.info(f"检测模型路径: {detector_model_path}")
            logger.info(f"检测器使用GPU: {detector_use_gpu}")
            logger.info(f"OCR使用GPU: {ocr_use_gpu}")
            
            # 在容器环境中禁用可能导致段错误的功能
            ocr_use_doc_orientation = os.getenv('OCR_USE_DOC_ORIENTATION_CLASSIFY', 'false').lower() == 'true'
            ocr_use_doc_unwarping = os.getenv('OCR_USE_DOC_UNWARPING', 'false').lower() == 'true'
            ocr_use_textline_orientation = os.getenv('OCR_USE_TEXTLINE_ORIENTATION', 'false').lower() == 'true'
            ocr_enable_mkldnn = os.getenv('OCR_ENABLE_MKLDNN', 'false').lower() == 'true'
            
            pipeline = LicensePlatePipelineKP(
                detector_model_path=detector_model_path,
                detector_conf_threshold=detector_conf_threshold,
                detector_iou_threshold=detector_iou_threshold,
                detector_use_gpu=detector_use_gpu,
                ocr_use_gpu=ocr_use_gpu,
                ocr_use_doc_orientation_classify=ocr_use_doc_orientation,
                ocr_use_doc_unwarping=ocr_use_doc_unwarping,
                ocr_use_textline_orientation=ocr_use_textline_orientation,
                ocr_enable_mkldnn=ocr_enable_mkldnn,
                enable_logging=enable_logging,
                use_concatenated_image=use_concatenated_image,
                enable_fallback_recognition=enable_fallback
            )
            logger.info("Pipeline 初始化完成")
        except Exception as e:
            logger.error(f"Pipeline 初始化失败: {str(e)}")
            raise
    return pipeline


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化 Pipeline"""
    init_pipeline()


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "service": "车牌识别 API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recognize": "/api/v1/recognize",
            "recognize_batch": "/api/v1/recognize_batch",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        if pipeline is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Pipeline not initialized"}
            )
        return {"status": "healthy", "message": "Service is running"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """从字节流读取图像"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")


def save_uploaded_image(image_bytes: bytes, filename: str = None) -> str:
    """保存上传的图像到临时文件"""
    if filename is None:
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
    else:
        temp_dir = Path('/tmp/uploaded_images')
        temp_dir.mkdir(exist_ok=True)
        temp_path = str(temp_dir / filename)
    
    with open(temp_path, 'wb') as f:
        f.write(image_bytes)
    return temp_path


class RecognizeRequest(BaseModel):
    """识别请求模型"""
    image_path: Optional[str] = None
    angles: Optional[List[float]] = None
    use_best_detection: Optional[bool] = None
    target_width: Optional[int] = None
    target_height: Optional[int] = None


@app.post("/api/v1/recognize")
async def recognize_plate(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    angles: Optional[str] = Form(None),
    use_best_detection: Optional[str] = Form(None),
    target_width: Optional[str] = Form(None),
    target_height: Optional[str] = Form(None)
):
    """
    识别单张车牌图像
    
    参数:
    - file: 上传的图像文件（multipart/form-data）
    - image_path: 服务器上的图像路径（如果提供file则忽略）
    - angles: 检测角度列表，格式: "0,90,180,270"（可选）
    - use_best_detection: 是否使用最佳检测，格式: "true" 或 "false"（可选）
    - target_width: 透视变换目标宽度（可选）
    - target_height: 透视变换目标高度（可选）
    """
    try:
        # 确保 Pipeline 已初始化
        pipeline = init_pipeline()
        
        # 解析参数
        angles_list = None
        if angles:
            try:
                angles_list = [float(x.strip()) for x in angles.split(',')]
            except:
                angles_list = None
        
        use_best = None
        if use_best_detection:
            use_best = use_best_detection.lower() == 'true'
        
        # 安全地转换 target_width 和 target_height
        target_w = None
        if target_width:
            try:
                target_w = int(target_width)
            except (ValueError, TypeError):
                # 如果转换失败，忽略该参数
                pass
        
        target_h = None
        if target_height:
            try:
                target_h = int(target_height)
            except (ValueError, TypeError):
                # 如果转换失败，忽略该参数
                pass
        
        # 确定图像来源
        temp_path = None
        try:
            if file:
                # 从上传的文件读取
                image_bytes = await file.read()
                image = read_image_from_bytes(image_bytes)
                # 保存到临时文件
                temp_path = save_uploaded_image(image_bytes, file.filename)
                image_path_to_use = temp_path
            elif image_path:
                # 使用提供的路径
                if not os.path.exists(image_path):
                    raise HTTPException(status_code=404, detail=f"图像文件不存在: {image_path}")
                image_path_to_use = image_path
            else:
                raise HTTPException(status_code=400, detail="必须提供图像文件或图像路径")
            
            # 执行识别
            logger.info(f"开始识别: {image_path_to_use}")
            result = pipeline.recognize(
                image_path=image_path_to_use,
                angles=angles_list,
                use_best_detection=use_best,
                target_width=target_w,
                target_height=target_h
            )
            
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # 转换numpy数组为可序列化的格式
            response_data = {
                "success": result['success'],
                "text": result['text'],
                "raw_text": result['raw_text'],
                "confidence": float(result['confidence']),
                "ocr_confidence": float(result.get('ocr_confidence', 0.0)),
                "is_valid_plate": result['is_valid_plate'],
                "detection_count": result['detection_count'],
                "details": {
                    "texts": result['details'].get('texts', []),
                    "scores": [float(s) for s in result['details'].get('scores', [])],
                    "detection_score": float(result['details'].get('detection_score', 0.0)),
                    "corrected": result['details'].get('corrected', False),
                    "original_text": result['details'].get('original_text', ''),
                    "is_valid_plate": result['details'].get('is_valid_plate', False),
                    "fallback_used": result['details'].get('fallback_used', False)
                }
            }
            
            # 如果包含warped_image，可以选择返回base64编码（可选）
            if result.get('warped_image') is not None:
                # 不返回图像数据，只返回标识
                response_data['warped_image_available'] = True
            else:
                response_data['warped_image_available'] = False
            
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"识别失败: {str(e)}", exc_info=True)
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")
            
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/recognize_batch")
async def recognize_plate_batch(
    files: List[UploadFile] = File(...),
    angles: Optional[str] = Form(None),
    use_best_detection: Optional[str] = Form(None)
):
    """
    批量识别多张车牌图像
    
    参数:
    - files: 上传的图像文件列表（multipart/form-data）
    - angles: 检测角度列表，格式: "0,90,180,270"（可选）
    - use_best_detection: 是否使用最佳检测，格式: "true" 或 "false"（可选）
    """
    try:
        # 确保 Pipeline 已初始化
        pipeline = init_pipeline()
        
        # 解析参数
        angles_list = None
        if angles:
            try:
                angles_list = [float(x.strip()) for x in angles.split(',')]
            except:
                angles_list = None
        
        use_best = None
        if use_best_detection:
            use_best = use_best_detection.lower() == 'true'
        
        results = []
        temp_paths = []
        
        try:
            for file in files:
                temp_path = None
                try:
                    # 读取上传的文件
                    image_bytes = await file.read()
                    image = read_image_from_bytes(image_bytes)
                    # 保存到临时文件
                    temp_path = save_uploaded_image(image_bytes, file.filename)
                    temp_paths.append(temp_path)
                    
                    # 执行识别
                    logger.info(f"开始识别: {file.filename}")
                    result = pipeline.recognize(
                        image_path=temp_path,
                        angles=angles_list,
                        use_best_detection=use_best
                    )
                    
                    # 转换结果
                    result_data = {
                        "filename": file.filename,
                        "success": result['success'],
                        "text": result['text'],
                        "raw_text": result['raw_text'],
                        "confidence": float(result['confidence']),
                        "ocr_confidence": float(result.get('ocr_confidence', 0.0)),
                        "is_valid_plate": result['is_valid_plate'],
                        "detection_count": result['detection_count'],
                        "details": {
                            "texts": result['details'].get('texts', []),
                            "scores": [float(s) for s in result['details'].get('scores', [])],
                            "detection_score": float(result['details'].get('detection_score', 0.0)),
                            "corrected": result['details'].get('corrected', False),
                            "original_text": result['details'].get('original_text', ''),
                            "is_valid_plate": result['details'].get('is_valid_plate', False),
                            "fallback_used": result['details'].get('fallback_used', False)
                        }
                    }
                    results.append(result_data)
                    
                except Exception as e:
                    logger.error(f"处理文件 {file.filename} 失败: {str(e)}")
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "total": len(results),
                "results": results
            }
            
        finally:
            # 清理临时文件
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
    except Exception as e:
        logger.error(f"批量识别失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量识别失败: {str(e)}")


if __name__ == "__main__":
    # 开发环境直接运行
    import sys
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,  # 生产环境设为False
        log_level="info"
    )

