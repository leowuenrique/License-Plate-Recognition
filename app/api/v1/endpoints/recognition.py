"""
车牌识别 API 端点
"""
import os
import logging
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends

from app.core.pipeline import LicensePlatePipelineKP
from app.dependencies import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.post("/recognize")
async def recognize_plate(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    angles: Optional[str] = Form(None),
    use_best_detection: Optional[str] = Form(None),
    target_width: Optional[str] = Form(None),
    target_height: Optional[str] = Form(None),
    pipeline: LicensePlatePipelineKP = Depends(get_pipeline)
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
                pass
        
        target_h = None
        if target_height:
            try:
                target_h = int(target_height)
            except (ValueError, TypeError):
                pass
        
        # 确定图像来源
        temp_path = None
        try:
            if file:
                image_bytes = await file.read()
                image = read_image_from_bytes(image_bytes)
                temp_path = save_uploaded_image(image_bytes, file.filename)
                image_path_to_use = temp_path
            elif image_path:
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
            
            # 转换结果
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
            
            if result.get('warped_image') is not None:
                response_data['warped_image_available'] = True
            else:
                response_data['warped_image_available'] = False
            
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"识别失败: {str(e)}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")
            
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


