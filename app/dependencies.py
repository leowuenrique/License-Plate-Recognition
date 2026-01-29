"""
依赖注入模块
"""
import logging
from typing import Optional
from app.core.pipeline import LicensePlatePipelineKP
from app.config import settings

logger = logging.getLogger(__name__)

# 全局 Pipeline 实例
_pipeline: Optional[LicensePlatePipelineKP] = None


def get_pipeline() -> LicensePlatePipelineKP:
    """获取 Pipeline 实例（单例模式）"""
    global _pipeline
    if _pipeline is None:
        try:
            logger.info("正在初始化车牌识别 Pipeline...")
            logger.info(f"检测模型路径: {settings.DETECTOR_MODEL_PATH}")
            logger.info(f"检测器使用GPU: {settings.DETECTOR_USE_GPU}")
            logger.info(f"OCR使用GPU: {settings.OCR_USE_GPU}")
            
            _pipeline = LicensePlatePipelineKP(
                detector_model_path=settings.DETECTOR_MODEL_PATH,
                detector_conf_threshold=settings.DETECTOR_CONF_THRESHOLD,
                detector_iou_threshold=settings.DETECTOR_IOU_THRESHOLD,
                detector_use_gpu=settings.DETECTOR_USE_GPU,
                detector_stop_on_first_detection=settings.DETECTOR_STOP_ON_FIRST_DETECTION,
                ocr_lang=settings.OCR_LANG,
                ocr_use_gpu=settings.OCR_USE_GPU,
                ocr_use_doc_orientation_classify=settings.OCR_USE_DOC_ORIENTATION_CLASSIFY,
                ocr_use_doc_unwarping=settings.OCR_USE_DOC_UNWARPING,
                ocr_use_textline_orientation=settings.OCR_USE_TEXTLINE_ORIENTATION,
                ocr_enable_mkldnn=settings.OCR_ENABLE_MKLDNN,
                ocr_use_tensorrt=settings.OCR_USE_TENSORRT,
                default_angles=settings.DEFAULT_ANGLES,
                default_use_best_detection=settings.DEFAULT_USE_BEST_DETECTION,
                default_target_width=settings.DEFAULT_TARGET_WIDTH,
                default_target_height=settings.DEFAULT_TARGET_HEIGHT,
                keypoint_expand_ratio=settings.KEYPOINT_EXPAND_RATIO,
                keypoint_expand_ratio_first=settings.KEYPOINT_EXPAND_RATIO_FIRST,
                keypoint_expand_pixels=settings.KEYPOINT_EXPAND_PIXELS,
                use_concatenated_image=settings.USE_CONCATENATED_IMAGE,
                enable_logging=settings.ENABLE_LOGGING,
                enable_fallback_recognition=settings.ENABLE_FALLBACK_RECOGNITION
            )
            logger.info("Pipeline 初始化完成")
        except Exception as e:
            logger.error(f"Pipeline 初始化失败: {str(e)}", exc_info=True)
            raise
    return _pipeline

