"""
车牌识别 Pipeline（基于关键点的透视变换）
整合检测、透视变换和OCR识别
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from detector import LicensePlatePoseDetector, select_highest_confidence
from utils import (perspective_transform_plate, get_best_perspective_transform, expand_keypoints,
                   correct_license_plate_text_v4, create_concatenated_image_for_recognition,
                   select_center_detection_from_detector_result, validate_license_plate)
from rec import PaddleOCRRecognition


class LicensePlatePipelineKP:
    """车牌识别 Pipeline（基于关键点）"""
    
    def __init__(self,
                 # 检测器参数
                 detector_model_path: str,
                 detector_conf_threshold: float = 0.25,
                 detector_iou_threshold: float = 0.45,
                 detector_use_gpu: bool = True,
                 detector_stop_on_first_detection: bool = False,
                 # OCR识别器参数
                 ocr_lang: str = "ch",
                 ocr_use_doc_orientation_classify: bool = False,
                 ocr_use_doc_unwarping: bool = False,
                 ocr_use_textline_orientation: bool = True,
                 ocr_enable_mkldnn: bool = True,
                 ocr_use_tensorrt: bool = False,
                 ocr_use_gpu: bool = True,
                 ocr_kwargs: Optional[Dict] = None,
                 # 识别方法默认参数
                 default_angles: Optional[List[float]] = None,
                 default_use_best_detection: bool = True,
                 default_target_width: Optional[int] = None,
                 default_target_height: Optional[int] = None,
                 # 关键点扩展参数
                 keypoint_expand_ratio: float = 0.2,
                 keypoint_expand_ratio_first: float = 0.1,
                 keypoint_expand_pixels: Optional[int] = None,
                # 其他参数
                use_concatenated_image: bool = True,
                enable_logging: bool = True,
                log_level: int = logging.INFO,
                enable_fallback_recognition: bool = False):
        """
        初始化车牌识别 Pipeline
        
        Args:
            # 检测器参数
            detector_model_path: YOLO11-pose 检测模型路径
            detector_conf_threshold: 检测器置信度阈值
            detector_iou_threshold: 检测器IoU阈值
            detector_use_gpu: 检测器是否使用GPU
            detector_stop_on_first_detection: 检测器是否在第一次检测到车牌后停止
            
            # OCR识别器参数
            ocr_lang: OCR语言类型，默认 "ch"（中文）
            ocr_use_doc_orientation_classify: 是否使用文档方向分类
            ocr_use_doc_unwarping: 是否使用文档反扭曲
            ocr_use_textline_orientation: 是否使用文本行方向检测
            ocr_enable_mkldnn: 是否启用 Intel MKL-DNN 加速
            ocr_use_tensorrt: 是否使用 TensorRT
            ocr_use_gpu: OCR是否使用GPU
            ocr_kwargs: OCR其他初始化参数（字典格式）
            
            # 识别方法默认参数
            default_angles: 默认检测角度列表，如果为None则使用 [0, 90, 180, 270]
            default_use_best_detection: 默认是否只使用置信度最高的检测结果
            default_target_width: 默认透视变换目标宽度（如果为None，则根据关键点自动计算）
            default_target_height: 默认透视变换目标高度（如果为None，则根据关键点自动计算）
            
            # 关键点扩展参数
            keypoint_expand_ratio: 最佳检测结果的关键点扩展比例（默认0.2，即20%）
            keypoint_expand_ratio_first: 第一个检测结果的关键点扩展比例（默认0.1，即10%）
            keypoint_expand_pixels: 固定像素扩展值，如果指定则优先使用此值（None表示使用比例扩展）
            
            # 其他参数
            use_concatenated_image: 是否在识别前拼接180度图像（提高识别准确率）
            enable_logging: 是否启用日志
            log_level: 日志级别
            enable_fallback_recognition: 是否启用容错机制（当检测失败或识别失败时，尝试对整张图片进行识别），默认False
        """
        self.enable_logging = enable_logging
        
        # 保存参数到实例变量
        # 检测器参数
        self.detector_model_path = detector_model_path
        self.detector_conf_threshold = detector_conf_threshold
        self.detector_iou_threshold = detector_iou_threshold
        self.detector_use_gpu = detector_use_gpu
        self.detector_stop_on_first_detection = detector_stop_on_first_detection
        
        # OCR识别器参数
        self.ocr_lang = ocr_lang
        self.ocr_use_doc_orientation_classify = ocr_use_doc_orientation_classify
        self.ocr_use_doc_unwarping = ocr_use_doc_unwarping
        self.ocr_use_textline_orientation = ocr_use_textline_orientation
        self.ocr_enable_mkldnn = ocr_enable_mkldnn
        self.ocr_use_tensorrt = ocr_use_tensorrt
        self.ocr_use_gpu = ocr_use_gpu
        self.ocr_kwargs = ocr_kwargs or {}
        
        # 识别方法默认参数
        self.default_angles = default_angles if default_angles is not None else [0, 90, 180, 270]
        self.default_use_best_detection = default_use_best_detection
        self.default_target_width = default_target_width
        self.default_target_height = default_target_height
        
        # 关键点扩展参数
        self.keypoint_expand_ratio = keypoint_expand_ratio
        self.keypoint_expand_ratio_first = keypoint_expand_ratio_first
        self.keypoint_expand_pixels = keypoint_expand_pixels
        
        # 其他参数
        self.use_concatenated_image = use_concatenated_image
        self.enable_fallback_recognition = enable_fallback_recognition
        
        # 配置日志
        if enable_logging:
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            self.logger.setLevel(log_level)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            
            # 配置 utils 模块的文本校正日志
            logger_v4 = logging.getLogger('utils')
            logger_v4.setLevel(log_level)
            if not logger_v4.handlers:
                handler_v4 = logging.StreamHandler()
                handler_v4.setLevel(log_level)
                formatter_v4 = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler_v4.setFormatter(formatter_v4)
                logger_v4.addHandler(handler_v4)
        else:
            self.logger = None
        
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("初始化车牌识别 Pipeline（基于关键点）")
            self.logger.info("=" * 80)
        
        # 初始化检测器
        if self.logger:
            self.logger.info("[1/2] 初始化检测器...")
        self.detector = LicensePlatePoseDetector(
            model_path=detector_model_path,
            conf_threshold=detector_conf_threshold,
            iou_threshold=detector_iou_threshold,
            use_gpu=detector_use_gpu,
            stop_on_first_detection=detector_stop_on_first_detection
        )
        if self.logger:
            self.logger.info(f"检测器初始化完成，置信度阈值: {detector_conf_threshold}, IoU阈值: {detector_iou_threshold}, "
                           f"使用GPU: {detector_use_gpu}, 检测到车牌即停止: {detector_stop_on_first_detection}")
        
        # 初始化OCR识别器
        if self.logger:
            self.logger.info("[2/2] 初始化识别器...")
        self.ocr = PaddleOCRRecognition(
            lang=self.ocr_lang,
            use_doc_orientation_classify=self.ocr_use_doc_orientation_classify,
            use_doc_unwarping=self.ocr_use_doc_unwarping,
            use_textline_orientation=self.ocr_use_textline_orientation,
            enable_mkldnn=self.ocr_enable_mkldnn,
            use_tensorrt=self.ocr_use_tensorrt,
            use_gpu=self.ocr_use_gpu,
            **self.ocr_kwargs
        )
        if self.logger:
            self.logger.info(f"OCR识别器初始化完成，语言: {self.ocr_lang}, 使用GPU: {self.ocr_use_gpu}")
            self.logger.info(f"默认检测角度: {self.default_angles}")
            self.logger.info(f"默认使用最佳检测: {self.default_use_best_detection}")
            self.logger.info(f"默认目标尺寸: {self.default_target_width}x{self.default_target_height}" if self.default_target_width or self.default_target_height else "默认目标尺寸: 自动计算")
            self.logger.info(f"关键点扩展比例（最佳）: {self.keypoint_expand_ratio}, （第一个）: {self.keypoint_expand_ratio_first}")
            self.logger.info(f"拼接180度图像: {self.use_concatenated_image}")
            self.logger.info("=" * 80)
    
    def recognize(self,
                  image_path: str,
                  angles: Optional[List[float]] = None,
                  use_best_detection: Optional[bool] = None,
                  target_width: Optional[int] = None,
                  target_height: Optional[int] = None) -> Dict:
        """
        完整的车牌识别流程：检测 -> 透视变换 -> 拼接180度图像（可选） -> OCR识别 -> 文本校正
        
        Args:
            image_path: 图像路径
            angles: 检测角度列表，如果为None则使用类初始化时设置的默认角度
            use_best_detection: 是否只使用置信度最高的检测结果，如果为None则使用类初始化时的默认值
            target_width: 透视变换目标宽度（如果为None，则使用类初始化时的默认值或根据关键点自动计算）
            target_height: 透视变换目标高度（如果为None，则使用类初始化时的默认值或根据关键点自动计算）
        
        Returns:
            识别结果字典：
            {
                'success': bool,  # 是否成功识别
                'text': str or None,  # 校正后的车牌文本（如果验证未通过则为 None）
                'raw_text': str,  # 原始OCR识别文本
                'confidence': float,  # 检测置信度
                'ocr_confidence': float,  # OCR置信度
                'is_valid_plate': bool,  # 是否通过车牌正则表达式验证
                'detection_count': int,  # 检测到的车牌数量
                'warped_image': np.ndarray,  # 透视变换后的车牌图像（可选）
                'details': Dict,  # 详细信息，包含原始文本、OCR详情等
            }
        """
        # 使用默认参数（如果未提供）
        if angles is None:
            angles = self.default_angles
        if use_best_detection is None:
            use_best_detection = self.default_use_best_detection
        if target_width is None:
            target_width = self.default_target_width
        if target_height is None:
            target_height = self.default_target_height
        
        # 步骤1: 检测车牌
        if self.logger:
            self.logger.info(f"[步骤1] 检测车牌: {image_path}")
            if angles:
                self.logger.info(f"检测角度: {angles}")
        
        detection_result = self.detector.detect(image_path, angles=angles)
        
        if detection_result['count'] == 0:
            if self.logger:
                self.logger.warning("未检测到车牌")
            
            # 如果启用了容错机制，尝试对整张图片进行识别
            if self.enable_fallback_recognition:
                if self.logger:
                    self.logger.info("容错机制已启用，尝试对整张图片进行识别...")
                
                # 读取原始图像
                image = cv2.imread(image_path)
                if image is None:
                    return {
                        'success': False,
                        'text': None,
                        'raw_text': '',
                        'confidence': 0.0,
                        'ocr_confidence': 0.0,
                        'is_valid_plate': False,
                        'detection_count': 0,
                        'warped_image': None,
                        'details': {
                            'error': '无法读取图像'
                        }
                    }
                
                # 直接使用整张图片进行识别（不进行透视变换）
                if self.logger:
                    self.logger.info("使用整张图片进行识别...")
                
                try:
                    # 可选：创建拼接图像（如果启用）
                    if self.use_concatenated_image:
                        try:
                            recognition_image = create_concatenated_image_for_recognition(image)
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"创建拼接图像失败，使用原始图像: {str(e)}")
                            recognition_image = image
                    else:
                        recognition_image = image
                    
                    # OCR识别
                    ocr_result = self.ocr(recognition_image)
                    
                    # 处理OCR结果
                    if isinstance(ocr_result, dict):
                        raw_text = ocr_result.get('text', '')
                        texts = ocr_result.get('texts', [])
                        scores = ocr_result.get('scores', [])
                        details = ocr_result.get('details', [])
                        orientation = ocr_result.get('orientation', {})
                    else:
                        raw_text = str(ocr_result) if ocr_result else ''
                        texts = [raw_text] if raw_text else []
                        scores = []
                        details = []
                        orientation = {}
                    
                    # 文本校正
                    corrected_text = ''
                    if raw_text or details:
                        if isinstance(ocr_result, dict) and ocr_result.get('texts') and ocr_result.get('details'):
                            corrected_text = correct_license_plate_text_v4(ocr_result)
                            if self.logger:
                                self.logger.info("使用文本校正函数V4（专家级融合算法）进行校正")
                    
                    final_text = corrected_text if corrected_text else raw_text
                    ocr_confidence = sum(scores) / len(scores) if scores else 0.0
                    
                    # 验证车牌格式
                    is_valid_plate = validate_license_plate(final_text) if final_text else False
                    
                    if final_text and not is_valid_plate:
                        if self.logger:
                            self.logger.warning(f"车牌格式验证未通过，预测结果设为 None: {final_text}")
                        final_text = None
                    
                    if self.logger:
                        if final_text:
                            self.logger.info(f"整张图片识别结果: {final_text}")
                        else:
                            self.logger.warning("整张图片也未识别出有效车牌")
                    
                    return {
                        'success': bool(final_text),
                        'text': final_text,
                        'raw_text': raw_text,
                        'confidence': 0.0,  # 检测失败，置信度为0
                        'ocr_confidence': ocr_confidence,
                        'is_valid_plate': is_valid_plate,
                        'detection_count': 0,
                        'warped_image': None,
                        'details': {
                            'texts': texts,
                            'scores': scores,
                            'ocr_details': details,
                            'detection_score': 0.0,
                            'corrected': bool(corrected_text),
                            'original_text': raw_text,
                            'orientation': orientation,
                            'is_valid_plate': is_valid_plate,
                            'fallback_used': True  # 标记使用了容错机制
                        }
                    }
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"整张图片识别失败: {str(e)}")
                    return {
                        'success': False,
                        'text': None,
                        'raw_text': '',
                        'confidence': 0.0,
                        'ocr_confidence': 0.0,
                        'is_valid_plate': False,
                        'detection_count': 0,
                        'warped_image': None,
                        'details': {
                            'error': f'整张图片识别失败: {str(e)}',
                            'fallback_used': True
                        }
                    }
            else:
                # 容错机制未启用，直接返回失败
                return {
                    'success': False,
                    'text': None,
                    'raw_text': '',
                    'confidence': 0.0,
                    'ocr_confidence': 0.0,
                    'is_valid_plate': False,
                    'detection_count': 0,
                    'warped_image': None,
                    'details': {
                        'error': '未检测到车牌'
                    }
                }
        
        if self.logger:
            self.logger.info(f"检测到 {detection_result['count']} 个车牌")
        
        # 步骤2: 透视变换
        if self.logger:
            self.logger.info("[步骤2] 透视变换...")
        
        if use_best_detection:
            # 使用离图像中心最近的检测结果（检测框中心点离图像中心点更近）
            # 选择方法代码在 utils.py 中
            if self.logger:
                self.logger.info("选择离图像中心最近的检测结果")
            best_detection = select_center_detection_from_detector_result(detection_result)
            if best_detection is None:
                if self.logger:
                    self.logger.warning("无法选择最佳检测结果")
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'detection_count': detection_result['count'],
                    'warped_image': None,
                    'details': {
                        'error': '无法选择最佳检测结果'
                    }
                }
            
            if self.logger:
                self.logger.info(f"选择的最佳检测结果置信度: {best_detection.get('score', 0.0):.4f}")
            
            # 找到最佳检测结果在原始检测结果中的索引
            boxes = detection_result.get('boxes', [])
            best_box = best_detection.get('box')
            best_index = None
            
            if boxes and best_box:
                for i, box in enumerate(boxes):
                    if len(box) == 4 and len(best_box) == 4:
                        if abs(box[0] - best_box[0]) < 1e-5 and abs(box[1] - best_box[1]) < 1e-5:
                            best_index = i
                            break
            
            if best_index is None:
                return {
                    'success': False,
                    'text': '',
                    'confidence': best_detection.get('score', 0.0),
                    'detection_count': detection_result['count'],
                    'warped_image': None,
                    'details': {
                        'error': '无法找到最佳检测结果的索引'
                    }
                }
            
            # 扩大关键点区域（基于关键点）
            keypoints = detection_result.get('keypoints', [])
            if keypoints and best_index < len(keypoints) and keypoints[best_index] is not None:
                # 获取对应的旋转图像尺寸
                angles = detection_result.get('angles', [])
                rotated_images = detection_result.get('rotated_images', {})
                if best_index < len(angles):
                    angle = angles[best_index]
                    if angle in rotated_images:
                        rotated_image = rotated_images[angle]
                        image_shape = rotated_image.shape[:2]  # (height, width)
                        
                        # 扩大关键点
                        if self.logger:
                            self.logger.info(f"扩大关键点区域（基于关键点，扩大比例: {self.keypoint_expand_ratio}）")
                        expanded_keypoints = expand_keypoints(
                            keypoints[best_index],
                            image_shape,
                            expand_ratio=self.keypoint_expand_ratio,
                            expand_pixels=self.keypoint_expand_pixels
                        )
                        
                        # 更新检测结果中的关键点
                        detection_result['keypoints'][best_index] = expanded_keypoints
            
            # 使用选中的最佳检测结果进行透视变换
            warped_image = get_best_perspective_transform(
                detection_result,
                target_width=target_width,
                target_height=target_height
            )
            detection_score = best_detection['score'] if best_detection else 0.0
        else:
            # 使用第一个检测结果
            if self.logger:
                self.logger.info("使用第一个检测结果")
            # 扩大关键点区域（基于关键点）
            keypoints = detection_result.get('keypoints', [])
            if keypoints and len(keypoints) > 0 and keypoints[0] is not None:
                # 获取对应的旋转图像尺寸
                angles = detection_result.get('angles', [])
                rotated_images = detection_result.get('rotated_images', {})
                if len(angles) > 0:
                    angle = angles[0]
                    if angle in rotated_images:
                        rotated_image = rotated_images[angle]
                        image_shape = rotated_image.shape[:2]  # (height, width)
                        
                        # 扩大关键点
                        if self.logger:
                            self.logger.info(f"扩大关键点区域（基于关键点，扩大比例: {self.keypoint_expand_ratio_first}）")
                        expanded_keypoints = expand_keypoints(
                            keypoints[0],
                            image_shape,
                            expand_ratio=self.keypoint_expand_ratio_first,
                            expand_pixels=self.keypoint_expand_pixels
                        )
                        
                        # 更新检测结果中的关键点
                        detection_result['keypoints'][0] = expanded_keypoints
            
            warped_image = perspective_transform_plate(
                detection_result,
                detection_index=0,
                target_width=target_width,
                target_height=target_height
            )
            detection_score = detection_result['scores'][0] if detection_result['scores'] else 0.0
        
        if warped_image is None:
            if self.logger:
                self.logger.warning("透视变换失败（可能缺少关键点）")
            return {
                'success': False,
                'text': '',
                'confidence': detection_score,
                'detection_count': detection_result['count'],
                'warped_image': None,
                'details': {
                    'error': '透视变换失败（可能缺少关键点）'
                }
            }
        
        if self.logger:
            self.logger.info(f"透视变换完成，输出尺寸: {warped_image.shape}")
        
        # 步骤3: 准备识别图像（可选：拼接180度图像）
        if self.logger:
            self.logger.info("[步骤3] 准备识别图像...")
        if self.use_concatenated_image:
            if self.logger:
                self.logger.info("创建拼接180度图像用于识别...")
            try:
                recognition_image = create_concatenated_image_for_recognition(warped_image)
                if self.logger:
                    self.logger.info(f"拼接图像完成，输出尺寸: {recognition_image.shape}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"创建拼接图像失败，使用原始图像: {str(e)}")
                # 如果拼接失败，使用原始图像
                recognition_image = warped_image
        else:
            recognition_image = warped_image
        
        # 步骤4: OCR识别
        if self.logger:
            self.logger.info("[步骤4] OCR识别...")
        ocr_result = self.ocr(recognition_image)
        
        # 处理OCR结果
        if isinstance(ocr_result, dict):
            raw_text = ocr_result.get('text', '')
            texts = ocr_result.get('texts', [])
            scores = ocr_result.get('scores', [])
            details = ocr_result.get('details', [])
            orientation = ocr_result.get('orientation', {})  # 文本行方向信息
        else:
            # 向后兼容：如果返回字符串
            raw_text = str(ocr_result) if ocr_result else ''
            texts = [raw_text] if raw_text else []
            scores = []
            details = []
            orientation = {}
        
        # 步骤5: 文本校正（参考 pipeline.py 的文本纠正处理逻辑）
        if self.logger:
            self.logger.info("[步骤5] 文本校正...")
        
        corrected_text = ''
        if raw_text or details:
            if self.logger:
                self.logger.info(f"原始识别结果: {raw_text}")
                if details:
                    self.logger.info(f"OCR详细信息数量: {len(details)}")
            
            # 使用文本校正函数V4（专家级融合算法）
            if isinstance(ocr_result, dict) and ocr_result.get('texts') and ocr_result.get('details'):
                corrected_text = correct_license_plate_text_v4(ocr_result)
                if self.logger:
                    self.logger.info("使用文本校正函数V4（专家级融合算法）进行校正")
            
            if self.logger:
                if corrected_text != raw_text:
                    self.logger.info(f"校正后结果: {corrected_text}")
                else:
                    self.logger.info(f"识别结果: {corrected_text}")
        else:
            corrected_text = ''
            if self.logger:
                self.logger.warning("未识别到文本")
        
        # 使用校正后的文本作为最终结果，如果没有校正结果则使用原始文本
        final_text = corrected_text if corrected_text else raw_text
        
        # 计算OCR置信度（如果有多个文本，取平均值）
        ocr_confidence = sum(scores) / len(scores) if scores else 0.0
        
        # 验证最终结果是否通过车牌正则表达式
        is_valid_plate = validate_license_plate(final_text) if final_text else False
        
        # 如果验证未通过，将预测结果设为 None
        if final_text and not is_valid_plate:
            if self.logger:
                self.logger.warning(f"车牌格式验证未通过，预测结果设为 None: {final_text}")
            final_text = None
        
        # 如果识别失败且启用了容错机制，尝试对整张图片进行识别
        if not final_text and self.enable_fallback_recognition:
            if self.logger:
                self.logger.info("识别失败，容错机制已启用，尝试对整张图片进行识别...")
            
            # 读取原始图像
            image = cv2.imread(image_path)
            if image is not None:
                try:
                    # 可选：创建拼接图像（如果启用）
                    if self.use_concatenated_image:
                        try:
                            recognition_image = create_concatenated_image_for_recognition(image)
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"创建拼接图像失败，使用原始图像: {str(e)}")
                            recognition_image = image
                    else:
                        recognition_image = image
                    
                    # OCR识别
                    ocr_result = self.ocr(recognition_image)
                    
                    # 处理OCR结果
                    if isinstance(ocr_result, dict):
                        fallback_raw_text = ocr_result.get('text', '')
                        fallback_texts = ocr_result.get('texts', [])
                        fallback_scores = ocr_result.get('scores', [])
                        fallback_details = ocr_result.get('details', [])
                        fallback_orientation = ocr_result.get('orientation', {})
                    else:
                        fallback_raw_text = str(ocr_result) if ocr_result else ''
                        fallback_texts = [fallback_raw_text] if fallback_raw_text else []
                        fallback_scores = []
                        fallback_details = []
                        fallback_orientation = {}
                    
                    # 文本校正
                    fallback_corrected_text = ''
                    if fallback_raw_text or fallback_details:
                        if isinstance(ocr_result, dict) and ocr_result.get('texts') and ocr_result.get('details'):
                            fallback_corrected_text = correct_license_plate_text_v4(ocr_result)
                            if self.logger:
                                self.logger.info("使用文本校正函数V4（专家级融合算法）进行校正")
                    
                    fallback_final_text = fallback_corrected_text if fallback_corrected_text else fallback_raw_text
                    fallback_ocr_confidence = sum(fallback_scores) / len(fallback_scores) if fallback_scores else 0.0
                    
                    # 验证车牌格式
                    fallback_is_valid_plate = validate_license_plate(fallback_final_text) if fallback_final_text else False
                    
                    if fallback_final_text and not fallback_is_valid_plate:
                        if self.logger:
                            self.logger.warning(f"车牌格式验证未通过，预测结果设为 None: {fallback_final_text}")
                        fallback_final_text = None
                    
                    if self.logger:
                        if fallback_final_text:
                            self.logger.info(f"整张图片识别结果: {fallback_final_text}")
                        else:
                            self.logger.warning("整张图片也未识别出有效车牌")
                    
                    # 如果容错识别成功，使用容错结果
                    if fallback_final_text:
                        return {
                            'success': True,
                            'text': fallback_final_text,
                            'raw_text': fallback_raw_text,
                            'confidence': detection_score,  # 保留原始检测置信度
                            'ocr_confidence': fallback_ocr_confidence,
                            'is_valid_plate': fallback_is_valid_plate,
                            'detection_count': detection_result['count'],
                            'warped_image': warped_image,  # 保留原始透视变换图像
                            'details': {
                                'texts': fallback_texts,
                                'scores': fallback_scores,
                                'ocr_details': fallback_details,
                                'detection_score': detection_score,
                                'corrected': bool(fallback_corrected_text),
                                'original_text': fallback_raw_text,
                                'orientation': fallback_orientation,
                                'is_valid_plate': fallback_is_valid_plate,
                                'fallback_used': True  # 标记使用了容错机制
                            }
                        }
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"整张图片识别失败: {str(e)}")
        
        if self.logger:
            if final_text:
                self.logger.info(f"最终识别结果: {final_text}")
                self.logger.info(f"检测置信度: {detection_score:.4f}, OCR置信度: {ocr_confidence:.4f}")
                self.logger.info(f"车牌格式验证: {'通过' if is_valid_plate else '未通过'}")
            else:
                self.logger.warning("未识别到有效车牌")
        
        return {
            'success': bool(final_text),
            'text': final_text,  # 校正后的文本（如果验证未通过则为 None）
            'raw_text': raw_text,  # 原始OCR文本
            'confidence': detection_score,
            'ocr_confidence': ocr_confidence,
            'is_valid_plate': is_valid_plate,  # 是否通过车牌正则表达式验证
            'detection_count': detection_result['count'],
            'warped_image': warped_image,
            'details': {
                'texts': texts,
                'scores': scores,
                'ocr_details': details,
                'detection_score': detection_score,
                'corrected': bool(corrected_text),  # 是否进行了文本校正
                'original_text': raw_text,  # 保留原始文本
                'orientation': orientation,  # 文本行方向信息
                'is_valid_plate': is_valid_plate  # 是否通过车牌正则表达式验证
            }
        }
    
    def recognize_batch(self,
                       image_paths: List[str],
                       angles: Optional[List[float]] = None,
                       use_best_detection: Optional[bool] = None,
                       target_width: Optional[int] = None,
                       target_height: Optional[int] = None) -> List[Dict]:
        """
        批量识别多张图片
        
        Args:
            image_paths: 图像路径列表
            angles: 检测角度列表，如果为None则使用类初始化时设置的默认角度
            use_best_detection: 是否只使用置信度最高的检测结果，如果为None则使用类初始化时的默认值
            target_width: 透视变换目标宽度，如果为None则使用类初始化时的默认值
            target_height: 透视变换目标高度，如果为None则使用类初始化时的默认值
        
        Returns:
            识别结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.recognize(
                image_path,
                angles=angles,
                use_best_detection=use_best_detection,
                target_width=target_width,
                target_height=target_height
            )
            result['image_path'] = image_path
            results.append(result)
        
        return results


if __name__ == '__main__':
    # 使用示例
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='车牌识别 Pipeline（基于关键点）')
    parser.add_argument('image_path', type=str, help='图像路径')
    parser.add_argument('--detector_model', type=str, 
                       default='models/det/yolo11n-pose-LPR-best.pt',
                       help='检测模型路径')
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 90, 180, 270],
                       help='检测角度列表')
    parser.add_argument('--save_warped', action='store_true',
                       help='保存透视变换后的车牌图像')
    parser.add_argument('--detector_conf', type=float, default=0.5,
                       help='检测器置信度阈值')
    parser.add_argument('--detector_iou', type=float, default=0.45,
                       help='检测器IoU阈值')
    parser.add_argument('--detector_gpu', action='store_true', default=True,
                       help='检测器使用GPU')
    parser.add_argument('--detector_no_gpu', action='store_true',
                       help='检测器不使用GPU')
    parser.add_argument('--ocr_gpu', action='store_true', default=True,
                       help='OCR使用GPU')
    parser.add_argument('--ocr_no_gpu', action='store_true',
                       help='OCR不使用GPU')
    parser.add_argument('--no_concatenate', action='store_true',
                       help='不使用拼接180度图像（默认使用拼接）')
    parser.add_argument('--keypoint_expand_ratio', type=float, default=0.2,
                       help='关键点扩展比例（最佳检测，默认0.2）')
    parser.add_argument('--keypoint_expand_ratio_first', type=float, default=0.1,
                       help='关键点扩展比例（第一个检测，默认0.1）')
    parser.add_argument('--target_width', type=int, default=None,
                       help='透视变换目标宽度')
    parser.add_argument('--target_height', type=int, default=None,
                       help='透视变换目标高度')
    parser.add_argument('--use_best_detection', action='store_true', default=True,
                       help='使用最佳检测结果（默认True）')
    parser.add_argument('--use_first_detection', action='store_true',
                       help='使用第一个检测结果（覆盖use_best_detection）')
    
    args = parser.parse_args()
    
    # 初始化 Pipeline
    pipeline = LicensePlatePipelineKP(
        # 检测器参数
        detector_model_path=args.detector_model,
        detector_conf_threshold=args.detector_conf,
        detector_iou_threshold=args.detector_iou,
        detector_use_gpu=args.detector_gpu and not args.detector_no_gpu,
        # OCR参数
        ocr_use_gpu=args.ocr_gpu and not args.ocr_no_gpu,
        # 识别方法默认参数
        default_angles=args.angles,
        default_use_best_detection=args.use_best_detection and not args.use_first_detection,
        default_target_width=args.target_width,
        default_target_height=args.target_height,
        # 关键点扩展参数
        keypoint_expand_ratio=args.keypoint_expand_ratio,
        keypoint_expand_ratio_first=args.keypoint_expand_ratio_first,
        # 其他参数
        use_concatenated_image=not args.no_concatenate
    )
    
    # 执行识别
    print(f"正在识别: {args.image_path}")
    result = pipeline.recognize(
        image_path=args.image_path,
        angles=args.angles
    )
    
    # 输出结果
    print("\n" + "=" * 80)
    print("识别结果:")
    print("=" * 80)
    print(f"成功: {result['success']}")
    print(f"校正后文本: {result['text']}")
    if result.get('raw_text') and result['raw_text'] != result['text']:
        print(f"原始OCR文本: {result['raw_text']}")
    print(f"检测置信度: {result['confidence']:.4f}")
    if 'ocr_confidence' in result:
        print(f"OCR置信度: {result['ocr_confidence']:.4f}")
    print(f"检测到车牌数: {result['detection_count']}")
    if result.get('details', {}).get('corrected'):
        print(f"已进行文本校正")
    orientation_info = result.get('details', {}).get('orientation', {})
    if orientation_info:
        angle = orientation_info.get('angle', 0.0)
        score = orientation_info.get('score', 0.0)
        if abs(angle) > 0.5:
            print(f"文本行方向: 检测到 {angle}° 旋转 (置信度: {score:.4f})")
    
    if result['success']:
        print(f"\n详细信息:")
        if result['details'].get('texts'):
            for i, (text, score) in enumerate(zip(result['details']['texts'], 
                                                  result['details'].get('scores', [])), 1):
                print(f"  文本 {i}: {text} (置信度: {score:.4f})")
    else:
        print(f"\n错误: {result['details'].get('error', '未知错误')}")
    
    # 保存透视变换后的图像
    if args.save_warped and result['warped_image'] is not None:
        # 自动创建保存路径：基于输入图片路径
        input_path = Path(args.image_path)
        output_dir = input_path.parent / 'warped_plates'
        output_dir.mkdir(exist_ok=True)
        
        # 生成输出文件名：原文件名_warped.jpg
        output_filename = f"{input_path.stem}_warped.jpg"
        output_path = output_dir / output_filename
        
        cv2.imwrite(str(output_path), result['warped_image'])
        print(f"\n透视变换后的车牌图像已保存到: {output_path}")

