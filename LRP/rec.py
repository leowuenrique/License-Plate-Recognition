#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 识别器
参考 hyperlpr3_integrated.py 中的 PaddleOCRRecognition 类实现
输入: np.ndarray 图像
输出: 识别的文本结果
"""

import cv2
import numpy as np
import threading
import os
import tempfile
from typing import Tuple, Optional, Dict, List, Union


class PaddleOCRRecognition:
    """PaddleOCR 识别器 - 基于 PaddleOCR"""
    
    def __init__(self, 
                 lang: str = "ch",
                 use_doc_orientation_classify: bool = False,
                 use_doc_unwarping: bool = False,
                 use_textline_orientation: bool = True,
                 enable_mkldnn: bool = True,
                 use_tensorrt: bool = False,
                 use_gpu: bool = True,
                 **kwargs):
        """
        初始化 PaddleOCR 识别器
        
        Args:
            lang: 语言类型，默认 "ch"（中文）
            use_doc_orientation_classify: 是否使用文档方向分类（默认 False）
            use_doc_unwarping: 是否使用文档反扭曲（默认 False）
            use_textline_orientation: 是否使用文本行方向检测（默认 False）
            enable_mkldnn: 是否启用 Intel MKL-DNN 加速
            use_tensorrt: 是否使用 TensorRT
            use_gpu: 是否使用GPU，如果为False则通过环境变量强制使用CPU
            **kwargs: 其他 PaddleOCR 初始化参数
        """
        # 如果禁用GPU，设置环境变量（必须在导入PaddleOCR之前）
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("请安装 PaddleOCR: pip install paddleocr")
        
        # PaddleOCR 配置（保留原有参数设置）
        config = {
            "lang": lang,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            "enable_mkldnn": enable_mkldnn,
            "use_tensorrt": use_tensorrt,
            # "text_det_limit_side_len": 736,
            "text_recognition_model_dir": "models/rec/PP-OCRv5_server_rec_infer",
            **kwargs
        }
        
        # 初始化 PaddleOCR 模型
        self.ocr_model = PaddleOCR(**config)
        
        # 线程锁（确保 PaddleOCR 模型线程安全）
        self.lock = threading.Lock()
    
    def __call__(self, image: np.ndarray) -> Union[str, Dict]:
        """
        识别图像中的文本
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
        
        Returns:
            如果返回详细信息（包含坐标、置信度），返回字典：
            {
                'text': str,  # 识别到的文本（多个文本用换行符分隔）
                'texts': List[str],  # 文本列表
                'boxes': List[List[List[float]]],  # 每个文本的坐标框（四点坐标）
                'scores': List[float],  # 每个文本的置信度
                'details': List[Dict],  # 详细信息列表，每个元素包含 {'text': str, 'box': List, 'score': float}
            }
            如果只返回文本（向后兼容），返回字符串
        """
        try:
            # 线程安全：使用锁保护 PaddleOCR 模型访问
            with self.lock:
                # 尝试直接使用 numpy 数组
                try:
                    result = self.ocr_model.predict(image)
                except (TypeError, AttributeError):
                    # 如果不支持 numpy 数组，使用临时文件
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    try:
                        cv2.imwrite(tmp_path, image)
                        result = self.ocr_model.predict(tmp_path)
                    finally:
                        # 清理临时文件
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
            
            # 处理识别结果
            if not result:
                return {
                    'text': '',
                    'texts': [],
                    'boxes': [],
                    'scores': [],
                    'details': [],
                    'output_img': None
                }
            
            # PaddleOCR可能返回多种格式
            rec_texts = []
            rec_boxes = []
            rec_scores = []
            output_img = None  # 从 doc_preprocessor_res 中获取的处理后图像
            
            # 尝试解析结果
            if isinstance(result, list):
                if len(result) == 0:
                    return {
                        'text': '',
                        'texts': [],
                        'boxes': [],
                        'scores': [],
                        'details': [],
                        'output_img': None
                    }
                
                # 检查是否是字典格式（新版本PaddleOCR）
                if isinstance(result[0], dict):
                    res = result[0]
                    rec_texts = res.get('rec_texts', [])
                    # 尝试多个可能的字段名
                    rec_boxes = res.get('rec_boxes', res.get('dt_boxes', []))
                    rec_scores = res.get('rec_scores', [])
                    
                    # 从 doc_preprocessor_res 中获取 output_img
                    output_img = None
                    doc_preprocessor_res = res.get('doc_preprocessor_res', {})
                    if doc_preprocessor_res:
                        output_img = doc_preprocessor_res.get('output_img')
                        # 如果是numpy数组，确保是BGR格式
                        if output_img is not None and isinstance(output_img, np.ndarray):
                            # 如果只有2个维度，转换为3通道
                            if len(output_img.shape) == 2:
                                output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
                            elif len(output_img.shape) == 3 and output_img.shape[2] == 1:
                                output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
                # 检查是否是列表格式（旧版本PaddleOCR: [[坐标, (文本, 置信度)], ...]）
                elif isinstance(result[0], list) and len(result[0]) >= 2:
                    for line in result:
                        if len(line) >= 2:
                            box = line[0]  # 坐标
                            text_info = line[1]  # (文本, 置信度) 或 文本
                            
                            if isinstance(text_info, tuple) and len(text_info) >= 2:
                                text, score = text_info[0], text_info[1]
                            elif isinstance(text_info, str):
                                text, score = text_info, 1.0
                            else:
                                continue
                            
                            rec_texts.append(text)
                            rec_boxes.append(box)
                            rec_scores.append(float(score))
            
            if not rec_texts:
                return {
                    'text': '',
                    'texts': [],
                    'boxes': [],
                    'scores': [],
                    'details': [],
                    'output_img': output_img
                }
            
            # 确保所有列表长度一致
            num_texts = len(rec_texts)
            if len(rec_boxes) != num_texts:
                # 如果坐标数量不匹配，创建空坐标
                rec_boxes = [[] for _ in range(num_texts)]
            if len(rec_scores) != num_texts:
                # 如果置信度数量不匹配，创建默认置信度
                rec_scores = [0.0 for _ in range(num_texts)]
            
            # 构建详细信息列表
            details = []
            for i in range(num_texts):
                # 处理坐标格式：如果是numpy数组，转换为列表
                box = rec_boxes[i] if i < len(rec_boxes) else []
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                elif isinstance(box, np.ndarray):
                    box = box.tolist()
                
                details.append({
                    'text': rec_texts[i],
                    'box': box,
                    'score': float(rec_scores[i]) if i < len(rec_scores) else 0.0
                })
            
            # 合并所有识别到的文本，用换行符分隔
            text_result = '\n'.join(rec_texts)
            
            return {
                'text': text_result,  # 向后兼容：文本字符串
                'texts': rec_texts,  # 文本列表
                'boxes': rec_boxes,  # 坐标列表
                'scores': rec_scores,  # 置信度列表
                'details': details,  # 详细信息列表
                'output_img': output_img  # 处理后的图像（从 doc_preprocessor_res 中获取）
            }
            
        except Exception as e:
            # 识别失败，返回空结果
            return {
                'text': '',
                'texts': [],
                'boxes': [],
                'scores': [],
                'details': [],
                'output_img': None
            }

# 使用示例
if __name__ == "__main__":
    # 测试完整流程：检测 -> 校正 -> 识别
    image_path = "test.jpg"
    ocr = PaddleOCRRecognition()
    image = cv2.imread(image_path)
    result = ocr(image)
    print(result)
    