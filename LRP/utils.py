import cv2
import numpy as np
import re
import math
import logging
import itertools
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 文本校正相关的logger
_text_correction_logger = logging.getLogger(__name__)

# 中国省份简称列表（用于文本校正）
PROVINCE_ABBREVIATIONS = {
    '京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', 
    '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', 
    '蒙', '陕', '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼'
}

def create_concatenated_image_for_recognition(corrected_image: np.ndarray, 
                                               original_roi: Optional[np.ndarray] = None) -> np.ndarray:
    """
    创建用于识别的拼接图像
    
    将校正后的图像旋转180度，然后与校正后的图像进行上下拼接
    这样可以提高OCR识别的准确率，因为OCR可以同时看到正向和反向的图像
    
    Args:
        corrected_image: 校正后的车牌区域图像 (np.ndarray)
        original_roi: 原始车牌区域图像（可选，已废弃，不再使用）
    
    Returns:
        拼接后的图像 (np.ndarray)，校正后的图像在上，旋转180度的图像在下
    """
    if corrected_image is None or corrected_image.size == 0:
        raise ValueError("校正后的图像不能为空")
    
    # 将校正后的图像旋转180度
    h, w = corrected_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated_image = cv2.warpAffine(
        corrected_image, 
        rotation_matrix, 
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0) if len(corrected_image.shape) == 3 else 0
    )
    
    # 确保两个图像的宽度相同（取较大的宽度）
    max_width = max(corrected_image.shape[1], rotated_image.shape[1])
    
    # 调整图像宽度（如果需要）
    if corrected_image.shape[1] != max_width:
        corrected_image = cv2.resize(corrected_image, (max_width, corrected_image.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
    if rotated_image.shape[1] != max_width:
        rotated_image = cv2.resize(rotated_image, (max_width, rotated_image.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
    
    # 上下拼接：校正后的图像在上，旋转180度的图像在下
    concatenated = np.vstack([corrected_image, rotated_image])
    
    return concatenated


def validate_license_plate(plate_text: str) -> bool:
    """
    使用正则表达式验证车牌格式是否符合中国车牌规则
    
    中国车牌规则：
    - 第一位：省份简称（1个汉字）
    - 第二位：大写字母（A-Z，不包括I和O）
    - 后5-6位：数字和大写字母组合（不包括I和O）
    - 总长度：7-8个字符
    
    Args:
        plate_text: 车牌文本
    
    Returns:
        如果符合车牌格式返回True，否则返回False
    """
    if not plate_text:
        return False
    
    # 检查长度（7-8位）
    if len(plate_text) < 7 or len(plate_text) > 8:
        return False
    
    # 检查第一位是否是有效的省份简称（使用全局 PROVINCE_ABBREVIATIONS）
    if plate_text[0] not in PROVINCE_ABBREVIATIONS:
        return False
    
    # 检查第二位是否是大写字母（不包括I和O）
    if len(plate_text) < 2:
        return False
    if plate_text[1] not in 'ABCDEFGHJKLMNPQRSTUVWXYZ':
        return False
    
    # 检查后5-6位是否只包含数字和大写字母（不包括I和O）
    remaining = plate_text[2:]
    if not remaining:
        return False
    
    # 后5-6位应该只包含数字和大写字母（不包括I和O）
    # 匹配：5-6位，只包含数字0-9和大写字母A-H, J-N, P-Z（不包括I和O）
    pattern = r'^[A-HJ-NP-Z0-9]{5,6}$'
    if not re.match(pattern, remaining):
        return False
    
    return True


def select_center_detection_from_detector_result(detection_result: Dict, 
                                                  iou_threshold: float = 0.3,
                                                  distance_threshold_ratio: float = 0.1) -> Optional[Dict]:
    """
    从 detector_kp.py 的检测结果中选择检测框中心点离图像中心点最近的检测框
    
    此函数用于 pipeline_kp.py 的步骤2中，选择最佳检测结果进行透视变换。
    选择逻辑：
    1. 计算每个检测框的中心点，找到离图像中心点最近的检测框
    2. 如果有多个检测框与最近检测框重叠（IoU > threshold），在这些重叠的检测框中选择置信度最高的
    
    Args:
        detection_result: 检测结果字典，格式为：
            {
                'boxes': [[x1, y1, x2, y2], ...],  # 旋转图像坐标系
                'keypoints': [[[x1, y1], ...], ...],  # 旋转图像坐标系
                'scores': [score1, score2, ...],
                'classes': [cls1, cls2, ...],
                'angles': [angle1, angle2, ...],
                'count': int,
                'image': np.ndarray (可选) 原始图像
                'rotated_images': {angle: np.ndarray, ...} (可选) 旋转后的图像字典
            }
        iou_threshold: IoU阈值，用于判断检测框是否重叠，默认0.3
        distance_threshold_ratio: 距离阈值比例，用于判断检测框是否距离相近，默认0.1（图像对角线长度的10%）
    
    Returns:
        如果找到检测框，返回包含离图像中心最近的检测框信息的字典：
        {
            'box': [x1, y1, x2, y2],  # 旋转图像坐标系
            'keypoints': [[x1, y1], ...] 或 None,  # 旋转图像坐标系
            'score': float,
            'class': int,
            'angle': float,
            'image': np.ndarray (原始图像，如果原检测结果包含)
            'rotated_image': np.ndarray (对应的旋转图像，如果原检测结果包含)
        }
        如果没有检测框，返回 None
    
    Note:
        检测框坐标是旋转图像坐标系，需要根据对应的角度和旋转图像来计算中心点距离。
        这里使用原始图像作为参考，将旋转图像上的检测框转换回原图坐标系后计算距离。
    """
    if not detection_result or detection_result.get('count', 0) == 0:
        return None
    
    boxes = detection_result.get('boxes', [])
    keypoints = detection_result.get('keypoints', [])
    scores = detection_result.get('scores', [])
    classes = detection_result.get('classes', [])
    angles = detection_result.get('angles', [])
    
    if not boxes:
        return None
    
    # 获取原始图像尺寸
    image = detection_result.get('image')
    if image is None:
        return None
    
    h, w = image.shape[:2]
    image_center_x = w / 2.0
    image_center_y = h / 2.0
    
    # 计算图像对角线长度（用于距离阈值）
    image_diagonal = math.sqrt(w * w + h * h)
    distance_threshold = image_diagonal * distance_threshold_ratio
    
    # 获取旋转图像字典
    rotated_images = detection_result.get('rotated_images', {})
    
    # 存储所有检测框的中心点和距离信息
    detection_info = []
    
    # 遍历所有检测框，计算到图像中心的距离
    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        
        # 获取对应的角度
        if i < len(angles):
            angle = angles[i]
        else:
            angle = 0.0
        
        # 计算检测框在原图坐标系中的中心点和边界框
        if angle == 0:
            x1, y1, x2, y2 = box
            box_center_x = (x1 + x2) / 2.0
            box_center_y = (y1 + y2) / 2.0
            original_box = [x1, y1, x2, y2]
        else:
            # 需要将旋转图像上的检测框转换回原图坐标系
            if angle in rotated_images:
                rotated_image = rotated_images[angle]
                rot_h, rot_w = rotated_image.shape[:2]
                
                # 计算旋转中心（原图中心）
                center = (w / 2.0, h / 2.0)
                
                # 计算旋转矩阵
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # 计算旋转后的图像尺寸
                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # 调整旋转矩阵的平移部分
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                # 计算逆变换矩阵
                inv_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
                
                # 转换检测框的四个角点
                x1, y1, x2, y2 = box
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.float32)
                
                # 应用逆变换
                corners_homogeneous = np.column_stack([corners, np.ones(4)])
                transformed_corners = (inv_rotation_matrix @ corners_homogeneous.T).T
                
                # 计算转换后的边界框
                x_min = np.min(transformed_corners[:, 0])
                x_max = np.max(transformed_corners[:, 0])
                y_min = np.min(transformed_corners[:, 1])
                y_max = np.max(transformed_corners[:, 1])
                
                # 裁剪到原始图像范围内
                x_min = max(0, min(w, x_min))
                x_max = max(0, min(w, x_max))
                y_min = max(0, min(h, y_min))
                y_max = max(0, min(h, y_max))
                
                # 计算中心点
                box_center_x = (x_min + x_max) / 2.0
                box_center_y = (y_min + y_max) / 2.0
                original_box = [x_min, y_min, x_max, y_max]
            else:
                # 如果没有旋转图像，使用旋转图像坐标（近似）
                x1, y1, x2, y2 = box
                box_center_x = (x1 + x2) / 2.0
                box_center_y = (y1 + y2) / 2.0
                original_box = [x1, y1, x2, y2]
        
        # 计算到图像中心的欧氏距离
        distance = math.sqrt(
            (box_center_x - image_center_x) ** 2 + 
            (box_center_y - image_center_y) ** 2
        )
        
        detection_info.append({
            'index': i,
            'distance': distance,
            'score': scores[i] if i < len(scores) else 0.0,
            'original_box': original_box,  # 原图坐标系中的检测框
            'box': box,  # 旋转图像坐标系中的检测框
            'angle': angle
        })
    
    if not detection_info:
        return None
    
    # 按距离排序
    detection_info.sort(key=lambda x: x['distance'])
    
    # 找到离中心最近的检测框
    closest_detection = detection_info[0]
    min_distance = closest_detection['distance']
    
    # 找出所有与最近检测框重叠的检测框（IoU > threshold）
    overlapping_detections = [closest_detection]
    
    for det in detection_info[1:]:
        # 如果距离相差太大，跳过
        if det['distance'] - min_distance > distance_threshold:
            break
        
        # 计算IoU
        iou = calculate_iou(closest_detection['original_box'], det['original_box'])
        if iou > iou_threshold:
            overlapping_detections.append(det)
    
    # 在重叠的检测框中选择置信度最高的
    best_detection = max(overlapping_detections, key=lambda x: x['score'])
    best_index = best_detection['index']
    
    if best_index is None:
        return None
    
    # 返回离图像中心最近的检测框信息
    result = {
        'box': boxes[best_index],
        'score': scores[best_index] if best_index < len(scores) else 0.0,
    }
    
    # 添加关键点信息（如果存在）
    if keypoints and best_index < len(keypoints):
        result['keypoints'] = keypoints[best_index]
    else:
        result['keypoints'] = None
    
    # 添加类别信息（如果存在）
    if classes and best_index < len(classes):
        result['class'] = classes[best_index]
    
    # 添加角度信息（如果存在）
    detected_angle = None
    if angles and best_index < len(angles):
        detected_angle = angles[best_index]
        result['angle'] = detected_angle
    
    # 保留原始图像（如果存在）
    if 'image' in detection_result:
        result['image'] = detection_result['image']
    
    # 添加对应的旋转图像（如果存在）
    if 'rotated_images' in detection_result and detected_angle is not None:
        rotated_images = detection_result['rotated_images']
        if detected_angle in rotated_images:
            result['rotated_image'] = rotated_images[detected_angle]
    
    return result


# ============================================================================
# 文本校正函数 V4（从 text_correction_v4.py 集成）
# ============================================================================

def calculate_iou(box1: List, box2: List) -> float:
    """
    计算两个框的IoU (Intersection over Union)
    
    Args:
        box1: [x1, y1, x2, y2] 或 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        box2: 同上
    
    Returns:
        IoU值 (0-1)
    """
    # 转换为矩形格式 [x1, y1, x2, y2]
    if isinstance(box1[0], (list, tuple)):
        x_coords1 = [p[0] for p in box1]
        y_coords1 = [p[1] for p in box1]
        x1_min, x1_max = min(x_coords1), max(x_coords1)
        y1_min, y1_max = min(y_coords1), max(y_coords1)
    else:
        x1_min, y1_min, x1_max, y1_max = box1
    
    if isinstance(box2[0], (list, tuple)):
        x_coords2 = [p[0] for p in box2]
        y_coords2 = [p[1] for p in box2]
        x2_min, x2_max = min(x_coords2), max(x_coords2)
        y2_min, y2_max = min(y_coords2), max(y_coords2)
    else:
        x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    非极大值抑制 (Non-Maximum Suppression)
    
    Args:
        detections: 检测结果列表，每个元素包含 {'text', 'box', 'score', ...}
        iou_threshold: IoU阈值
    
    Returns:
        过滤后的检测结果列表
    """
    if not detections:
        return []
    
    # 按置信度降序排序
    sorted_detections = sorted(detections, key=lambda x: x.get('score', 0.0), reverse=True)
    
    keep = []
    while sorted_detections:
        # 保留置信度最高的
        current = sorted_detections.pop(0)
        keep.append(current)
        
        # 移除与当前框IoU过高的框
        sorted_detections = [
            det for det in sorted_detections
            if calculate_iou(current.get('box', []), det.get('box', [])) < iou_threshold
        ]
    
    return keep


def calculate_box_center(box: List) -> Tuple[float, float]:
    """
    计算检测框的中心点
    
    Args:
        box: [x1, y1, x2, y2] 或 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    
    Returns:
        (x_center, y_center)
    """
    if isinstance(box[0], (list, tuple)):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x_center = sum(x_coords) / len(x_coords)
        y_center = sum(y_coords) / len(y_coords)
    else:
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
    
    return x_center, y_center


def calculate_box_height(box: List) -> float:
    """
    计算检测框的高度
    
    Args:
        box: [x1, y1, x2, y2] 或 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    
    Returns:
        高度
    """
    if isinstance(box[0], (list, tuple)):
        y_coords = [p[1] for p in box]
        return max(y_coords) - min(y_coords)
    else:
        x1, y1, x2, y2 = box
        return abs(y2 - y1)


def spatial_grouping(detections: List[Dict], y_threshold_ratio: float = 0.5) -> List[List[Dict]]:
    """
    空间布局聚类：将属于同一行的检测框归为一组
    
    Args:
        detections: 检测结果列表
        y_threshold_ratio: Y坐标差异阈值（相对于平均高度）
    
    Returns:
        分组后的检测结果列表 [[group1], [group2], ...]
    """
    if not detections:
        return []
    
    # 计算平均高度
    heights = [calculate_box_height(det.get('box', [])) for det in detections]
    avg_height = sum(heights) / len(heights) if heights else 50.0
    y_threshold = avg_height * y_threshold_ratio
    
    # 按Y坐标中心点排序
    detections_with_y = []
    for det in detections:
        x_center, y_center = calculate_box_center(det.get('box', []))
        detections_with_y.append((y_center, det))
    
    detections_with_y.sort(key=lambda x: x[0])
    
    # 聚类
    groups = []
    current_group = []
    current_y = None
    
    for y_center, det in detections_with_y:
        if current_y is None or abs(y_center - current_y) <= y_threshold:
            # 属于当前组
            current_group.append(det)
            if current_y is None:
                current_y = y_center
            else:
                current_y = (current_y + y_center) / 2  # 更新组中心
        else:
            # 新组
            if current_group:
                groups.append(current_group)
            current_group = [det]
            current_y = y_center
    
    if current_group:
        groups.append(current_group)
    
    return groups


def clean_text(text: str) -> str:
    """
    清理文本：去除无效符号并转换为大写
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return ''
    
    cleaned = text.replace('·', '').replace('•', '').replace(' ', '')
    cleaned = cleaned.replace('\n', '').replace('\r', '').replace('\t', '')
    cleaned = cleaned.replace('-', '')  # 去除无效符号 '-'
    # 去除中文标点符号
    cleaned = cleaned.replace('：', '').replace('，', '').replace('。', '')
    cleaned = cleaned.replace('、', '').replace('；', '')
    cleaned = cleaned.upper()
    
    return cleaned


def extract_valid_chars(text: str) -> List[str]:
    """
    提取有效字符（汉字、字母、数字）
    
    Args:
        text: 文本
    
    Returns:
        有效字符列表
    """
    pattern = r'[\u4e00-\u9fa5A-Z0-9]'
    return re.findall(pattern, text)


def contains_only_valid_chars(text: str) -> bool:
    """
    检查文本是否只包含有效字符（数字、字母、省份简称）
    
    Args:
        text: 文本
    
    Returns:
        如果只包含有效字符返回True，否则返回False
    """
    if not text:
        return False
    
    # 清理文本（去除无效符号）
    cleaned_text = clean_text(text)
    
    # 使用正则表达式检查是否只包含汉字、字母、数字
    pattern = r'^[\u4e00-\u9fa5A-Z0-9]+$'
    if not re.match(pattern, cleaned_text):
        return False
    
    # 检查文本中的汉字是否都是有效的省份简称
    # 提取所有汉字字符
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', cleaned_text)
    
    # 如果包含汉字，检查是否都是有效的省份简称
    for char in chinese_chars:
        if char not in PROVINCE_ABBREVIATIONS:
            # 如果汉字不是有效的省份简称，返回False
            return False
    
    return True


def extract_valid_part(text: str) -> Optional[str]:
    """
    从包含无效字符的文本中提取有效部分
    例如："赣日" -> "赣"（提取有效的省份简称）
    
    Args:
        text: 原始文本
    
    Returns:
        提取的有效部分，如果无法提取则返回None
    """
    if not text:
        return None
    
    # 清理文本（去除无效符号）
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return None
    
    # 如果文本只包含有效字符，直接返回
    if contains_only_valid_chars(cleaned_text):
        return cleaned_text
    
    # 尝试提取有效的省份简称部分
    # 查找文本开头的有效省份简称
    for i in range(1, min(len(cleaned_text) + 1, 2)):  # 省份简称通常是1个字符
        prefix = cleaned_text[:i]
        if prefix in PROVINCE_ABBREVIATIONS:
            # 如果找到有效的省份简称，返回它
            if _text_correction_logger.isEnabledFor(logging.DEBUG):
                _text_correction_logger.debug(f"  从文本中提取有效部分: '{text}' -> '{prefix}'")
            return prefix
    
    # 尝试提取包含省份简称的有效部分
    # 查找文本中是否包含有效的省份简称
    for i, char in enumerate(cleaned_text):
        if char in PROVINCE_ABBREVIATIONS:
            # 找到省份简称，尝试提取从省份简称开始的有效部分
            # 提取省份简称 + 后面连续的字母和数字（最多6个字符）
            valid_part = char
            j = i + 1
            while j < len(cleaned_text) and len(valid_part) < 7:
                next_char = cleaned_text[j].upper()
                if next_char.isalnum():
                    valid_part += next_char
                    j += 1
                else:
                    break
            
            # 如果提取的部分只包含有效字符，返回它
            if contains_only_valid_chars(valid_part):
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  从文本中提取有效部分: '{text}' -> '{valid_part}'")
                return valid_part
            else:
                # 如果提取的部分仍然包含无效字符，只返回省份简称
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  从文本中提取有效部分: '{text}' -> '{char}'")
                return char
    
    return None


def check_single_item_validity(text: str) -> Optional[str]:
    """
    检查单个文本项是否是有效车牌（去除无效符号后）
    
    Args:
        text: 文本项
    
    Returns:
        如果是有效车牌，返回纠正后的文本；否则返回None
    """
    if not text:
        return None
    
    cleaned_text = clean_text(text)
    return validate_and_correct_license_plate(cleaned_text)


def calculate_avg_score(group: List[Dict]) -> float:
    """
    计算组的平均置信度
    
    Args:
        group: 检测结果组
    
    Returns:
        平均置信度
    """
    if not group:
        return 0.0
    return sum(det.get('score', 0.0) for det in group) / len(group)


def try_cross_group_combination(group_texts_list: List[str], 
                                 group_scores_list: List[float],
                                 max_groups: int = 2) -> Tuple[Optional[str], float]:
    """
    尝试跨组组合
    
    Args:
        group_texts_list: 组文本列表
        group_scores_list: 组置信度列表
        max_groups: 最大组合组数（2或3）
    
    Returns:
        (最佳结果, 最佳置信度)
    """
    best_result = None
    best_score = 0.0
    
    if max_groups == 2 and len(group_texts_list) >= 2:
        # 尝试2个组的组合
        for i, j in itertools.combinations(range(len(group_texts_list)), 2):
            text1 = group_texts_list[i]
            text2 = group_texts_list[j]
            combined_score = (group_scores_list[i] + group_scores_list[j]) / 2
            
            # 尝试两种顺序
            for combined_text in [text1 + text2, text2 + text1]:
                if len(combined_text) > 8:
                    if _text_correction_logger.isEnabledFor(logging.DEBUG):
                        _text_correction_logger.debug(f"  跳过跨组组合（超过8个字符）: '{combined_text}' (长度: {len(combined_text)})")
                    continue
                
                if _text_correction_logger.isEnabledFor(logging.INFO):
                    _text_correction_logger.info(f"  尝试跨组组合: '{text1}' + '{text2}' = '{combined_text}'")
                
                refined = rule_based_refinement(combined_text)
                if refined and combined_score > best_score:
                    best_result = refined
                    best_score = combined_score
                    if _text_correction_logger.isEnabledFor(logging.INFO):
                        _text_correction_logger.info(f"  [跨组组合] 找到有效车牌: '{combined_text}' -> '{refined}' (置信度: {combined_score:.4f})")
    
    elif max_groups == 3 and len(group_texts_list) >= 3:
        # 尝试3个组的组合
        for i, j, k in itertools.permutations(range(len(group_texts_list)), 3):
            text1 = group_texts_list[i]
            text2 = group_texts_list[j]
            text3 = group_texts_list[k]
            combined_text = text1 + text2 + text3
            combined_score = (group_scores_list[i] + group_scores_list[j] + group_scores_list[k]) / 3
            
            if len(combined_text) > 8:
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  跳过跨组组合（超过8个字符）: '{combined_text}' (长度: {len(combined_text)})")
                continue
            
            if _text_correction_logger.isEnabledFor(logging.INFO):
                _text_correction_logger.info(f"  尝试跨组组合（3组）: '{text1}' + '{text2}' + '{text3}' = '{combined_text}'")
            
            refined = rule_based_refinement(combined_text)
            if refined and combined_score > best_score:
                best_result = refined
                best_score = combined_score
                if _text_correction_logger.isEnabledFor(logging.INFO):
                    _text_correction_logger.info(f"  [跨组组合] 找到有效车牌: '{combined_text}' -> '{refined}' (置信度: {combined_score:.4f})")
    
    return best_result, best_score


def correct_ocr_errors_simple(text: str) -> str:
    """
    简单纠正OCR识别错误：将 O 替换成 0，I 替换成 1
    用于去重检查，不区分位置
    
    Args:
        text: 车牌文本
    
    Returns:
        纠正后的文本
    """
    if not text:
        return text
    
    corrected = ''
    for char in text:
        if char == 'O':
            corrected += '0'
        elif char == 'I':
            corrected += '1'
        else:
            corrected += char
    
    return corrected


def correct_ocr_errors(text: str) -> str:
    """
    纠正OCR识别错误：将 O 替换成 0，I 替换成 1
    
    Args:
        text: 车牌文本
    
    Returns:
        纠正后的文本
    """
    if not text or len(text) < 2:
        return text
    
    # 第一位是省份简称（汉字），保持不变
    province_char = text[0]
    remaining = text[1:]
    
    # 处理第二位：如果是 I 或 O，替换成 L 或 D
    if len(remaining) > 0:
        second_char = remaining[0]
        if second_char == 'I':
            second_char = 'L'
        elif second_char == 'O':
            second_char = 'D'
        remaining = second_char + remaining[1:]
    
    # 处理后5-6位：O 替换成 0，I 替换成 1
    corrected_remaining = ''
    for char in remaining:
        if char == 'O':
            corrected_remaining += '0'
        elif char == 'I':
            corrected_remaining += '1'
        else:
            corrected_remaining += char
    
    return province_char + corrected_remaining


def validate_and_correct_license_plate(text: str, max_length: int = 8) -> Optional[str]:
    """
    纠正OCR错误并验证车牌格式
    
    Args:
        text: 车牌文本
        max_length: 最大长度，默认8
    
    Returns:
        如果通过验证，返回纠正后的文本；否则返回None
    """
    if not text:
        return None
    
    # 纠正OCR错误
    corrected_text = correct_ocr_errors(text)
    
    # 检查长度
    if len(corrected_text) > max_length:
        return None
    
    # 验证车牌格式
    if validate_license_plate(corrected_text):
        return corrected_text
    
    return None


def sort_and_concatenate(group: List[Dict], check_rotation: bool = False) -> str:
    """
    排序与拼接：对同一组内的检测框按X坐标排序并拼接
    
    注意：识别器处理的结果已经是校正过的，不需要进行倒置检测
    
    Args:
        group: 同一组的检测结果列表
        check_rotation: 是否检查旋转（倒置），默认False（识别器已校正）
    
    Returns:
        拼接后的文本
    """
    if not group:
        return ''
    
    # 按X坐标中心点排序（从左到右）
    group_with_x = []
    for det in group:
        x_center, y_center = calculate_box_center(det.get('box', []))
        group_with_x.append((x_center, det))
    
    group_with_x.sort(key=lambda x: x[0])
    
    # 拼接文本（按X坐标从左到右的顺序）
    texts = []
    for x_center, det in group_with_x:
        text = det.get('text', '').strip()
        if text:
            texts.append(text)
    
    result = ''.join(texts)
    
    # 不再进行倒置检测，因为识别器处理的结果已经是校正过的
    # 如果拼接后的结果不符合车牌格式，会在步骤4的规则后处理中进行修正
    
    return result


def rule_based_refinement(text: str) -> Optional[str]:
    """
    基于业务规则的后处理
    
    Args:
        text: 拼接后的文本
    
    Returns:
        修正后的文本，如果不符合规则返回None
    """
    if not text:
        return None
    
    # 清理文本并提取有效字符
    cleaned_text = clean_text(text)
    chars = extract_valid_chars(cleaned_text)
    
    if not chars:
        return None
    
    # 首先尝试识别并保持有效的前缀模式（省份+字母），如"新AF"
    # 查找省份简称
    province_char = None
    province_index = -1
    
    for i, char in enumerate(chars):
        if char in PROVINCE_ABBREVIATIONS:
            province_char = char
            province_index = i
            break
    
    if not province_char:
        # 如果没有找到省份，尝试查找汉字
        for i, char in enumerate(chars):
            if '\u4e00' <= char <= '\u9fa5':
                province_char = char
                province_index = i
                break
    
    if not province_char:
        return None
    
    # 优化：检测省份后面连续的字母数字串，整体移到前面
    # 例如："J9963新AF" -> "新AFJ9963"
    prefix_pattern = None
    prefix_end_index = province_index
    
    # 从省份后面开始，收集所有连续的字母和数字
    if province_index + 1 < len(chars):
        prefix_chars = [province_char]
        prefix_end_index = province_index
        
        # 收集省份后面连续的字母和数字（最多6个字符，因为车牌后5-6位）
        i = province_index + 1
        while i < len(chars) and len(prefix_chars) < 7:  # 省份(1) + 最多6个字符
            char = chars[i].upper()
            # 如果是字母或数字，就包含它
            if char.isalnum():
                prefix_chars.append(char)
                prefix_end_index = i
                i += 1
            else:
                # 遇到非字母数字字符，停止收集
                break
        
        # 如果收集到了至少一个字符（省份+至少1个字母/数字），构建前缀模式
        if len(prefix_chars) >= 2:
            prefix_pattern = ''.join(prefix_chars)
    
    # 如果找到了有效的前缀模式，保持它作为整体并移到前面
    if prefix_pattern and len(prefix_pattern) >= 2:
        # 前缀模式已找到，如"新AF"、"新A123"等
        corrected_chars = []
        
        # 处理前缀模式：省份 + 后面的字母数字串
        # 第一位：省份（保持不变）
        corrected_chars.append(prefix_pattern[0])
        
        # 第二位：必须是字母（车牌规则）
        if len(prefix_pattern) >= 2:
            second_char = prefix_pattern[1]
            # 处理"口"被误识别为D的情况（"口"是汉字，需要转换为D）
            if second_char == '口':
                second_char = 'D'
            else:
                second_char = second_char.upper()
                # 错误校正：I->L, O->D, 0->D, 1->L
                if second_char == 'I':
                    second_char = 'L'
                elif second_char == 'O':
                    second_char = 'D'
                elif second_char == '0':
                    second_char = 'D'
                elif second_char == '1':
                    second_char = 'L'
            
            if second_char.isalpha() and second_char not in ['I', 'O']:
                corrected_chars.append(second_char)
            elif second_char.isdigit():
                digit_to_letter = {'0': 'D', '1': 'L', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
                corrected_chars.append(digit_to_letter.get(second_char, 'A'))
            else:
                corrected_chars.append('A')
        
        # 第三位及以后：保持原样（因为它们是原文本项的一部分，不能拆分）
        # 但需要进行OCR错误校正（O->0, I->1）
        # 注意：后5-6位可以是字母或数字，所以保持原样
        if len(prefix_pattern) >= 3:
            for char in prefix_pattern[2:]:
                char_upper = char.upper()
                # OCR错误校正：O->0, I->1（但保持原文本项的完整性）
                if char_upper == 'O':
                    corrected_chars.append('0')
                elif char_upper == 'I':
                    corrected_chars.append('1')
                elif char_upper.isalnum():  # 字母或数字都直接添加
                    corrected_chars.append(char_upper)
        
        # 收集剩余字符（排除已使用的前缀字符），这些字符将放在前缀后面
        # 前缀部分是从 province_index 到 prefix_end_index（包含）
        remaining_chars = []
        for i, char in enumerate(chars):
            # 跳过前缀部分（从省份开始到前缀结束）
            if province_index <= i <= prefix_end_index:
                continue  # 跳过前缀部分
            if char not in PROVINCE_ABBREVIATIONS:
                remaining_chars.append(char)
        
        # 将剩余字符添加到后面（处理后5-6位）
        for char in remaining_chars[:6]:
            char_upper = char.upper()
            if char_upper == 'O':
                corrected_chars.append('0')
            elif char_upper == 'I':
                corrected_chars.append('1')
            elif char_upper.isdigit():
                corrected_chars.append(char_upper)
            elif char_upper.isalpha() and char_upper not in ['I', 'O']:
                corrected_chars.append(char_upper)
        
        result_text = ''.join(corrected_chars)
        
        if _text_correction_logger.isEnabledFor(logging.INFO):
            _text_correction_logger.info(f"  规则后处理: '{text}' -> '{result_text}'")
            if _text_correction_logger.isEnabledFor(logging.DEBUG):
                _text_correction_logger.debug(f"    前缀模式: '{prefix_pattern}', 剩余字符: {remaining_chars}, 最终字符: {corrected_chars}")
        
        # 验证和纠正
        validated_result = validate_and_correct_license_plate(result_text)
        if validated_result:
            return validated_result
        else:
            # 如果验证失败，记录调试信息
            if _text_correction_logger.isEnabledFor(logging.DEBUG):
                _text_correction_logger.debug(f"  验证失败: '{result_text}' 不符合车牌格式")
            return None
    
    # 如果没有找到有效的前缀模式，使用原来的逻辑
    # 重新组织：省份放在最前面
    corrected_chars = [province_char]
    
    # 优先查找省份后面紧跟的字母（作为第二位）
    second_char = None
    second_char_index = -1
    
    # 查找省份后面紧跟的字母
    if province_index + 1 < len(chars):
        next_char = chars[province_index + 1]
        # 处理"口"被误识别为D的情况（"口"是汉字，需要转换为D）
        if next_char == '口':
            second_char = 'D'
            second_char_index = province_index + 1
        else:
            next_char_upper = next_char.upper()
            # 如果是字母（A-Z，不包括I和O），或者是I/O需要转换
            if next_char_upper.isalpha() or next_char_upper in ['I', 'O', '0', '1']:
                second_char = next_char_upper
                second_char_index = province_index + 1
    
    # 如果没有找到省份后面的字母，查找其他位置的字母
    if second_char is None:
        for i, char in enumerate(chars):
            if i == province_index:
                continue
            if char in PROVINCE_ABBREVIATIONS:
                continue
            # 处理"口"被误识别为D的情况
            if char == '口':
                second_char = 'D'
                second_char_index = i
                break
            char_upper = char.upper()
            if char_upper.isalpha() or char_upper in ['I', 'O', '0', '1']:
                second_char = char_upper
                second_char_index = i
                break
    
    if second_char is None:
        return None
    
    # 处理第二位字母（错误校正）
    # 处理"口"被误识别为D的情况（"口"是汉字，需要转换为D）
    if second_char == '口':
        second_char = 'D'
    elif second_char == 'I':
        second_char = 'L'
    elif second_char == 'O':
        second_char = 'D'
    elif second_char == '0':
        second_char = 'D'
    elif second_char == '1':
        second_char = 'L'
    
    if second_char.isalpha() and second_char not in ['I', 'O']:
        corrected_chars.append(second_char)
    elif second_char.isdigit():
        digit_to_letter = {'0': 'D', '1': 'L', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
        corrected_chars.append(digit_to_letter.get(second_char, 'A'))
    else:
        corrected_chars.append('A')
    
    # 收集剩余字符（排除省份和第二位字母）
    remaining_chars = []
    for i, char in enumerate(chars):
        if i == province_index:
            continue  # 跳过省份
        if i == second_char_index:
            continue  # 跳过第二位字母
        if char not in PROVINCE_ABBREVIATIONS:
            remaining_chars.append(char)
    
    # 处理后5-6位
    for char in remaining_chars[:6]:
        char_upper = char.upper()
        if char_upper == 'O':
            corrected_chars.append('0')
        elif char_upper == 'I':
            corrected_chars.append('1')
        elif char_upper.isdigit():
            corrected_chars.append(char_upper)
        elif char_upper.isalpha() and char_upper not in ['I', 'O']:
            corrected_chars.append(char_upper)
    
    result_text = ''.join(corrected_chars)
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"  规则后处理: '{text}' -> '{result_text}'")
    
    # 验证和纠正
    return validate_and_correct_license_plate(result_text)


def correct_license_plate_text_v4(ocr_result: Dict) -> str:
    """
    文本校正函数 V4
    
    采用4步处理流程：
    1. 过滤与置信度筛选 (Heuristic Filtering)
    2. 空间布局聚类 (Spatial Grouping)
    3. 排序与拼接 (Sorting & Concatenation)
    4. 基于业务规则的后处理 (Rule-based Refinement)
    
    Args:
        ocr_result: OCR识别结果字典，包含：
            {
                'texts': List[str],  # 文本列表
                'details': List[Dict],  # 详细信息列表，每个元素包含
                    {
                        'text': str,  # 识别到的文本
                        'box': List,  # 坐标框
                        'score': float  # 置信度
                    }
            }
    
    Returns:
        校正后的车牌文本
    """
    if not ocr_result:
        return ''
    
    texts = ocr_result.get('texts', [])
    details = ocr_result.get('details', [])
    
    if not texts or not details:
        return ''
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[文本校正V4] 开始处理，共 {len(texts)} 个文本项")
    
    # ========== 步骤1: 过滤与置信度筛选 ==========
    confidence_threshold = 0.5  # 置信度阈值
    
    filtered_detections = []
    seen_corrected_texts = set()  # 用于去重（使用纠正后的文本）
    
    for detail in details:
        text = detail.get('text', '').strip()
        box = detail.get('box', [])
        score = detail.get('score', 0.0)
        
        if not text or not box:
            continue
        
        # 过滤低置信度的检测结果
        if score < confidence_threshold:
            if _text_correction_logger.isEnabledFor(logging.DEBUG):
                _text_correction_logger.debug(f"  过滤低置信度: '{text}' (score: {score:.4f})")
            continue
        
        # 步骤1：清除无效符号（如'-'、'·'、空格等）
        cleaned_text = clean_text(text)
        
        # 如果清理后文本为空，过滤掉
        if not cleaned_text:
            if _text_correction_logger.isEnabledFor(logging.DEBUG):
                _text_correction_logger.debug(f"  过滤空文本（清理后）: '{text}' -> ''")
            continue
        
        # 早期纠正"口"->"D"（在车牌第二位位置，"口"很可能是D的误识别）
        # 检查是否包含"口"字符，如果在第二位或后面位置，转换为"D"
        if '口' in cleaned_text:
            # 如果"口"在文本的开头（可能是省份简称），保留；否则转换为"D"
            if cleaned_text.startswith('口'):
                # "口"在开头，可能是省份简称的误识别，但通常省份简称不会是"口"
                # 这里保守处理：如果后面还有字符，将"口"转换为"D"
                if len(cleaned_text) > 1:
                    cleaned_text = 'D' + cleaned_text[1:]
                    if _text_correction_logger.isEnabledFor(logging.DEBUG):
                        _text_correction_logger.debug(f"  纠正开头'口'->'D': '{text}' -> '{cleaned_text}'")
                # 如果只有"口"一个字符，保留它（可能是误识别，但先不过滤）
            else:
                # "口"不在开头，转换为"D"
                cleaned_text = cleaned_text.replace('口', 'D')
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  纠正'口'->'D': '{text}' -> '{cleaned_text}'")
        
        # 检查文本是否只包含有效字符，如果不包含，尝试提取有效部分
        if not contains_only_valid_chars(cleaned_text):
            # 尝试从文本中提取有效部分（如从"赣日"提取"赣"）
            extracted_part = extract_valid_part(cleaned_text)
            if extracted_part and len(extracted_part) > 0:
                # 如果提取到了有效部分，使用提取的部分
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  从文本中提取有效部分: '{text}' -> '{cleaned_text}' -> '{extracted_part}'")
                cleaned_text = extracted_part
            else:
                # 如果无法提取有效部分，过滤掉
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  过滤包含无效字符的文本: '{text}' -> '{cleaned_text}' (无法提取有效部分)")
                continue
        
        # 先纠正OCR错误（I->1, O->0），用于去重检查
        # 使用简单纠正函数，不区分位置
        corrected_text = correct_ocr_errors_simple(cleaned_text)
        
        # 过滤单个字符且不是有效省份简称的文本项（如'量'、'峡'等）
        if len(corrected_text) == 1:
            # 检查是否是有效省份简称
            if corrected_text not in PROVINCE_ABBREVIATIONS:
                if _text_correction_logger.isEnabledFor(logging.DEBUG):
                    _text_correction_logger.debug(f"  过滤无效省份简称: '{text}' -> '{corrected_text}' (不是有效省份简称)")
                continue
        
        # 过滤重复的文本项（使用纠正后的文本进行去重检查，保留第一个，后续重复的过滤掉）
        if corrected_text in seen_corrected_texts:
            if _text_correction_logger.isEnabledFor(logging.DEBUG):
                _text_correction_logger.debug(f"  过滤重复文本项: '{text}' -> '{corrected_text}' (纠正后重复)")
            continue
        
        seen_corrected_texts.add(corrected_text)
        # 保留清理后的文本用于后续处理（无效符号已在步骤1清除）
        filtered_detections.append({
            'text': cleaned_text,  # 使用清理后的文本
            'box': box,
            'score': score
        })
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[步骤1] 置信度筛选后: {len(filtered_detections)} 个检测结果")
    
    if not filtered_detections:
        return ''
    
    # 非极大值抑制 (NMS)
    nms_detections = nms(filtered_detections, iou_threshold=0.5)
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[步骤1] NMS后: {len(nms_detections)} 个检测结果")
    
    if not nms_detections:
        return ''
    
    # ========== 步骤2: 空间布局聚类 ==========
    groups = spatial_grouping(nms_detections, y_threshold_ratio=0.5)
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[步骤2] 空间聚类后: {len(groups)} 个组")
        for i, group in enumerate(groups):
            texts_in_group = [det.get('text', '') for det in group]
            _text_correction_logger.info(f"  组 {i+1}: {texts_in_group}")
    
    if not groups:
        return ''
    
    # ========== 步骤3: 排序与拼接 ==========
    complete_plates = []  # 完整车牌（单个文本项本身就是完整车牌）
    concatenated_candidates = []  # 拼接结果（需要拼接多个文本项）
    
    for group in groups:
        # 首先检查每个文本项本身是否是有效车牌（去除无效符号后）
        valid_single_items = []
        for det in group:
            text = det.get('text', '').strip()
            if not text:
                continue
            
            validated = check_single_item_validity(text)
            if validated:
                score = det.get('score', 0.0)
                valid_single_items.append((validated, score))
                if _text_correction_logger.isEnabledFor(logging.INFO):
                    _text_correction_logger.info(f"  文本项本身是有效车牌: '{text}' -> '{validated}' (置信度: {score:.4f})")
        
        # 如果找到有效的单独文本项，作为完整车牌
        if valid_single_items:
            # 选择置信度最高的完整车牌
            best_single = max(valid_single_items, key=lambda x: x[1])
            complete_plates.append(best_single)
            if _text_correction_logger.isEnabledFor(logging.INFO):
                _text_correction_logger.info(f"  使用完整车牌: '{best_single[0]}' (置信度: {best_single[1]:.4f})")
        else:
            # 如果单独文本项都不是有效车牌，才进行拼接
            concatenated = sort_and_concatenate(group, check_rotation=False)
            if concatenated:
                avg_score = calculate_avg_score(group)
                concatenated_candidates.append((concatenated, avg_score))
                if _text_correction_logger.isEnabledFor(logging.INFO):
                    _text_correction_logger.info(f"  拼接文本: '{concatenated}' (平均置信度: {avg_score:.4f})")
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[步骤3] 排序拼接后: {len(complete_plates)} 个完整车牌, {len(concatenated_candidates)} 个拼接候选")
        for text, score in complete_plates:
            _text_correction_logger.info(f"  完整车牌: '{text}' (置信度: {score:.4f})")
        for text, score in concatenated_candidates:
            _text_correction_logger.info(f"  拼接候选: '{text}' (置信度: {score:.4f})")
    
    # ========== 步骤4: 基于业务规则的后处理 ==========
    # 优先处理完整车牌：如果有完整车牌，直接选择第一个（或置信度最高的）
    if complete_plates:
        # 选择置信度最高的完整车牌
        best_complete = max(complete_plates, key=lambda x: x[1])
        refined = rule_based_refinement(best_complete[0])
        if refined:
            if _text_correction_logger.isEnabledFor(logging.INFO):
                _text_correction_logger.info(f"[步骤4] 找到完整车牌: '{best_complete[0]}' -> '{refined}' (置信度: {best_complete[1]:.4f})")
                _text_correction_logger.info(f"[文本校正V4] 最终结果: '{refined}'")
            return refined
    
    # 如果没有完整车牌，处理拼接候选结果
    best_result = None
    best_score = 0.0
    
    for text, score in concatenated_candidates:
        refined = rule_based_refinement(text)
        if refined:
            if score > best_score:
                best_result = refined
                best_score = score
                if _text_correction_logger.isEnabledFor(logging.INFO):
                    _text_correction_logger.info(f"[步骤4] 找到有效车牌: '{text}' -> '{refined}' (置信度: {score:.4f})")
    
    if best_result:
        if _text_correction_logger.isEnabledFor(logging.INFO):
            _text_correction_logger.info(f"[文本校正V4] 最终结果: '{best_result}'")
        return best_result
    
    # 如果步骤4后无法找到有效车牌，尝试跨组组合
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[步骤4] 未找到有效车牌，尝试跨组组合...")
    
    # 收集所有组的文本（不清理，直接使用原始文本）
    group_texts_list = []
    group_scores_list = []
    
    for group in groups:
        group_texts = [det.get('text', '').strip() for det in group]
        combined_text = ''.join(group_texts)
        
        if combined_text:
            group_texts_list.append(combined_text)
            group_scores_list.append(calculate_avg_score(group))
    
    # 如果只有1个或0个组，无法组合
    if len(group_texts_list) < 2:
        if _text_correction_logger.isEnabledFor(logging.INFO):
            _text_correction_logger.info(f"[文本校正V4] 未找到有效车牌")
        return ''
    
    # 尝试跨组组合（先尝试2个组，再尝试3个组）
    cross_result, cross_score = try_cross_group_combination(group_texts_list, group_scores_list, max_groups=2)
    if cross_result and cross_score > best_score:
        best_result = cross_result
        best_score = cross_score
    
    if not best_result:
        cross_result, cross_score = try_cross_group_combination(group_texts_list, group_scores_list, max_groups=3)
        if cross_result and cross_score > best_score:
            best_result = cross_result
            best_score = cross_score
    
    if best_result:
        if _text_correction_logger.isEnabledFor(logging.INFO):
            _text_correction_logger.info(f"[文本校正V4] 最终结果（跨组组合）: '{best_result}'")
        return best_result
    
    if _text_correction_logger.isEnabledFor(logging.INFO):
        _text_correction_logger.info(f"[文本校正V4] 未找到有效车牌")
    return ''


# ============================================================================
# 车牌关键点透视变换工具（从 utils_kp.py 合并）
# ============================================================================

def expand_keypoints(
    keypoints: np.ndarray,
    image_shape: Tuple[int, int],
    expand_ratio: float = 0.1,
    expand_pixels: Optional[int] = None
) -> np.ndarray:
    """
    基于关键点扩大车牌区域
    
    通过将关键点向外扩展来扩大车牌区域，扩展方式：
    1. 计算关键点的中心点
    2. 将每个关键点从中心点向外移动一定距离
    
    Args:
        keypoints: 关键点数组，shape为 (4, 2) 或 (4, 3)，顺序为 [左上, 右上, 右下, 左下]
        image_shape: 图像尺寸 (height, width)
        expand_ratio: 扩大比例，默认 0.1 (10%)，相对于关键点形成的四边形尺寸
        expand_pixels: 固定像素扩大值，如果指定则优先使用此值（None 表示不使用固定值）
    
    Returns:
        扩大后的关键点数组，shape与输入相同
    """
    if keypoints is None or len(keypoints) != 4:
        return keypoints
    
    # 转换为numpy数组
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints, dtype=np.float32)
    else:
        keypoints = keypoints.astype(np.float32)
    
    # 提取x, y坐标（忽略可能的visibility）
    if keypoints.shape[1] >= 2:
        points = keypoints[:, :2].copy()
    else:
        return keypoints
    
    h, w = image_shape
    
    # 计算关键点的中心点
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    # 计算关键点形成的四边形的平均尺寸（用于比例扩大）
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    left_height = np.linalg.norm(points[3] - points[0])
    right_height = np.linalg.norm(points[2] - points[1])
    avg_width = (top_width + bottom_width) / 2.0
    avg_height = (left_height + right_height) / 2.0
    
    # 计算扩大值
    if expand_pixels is not None and expand_pixels > 0:
        # 使用固定像素值
        expand_x = float(expand_pixels)
        expand_y = float(expand_pixels)
    else:
        # 使用比例扩大
        expand_x = avg_width * expand_ratio
        expand_y = avg_height * expand_ratio
    
    # 扩大每个关键点：从中心点向外移动
    expanded_points = points.copy()
    for i in range(len(points)):
        # 计算从中心点到当前关键点的方向向量
        dx = points[i, 0] - center_x
        dy = points[i, 1] - center_y
        
        # 计算距离
        distance = np.sqrt(dx * dx + dy * dy)
        if distance > 0:
            # 归一化方向向量
            dx_norm = dx / distance
            dy_norm = dy / distance
            
            # 计算扩展距离：根据方向向量的分量来分配扩展值
            # 水平方向扩展：根据x方向的分量
            # 垂直方向扩展：根据y方向的分量
            expand_dist_x = expand_x * abs(dx_norm)
            expand_dist_y = expand_y * abs(dy_norm)
            
            # 扩展关键点：沿从中心点向外的方向移动
            expanded_points[i, 0] = points[i, 0] + dx_norm * expand_dist_x
            expanded_points[i, 1] = points[i, 1] + dy_norm * expand_dist_y
        else:
            # 如果距离为0（理论上不应该发生），使用平均扩展
            # 判断关键点位置
            is_right = points[i, 0] >= center_x
            is_bottom = points[i, 1] >= center_y
            expanded_points[i, 0] = points[i, 0] + expand_x if is_right else points[i, 0] - expand_x
            expanded_points[i, 1] = points[i, 1] + expand_y if is_bottom else points[i, 1] - expand_y
    
    # 裁剪到图像范围内
    expanded_points[:, 0] = np.clip(expanded_points[:, 0], 0, w - 1)
    expanded_points[:, 1] = np.clip(expanded_points[:, 1], 0, h - 1)
    
    # 如果原始关键点有第3列（visibility），保留它
    if keypoints.shape[1] == 3:
        visibility = keypoints[:, 2:3]
        expanded_keypoints = np.hstack([expanded_points, visibility])
    else:
        expanded_keypoints = expanded_points
    
    return expanded_keypoints


def calculate_destination_size(src_points: np.ndarray) -> Tuple[int, int]:
    """
    根据源关键点计算目标矩形的尺寸
    
    Args:
        src_points: 源关键点，shape为 (4, 2)，顺序为 [左上, 右上, 右下, 左下]
    
    Returns:
        (width, height) 目标矩形的宽高
    """
    # 计算上边和下边的长度
    top_width = np.linalg.norm(src_points[1] - src_points[0])
    bottom_width = np.linalg.norm(src_points[2] - src_points[3])
    avg_width = int((top_width + bottom_width) / 2)
    
    # 计算左边和右边的高度
    left_height = np.linalg.norm(src_points[3] - src_points[0])
    right_height = np.linalg.norm(src_points[2] - src_points[1])
    avg_height = int((left_height + right_height) / 2)
    
    # 确保最小尺寸
    avg_width = max(avg_width, 10)
    avg_height = max(avg_height, 10)
    
    return avg_width, avg_height


def perspective_transform_plate(
    detection_result: Dict,
    detection_index: int = 0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    利用关键点对车牌区域进行透视变换，校正为矩形
    
    Args:
        detection_result: detector_kp.py 的检测结果字典
        detection_index: 要处理的检测结果索引（默认0，即第一个检测结果）
        target_width: 目标宽度（如果为None，则根据关键点自动计算）
        target_height: 目标高度（如果为None，则根据关键点自动计算）
        padding: 目标图像周围的填充像素（默认0）
    
    Returns:
        透视变换后的车牌区域图像，如果失败则返回None
    """
    # 检查检测结果
    if detection_result.get('count', 0) == 0:
        return None
    
    if detection_index >= detection_result['count']:
        return None
    
    # 获取关键点
    keypoints = detection_result.get('keypoints', [])
    if not keypoints or detection_index >= len(keypoints):
        return None
    
    keypoint = keypoints[detection_index]
    if len(keypoint) != 4:
        return None
    
    # 获取对应的角度和旋转图像
    angles = detection_result.get('angles', [])
    rotated_images = detection_result.get('rotated_images', {})
    
    if detection_index >= len(angles):
        return None
    
    angle = angles[detection_index]
    if angle not in rotated_images:
        return None
    
    rotated_image = rotated_images[angle]
    
    # 转换关键点为numpy数组
    if isinstance(keypoint, list):
        keypoint = np.array(keypoint, dtype=np.float32)
    else:
        keypoint = keypoint.astype(np.float32)
    
    # 提取x, y坐标（忽略可能的visibility）
    if keypoint.shape[1] >= 2:
        src_points = keypoint[:, :2]
    else:
        return None
    
    # 计算目标尺寸
    if target_width is None or target_height is None:
        calc_width, calc_height = calculate_destination_size(src_points)
        target_width = target_width if target_width is not None else calc_width
        target_height = target_height if target_height is not None else calc_height
    
    # 定义目标矩形的四个角点（左上、右上、右下、左下）
    dst_points = np.float32([
        [padding, padding],  # 左上
        [target_width - 1 - padding, padding],  # 右上
        [target_width - 1 - padding, target_height - 1 - padding],  # 右下
        [padding, target_height - 1 - padding]  # 左下
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 执行透视变换
    warped = cv2.warpPerspective(
        rotated_image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return warped


def batch_perspective_transform(
    detection_result: Dict,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    padding: int = 0
) -> List[Optional[np.ndarray]]:
    """
    批量对检测结果中的所有车牌进行透视变换
    
    Args:
        detection_result: detector_kp.py 的检测结果字典
        target_width: 目标宽度（如果为None，则根据关键点自动计算）
        target_height: 目标高度（如果为None，则根据关键点自动计算）
        padding: 目标图像周围的填充像素（默认0）
    
    Returns:
        透视变换后的车牌区域图像列表，每个元素对应一个检测结果
    """
    count = detection_result.get('count', 0)
    results = []
    
    for i in range(count):
        warped = perspective_transform_plate(
            detection_result,
            detection_index=i,
            target_width=target_width,
            target_height=target_height,
            padding=padding
        )
        results.append(warped)
    
    return results


def get_best_perspective_transform(
    detection_result: Dict,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    获取最佳检测结果的透视变换结果（选择离图像中心最近的检测框）
    
    选择方法代码在 utils.py 中（select_center_detection_from_detector_result）
    
    Args:
        detection_result: detector_kp.py 的检测结果字典
        target_width: 目标宽度（如果为None，则根据关键点自动计算）
        target_height: 目标高度（如果为None，则根据关键点自动计算）
        padding: 目标图像周围的填充像素（默认0）
    
    Returns:
        透视变换后的车牌区域图像，如果失败则返回None
    """
    # 选择离图像中心最近的检测框（使用 utils.py 中的函数）
    best_detection = select_center_detection_from_detector_result(detection_result)
    if best_detection is None:
        return None
    
    # 找到对应的索引
    boxes = detection_result.get('boxes', [])
    best_box = best_detection.get('box')
    
    if not boxes or best_box is None:
        return None
    
    # 找到最佳检测框的索引
    best_index = None
    for i, box in enumerate(boxes):
        if len(box) == 4 and len(best_box) == 4:
            if abs(box[0] - best_box[0]) < 1e-5 and abs(box[1] - best_box[1]) < 1e-5:
                best_index = i
                break
    
    if best_index is None:
        return None
    
    return perspective_transform_plate(
        detection_result,
        detection_index=best_index,
        target_width=target_width,
        target_height=target_height,
        padding=padding
    )


def visualize_keypoints_and_box(
    image: np.ndarray,
    box: List[float],
    keypoints: np.ndarray,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化检测框和关键点
    
    Args:
        image: 图像
        box: 检测框 [x1, y1, x2, y2]
        keypoints: 关键点数组，shape为 (4, 2) 或 (4, 3)
        save_path: 保存路径（可选）
    
    Returns:
        绘制了检测框和关键点的图像
    """
    vis_image = image.copy()
    
    # 绘制检测框
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 确保 keypoints 是 numpy 数组
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints, dtype=np.float32)
    elif not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints, dtype=np.float32)
    
    # 提取关键点坐标
    if len(keypoints) == 0 or keypoints.shape[0] == 0:
        return vis_image
    
    if keypoints.shape[1] >= 2:
        points = keypoints[:, :2].astype(int)
    else:
        return vis_image
    
    # 绘制关键点
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 蓝、绿、红、青
    labels = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    
    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        x, y = point
        cv2.circle(vis_image, (x, y), 5, color, -1)
        cv2.putText(vis_image, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 连接关键点（形成四边形）
    if len(points) == 4:
        pts = points.reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], True, (255, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image


# --- 运行测试 ---
if __name__ == '__main__':
    pass
