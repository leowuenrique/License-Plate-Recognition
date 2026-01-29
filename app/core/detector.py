"""
YOLO11-pose 车牌检测器
支持检测框和关键点检测
"""

from ultralytics import YOLO
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import cv2
import numpy as np
import time


class LicensePlatePoseDetector:
    """车牌检测器（基于YOLO11-pose），支持检测框和关键点检测"""
    
    def __init__(self, model_path: str,
                 stop_on_first_detection: bool = False,
                 use_gpu: bool = True,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        初始化检测器
        
        Args:
            model_path: YOLO11-pose模型文件路径
            stop_on_first_detection: 如果为True，多角度检测时一旦检测到车牌就停止后续角度检测
            use_gpu: 是否使用GPU，如果为False则强制使用CPU
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值（用于NMS）
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.stop_on_first_detection = stop_on_first_detection
        self.use_gpu = use_gpu
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 如果禁用GPU，设置设备为CPU
        if not use_gpu:
            self.model.to('cpu')
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            
        Returns:
            (旋转后的图像, 变换矩阵)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
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
        
        # 执行旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
        
        return rotated_image, rotation_matrix
    
    def _transform_points_to_original(self, points: np.ndarray, rotation_matrix: np.ndarray,
                                     original_shape: Tuple[int, int]) -> np.ndarray:
        """
        将旋转图像上的点坐标转换回原始图像坐标系
        
        Args:
            points: 点坐标数组，shape为 (N, 2) 或 (N, 3)
            rotation_matrix: 旋转矩阵
            original_shape: 原始图像尺寸 (height, width)
            
        Returns:
            转换后的点坐标
        """
        if len(points) == 0:
            return points
        
        # 计算逆变换矩阵
        inv_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
        
        # 提取x, y坐标（忽略可能的visibility）
        if points.shape[1] >= 2:
            xy_points = points[:, :2]
        else:
            return points
        
        # 转换为齐次坐标
        ones = np.ones((xy_points.shape[0], 1))
        points_homogeneous = np.hstack([xy_points, ones])
        
        # 应用逆变换
        transformed_points = (inv_rotation_matrix @ points_homogeneous.T).T
        
        # 如果原始points有第3列（visibility），保留它
        if points.shape[1] == 3:
            visibility = points[:, 2:3]
            transformed_points = np.hstack([transformed_points, visibility])
        
        # 裁剪到原始图像范围内
        h_orig, w_orig = original_shape
        transformed_points[:, 0] = np.clip(transformed_points[:, 0], 0, w_orig)
        transformed_points[:, 1] = np.clip(transformed_points[:, 1], 0, h_orig)
        
        return transformed_points
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """
        计算两个检测框的IoU (Intersection over Union)
        
        Args:
            box1: 检测框坐标 [x1, y1, x2, y2]
            box2: 检测框坐标 [x1, y1, x2, y2]
        
        Returns:
            IoU值 (0-1)
        """
        x1_min, y1_min, x1_max, y1_max = box1
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
    
    def _transform_boxes_to_original(self, boxes: np.ndarray, rotation_matrix: np.ndarray, 
                                    original_shape: Tuple[int, int]) -> np.ndarray:
        """
        将旋转图像上的检测框坐标转换回原始图像坐标系
        
        Args:
            boxes: 检测框坐标 (x1, y1, x2, y2)
            rotation_matrix: 旋转矩阵
            original_shape: 原始图像尺寸 (height, width)
            
        Returns:
            转换后的检测框坐标
        """
        if len(boxes) == 0:
            return boxes
        
        # 计算逆变换矩阵
        inv_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
        
        h_orig, w_orig = original_shape
        
        transformed_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # 将旋转图像上的四个角点转换回原始坐标系
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
            x_min = max(0, min(w_orig, x_min))
            x_max = max(0, min(w_orig, x_max))
            y_min = max(0, min(h_orig, y_min))
            y_max = max(0, min(h_orig, y_max))
            
            transformed_boxes.append([x_min, y_min, x_max, y_max])
        
        return np.array(transformed_boxes)
    
    def detect(self, image_path: str, angles: Optional[List[float]] = None) -> Dict:
        """
        检测图片中的车牌，支持任意角度旋转检测，返回检测框和关键点
        
        Args:
            image_path: 图片路径
            angles: 要检测的旋转角度列表，如果为None则使用默认角度 [0, 90, 180, 270]
            
        Returns:
            检测结果字典，包含所有角度检测到的车牌信息：
            {
                'boxes': [[x1, y1, x2, y2], ...],  # 检测框坐标（旋转图像坐标系）
                'keypoints': [[[x1, y1], [x2, y2], ...], ...],  # 关键点坐标（旋转图像坐标系），每个检测框对应一组关键点
                'scores': [score1, score2, ...],  # 置信度
                'classes': [cls1, cls2, ...],  # 类别ID
                'angles': [angle1, angle2, ...],  # 每个检测框对应的检测角度
                'count': int,  # 检测到的车牌数量
                'image': np.ndarray,  # 原始图像
                'rotated_images': {angle: np.ndarray, ...}  # 旋转后的图像字典，键为角度，值为旋转后的图像
            }
            
        Note:
            返回的检测框和关键点坐标都是基于旋转后图像坐标系的，不进行坐标转换。
            每个检测结果对应的旋转图像可以通过 'rotated_images' 字典获取。
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 默认检测角度
        if angles is None:
            angles = [0, 90, 180, 270]
        
        # 合并所有角度的检测结果
        all_boxes = []
        all_keypoints = []
        all_scores = []
        all_classes = []
        detected_angles = []
        rotated_images = {}  # 保存每个角度对应的旋转图像
        rotation_matrices = {}  # 保存每个角度对应的旋转矩阵，用于坐标转换
        
        # 在每个角度上检测
        for angle in angles:
            # 旋转图像
            if angle == 0:
                rotated_image = image.copy()
                rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            else:
                rotated_image, rotation_matrix = self._rotate_image(image, angle)
            
            # 保存旋转后的图像和旋转矩阵
            rotated_images[angle] = rotated_image.copy()
            rotation_matrices[angle] = rotation_matrix
            
            # 执行检测
            results = self.model.predict(
                source=rotated_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # 解析检测结果
            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                boxes = result.boxes
                
                # 获取检测框（旋转图像坐标系，不转换）
                rotated_boxes = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                # 获取关键点（如果存在，旋转图像坐标系，不转换）
                keypoints = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    # keypoints.data shape: [num_detections, num_keypoints, 3] (x, y, visibility)
                    # 或 [num_detections, num_keypoints, 2] (x, y)
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    keypoints = keypoints_data
                
                # 直接使用旋转图像上的坐标，不转换回原图
                if len(rotated_boxes) > 0:
                    all_boxes.extend(rotated_boxes.tolist())
                    all_scores.extend(scores.tolist())
                    all_classes.extend(classes.tolist())
                    detected_angles.extend([angle] * len(rotated_boxes))
                    
                    # 处理关键点（直接使用旋转图像上的坐标）
                    if keypoints is not None:
                        if len(keypoints) > 0:
                            all_keypoints.extend(keypoints.tolist())
                    else:
                        # 如果没有关键点，为每个检测框添加空的关键点列表
                        all_keypoints.extend([[]] * len(rotated_boxes))
                    
                    # 如果启用了"检测到车牌就停止"功能，且已检测到车牌，则停止后续角度检测
                    if self.stop_on_first_detection and len(all_boxes) > 0:
                        break
        
        # 去重：将所有检测框转换回原图坐标系，使用IoU去重
        if len(all_boxes) > 0:
            # 将所有检测框转换回原图坐标系
            original_boxes = []
            for i, (box, angle) in enumerate(zip(all_boxes, detected_angles)):
                rotation_matrix = rotation_matrices[angle]
                original_shape = image.shape[:2]
                # 转换单个检测框
                transformed_box = self._transform_boxes_to_original(
                    np.array([box]), rotation_matrix, original_shape
                )[0]
                original_boxes.append(transformed_box.tolist())
            
            # 构建检测结果列表用于NMS
            detections = []
            for i in range(len(original_boxes)):
                detections.append({
                    'box': original_boxes[i],
                    'score': all_scores[i],
                    'class': all_classes[i] if i < len(all_classes) else 0,
                    'angle': detected_angles[i],
                    'keypoints': all_keypoints[i] if i < len(all_keypoints) else [],
                    'rotated_box': all_boxes[i],  # 保留旋转图像坐标系的检测框
                    'index': i
                })
            
            # 使用NMS去重（基于原图坐标系的IoU）
            # 按置信度降序排序
            sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            
            keep_indices = []
            iou_threshold = 0.5  # IoU阈值，超过此值认为是重复检测
            
            while sorted_detections:
                # 保留置信度最高的
                current = sorted_detections.pop(0)
                keep_indices.append(current['index'])
                
                # 移除与当前框IoU过高的框（认为是同一个车牌）
                sorted_detections = [
                    det for det in sorted_detections
                    if self._calculate_iou(current['box'], det['box']) < iou_threshold
                ]
            
            # 按置信度降序保留去重后的结果（保持NMS后的顺序）
            # keep_indices 已经是按置信度降序排列的
            filtered_boxes = [all_boxes[i] for i in keep_indices]
            filtered_keypoints = [all_keypoints[i] for i in keep_indices]
            filtered_scores = [all_scores[i] for i in keep_indices]
            filtered_classes = [all_classes[i] for i in keep_indices]
            filtered_angles = [detected_angles[i] for i in keep_indices]
            
            all_boxes = filtered_boxes
            all_keypoints = filtered_keypoints
            all_scores = filtered_scores
            all_classes = filtered_classes
            detected_angles = filtered_angles
        
        return {
            'boxes': all_boxes,
            'keypoints': all_keypoints,
            'scores': all_scores,
            'classes': all_classes,
            'angles': detected_angles,
            'count': len(all_boxes),
            'image': image,  # 原始图像
            'rotated_images': rotated_images  # 旋转后的图像字典
        }


def select_highest_confidence(detection_result: Dict) -> Optional[Dict]:
    """
    从检测结果中选择置信度最高的检测框
    
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
    
    Returns:
        如果找到检测框，返回包含最高置信度检测框信息的字典：
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
    """
    if not detection_result or detection_result.get('count', 0) == 0:
        return None
    
    boxes = detection_result.get('boxes', [])
    keypoints = detection_result.get('keypoints', [])
    scores = detection_result.get('scores', [])
    classes = detection_result.get('classes', [])
    angles = detection_result.get('angles', [])
    
    if not boxes or not scores:
        return None
    
    # 找到置信度最高的索引
    max_score_idx = scores.index(max(scores))
    
    # 返回最高置信度的检测框信息
    result = {
        'box': boxes[max_score_idx],
        'score': scores[max_score_idx],
    }
    
    # 添加关键点信息（如果存在）
    if keypoints and max_score_idx < len(keypoints):
        result['keypoints'] = keypoints[max_score_idx]
    else:
        result['keypoints'] = None
    
    # 添加类别信息（如果存在）
    if classes and max_score_idx < len(classes):
        result['class'] = classes[max_score_idx]
    
    # 添加角度信息（如果存在）
    detected_angle = None
    if angles and max_score_idx < len(angles):
        detected_angle = angles[max_score_idx]
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


def batch_detect(detector: LicensePlatePoseDetector, directory: str, 
                angles: Optional[List[float]] = None) -> List[Dict]:
    """
    批量检测目录中所有图片，支持任意角度旋转检测
    
    Args:
        detector: LicensePlatePoseDetector 实例
        directory: 图片目录路径
        angles: 要检测的旋转角度列表，如果为None则使用默认角度 [0, 90, 180, 270]
        
    Returns:
        检测结果列表
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 获取目录下所有图片文件
    image_dir = Path(directory)
    if not image_dir.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    image_files = [
        str(p) for p in image_dir.iterdir() 
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"目录中没有找到图片文件: {directory}")
        return []
    
    print(f"找到 {len(image_files)} 张图片，开始批量检测...")
    if angles:
        print(f"检测角度: {angles}")
    
    # 整理结果
    detection_results = []
    total_plates = 0
    detected_images_count = 0
    total_time = 0.0
    time_list = []
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).name
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 使用旋转检测
            result = detector.detect(
                image_path, 
                angles=angles
            )
            
            # 记录结束时间
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            time_list.append(elapsed_time)
            
            result['image_path'] = image_path
            result['image_name'] = image_name
            result['detection_time'] = elapsed_time
            
            plate_count = result['count']
            total_plates += plate_count
            
            # 选择置信度最高的检测框
            best_detection = select_highest_confidence(result)
            if best_detection:
                result['best_detection'] = best_detection
            
            # 统计检测到车牌的图片数量
            if plate_count > 0:
                detected_images_count += 1
            
            detection_results.append(result)
            
            # 打印检测信息
            if best_detection:
                keypoint_info = ""
                if best_detection.get('keypoints') is not None:
                    kp_count = len(best_detection['keypoints'])
                    keypoint_info = f", 关键点: {kp_count}个"
                print(f"[{i}/{len(image_files)}] {image_name}: 检测到 {plate_count} 个车牌, "
                      f"最高置信度: {best_detection['score']:.4f}{keypoint_info}, 耗时 {elapsed_time:.3f}s")
            else:
                print(f"[{i}/{len(image_files)}] {image_name}: 检测到 {plate_count} 个车牌, 耗时 {elapsed_time:.3f}s")
        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            total_time += elapsed_time
            time_list.append(elapsed_time)
            
            print(f"[{i}/{len(image_files)}] {image_name}: 错误 - {str(e)}")
            detection_results.append({
                'image_path': image_path,
                'image_name': image_name,
                'boxes': [],
                'keypoints': [],
                'scores': [],
                'classes': [],
                'angles': [],
                'count': 0,
                'detection_time': elapsed_time,
                'error': str(e),
                'best_detection': None,
                'image': None,
                'rotated_images': {}
            })
    
    # 计算统计信息
    avg_time = total_time / len(image_files) if len(image_files) > 0 else 0
    min_time = min(time_list) if time_list else 0
    max_time = max(time_list) if time_list else 0
    
    # 计算最高置信度统计
    best_scores = []
    for result in detection_results:
        if 'best_detection' in result and result['best_detection']:
            best_scores.append(result['best_detection']['score'])
    
    print(f"\n批量检测完成！")
    print(f"总图片数: {len(image_files)}")
    print(f"检测到车牌的图片数量: {detected_images_count}")
    print(f"总车牌数: {total_plates}")
    if detected_images_count > 0:
        print(f"平均每张有车牌的图片: {total_plates/detected_images_count:.2f} 个车牌")
    if len(image_files) > 0:
        print(f"平均每张图片: {total_plates/len(image_files):.2f} 个车牌")
    
    if best_scores:
        print(f"\n置信度统计:")
        print(f"  最高置信度: {max(best_scores):.4f}")
        print(f"  最低置信度: {min(best_scores):.4f}")
        print(f"  平均置信度: {sum(best_scores)/len(best_scores):.4f}")
    
    print(f"\n时间统计:")
    print(f"  总耗时: {total_time:.3f}s")
    print(f"  平均耗时: {avg_time:.3f}s/张")
    print(f"  最快: {min_time:.3f}s")
    print(f"  最慢: {max_time:.3f}s")
    
    return detection_results


if __name__ == '__main__':
    # 使用示例
    # 初始化检测器
    detector = LicensePlatePoseDetector(
        model_path='models/det/yolo11n-pose-LPR-best.pt',
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # 单张图片检测（带旋转）
    result = detector.detect('test.jpg', angles=[0, 90, 180, 270])
    print("检测结果:")
    print(f"  检测到 {result['count']} 个车牌")
    print(f"  检测框（旋转图像坐标系）: {result['boxes']}")
    print(f"  置信度: {result['scores']}")
    print(f"  检测角度: {result['angles']}")
    if result['keypoints']:
        print(f"  关键点数量: {len(result['keypoints'])}")
        for i, kp in enumerate(result['keypoints']):
            print(f"    车牌 {i+1} 关键点（旋转图像坐标系）: {kp}")
    
    # 显示旋转后的图像信息
    if 'rotated_images' in result:
        print(f"\n旋转后的图像:")
        for angle, rotated_img in result['rotated_images'].items():
            print(f"  角度 {angle}°: 图像尺寸 {rotated_img.shape}")
    
    # 选择置信度最高的检测框
    best_detection = select_highest_confidence(result)
    if best_detection:
        print("\n置信度最高的检测框:")
        print(f"  边界框（旋转图像坐标系）: {best_detection['box']}")
        print(f"  置信度: {best_detection['score']:.4f}")
        if 'class' in best_detection:
            print(f"  类别: {best_detection['class']}")
        if 'angle' in best_detection:
            print(f"  检测角度: {best_detection['angle']}°")
        if best_detection.get('keypoints') is not None:
            print(f"  关键点（旋转图像坐标系）: {best_detection['keypoints']}")
        if 'rotated_image' in best_detection:
            print(f"  对应的旋转图像尺寸: {best_detection['rotated_image'].shape}")
    
    # 透视变换测试
    print("\n" + "=" * 80)
    print("透视变换测试:")
    print("=" * 80)
    
    try:
        from utils import perspective_transform_plate, get_best_perspective_transform, batch_perspective_transform
        
        if result['count'] > 0:
            # 方法1：获取第一个检测结果的透视变换
            warped_0 = perspective_transform_plate(result, detection_index=0)
            if warped_0 is not None:
                output_path_0 = 'warped_plate_0.jpg'
                cv2.imwrite(output_path_0, warped_0)
                print(f"✓ 第一个检测结果透视变换成功")
                print(f"  输出尺寸: {warped_0.shape}")
                print(f"  已保存到: {output_path_0}")
            else:
                print("✗ 第一个检测结果透视变换失败（可能缺少关键点）")
            
            # 方法2：获取置信度最高的检测结果的透视变换
            best_warped = get_best_perspective_transform(result)
            if best_warped is not None:
                output_path_best = 'warped_plate_best.jpg'
                cv2.imwrite(output_path_best, best_warped)
                print(f"\n✓ 最佳检测结果透视变换成功")
                print(f"  输出尺寸: {best_warped.shape}")
                print(f"  已保存到: {output_path_best}")
            else:
                print("\n✗ 最佳检测结果透视变换失败（可能缺少关键点）")
            
            # 方法3：批量处理所有检测结果
            all_warped = batch_perspective_transform(result)
            success_count = sum(1 for w in all_warped if w is not None)
            print(f"\n批量透视变换:")
            print(f"  总检测数: {len(all_warped)}")
            print(f"  成功: {success_count}")
            print(f"  失败: {len(all_warped) - success_count}")
            
            # 保存所有成功的透视变换结果
            for i, warped in enumerate(all_warped):
                if warped is not None:
                    output_path = f'warped_plate_{i}.jpg'
                    cv2.imwrite(output_path, warped)
                    print(f"  已保存: {output_path} (尺寸: {warped.shape})")
        else:
            print("未检测到车牌，无法进行透视变换")
            
    except ImportError as e:
        print(f"导入 utils 模块失败: {e}")
    except Exception as e:
        print(f"透视变换测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 批量检测目录中所有图片
    # results = batch_detect(detector, 'test_images')

