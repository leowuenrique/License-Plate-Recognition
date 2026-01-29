#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 测试脚本
用于测试车牌识别 API 服务
"""

import requests
import sys
import json
from pathlib import Path


def test_health(base_url="http://localhost:8000"):
    """测试健康检查接口"""
    print("=" * 60)
    print("测试健康检查接口...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def test_recognize(base_url="http://localhost:8000", image_path=None):
    """测试识别接口"""
    print("\n" + "=" * 60)
    print("测试识别接口...")
    print("=" * 60)
    
    if image_path is None:
        # 尝试使用当前目录下的测试图片
        test_images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        if test_images:
            image_path = str(test_images[0])
            print(f"使用找到的测试图片: {image_path}")
        else:
            print("错误: 未找到测试图片，请提供图片路径")
            return False
    
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在: {image_path}")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            data = {
                "angles": "0,90,180,270",
                "use_best_detection": "true"
            }
            
            print(f"上传图片: {image_path}")
            response = requests.post(
                f"{base_url}/api/v1/recognize",
                files=files,
                data=data,
                timeout=60
            )
            
            print(f"\n状态码: {response.status_code}")
            result = response.json()
            print(f"\n识别结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            if result.get("success"):
                print(f"\n✓ 识别成功!")
                print(f"  车牌号码: {result.get('text')}")
                print(f"  检测置信度: {result.get('confidence', 0):.4f}")
                print(f"  OCR置信度: {result.get('ocr_confidence', 0):.4f}")
                print(f"  格式验证: {'通过' if result.get('is_valid_plate') else '未通过'}")
            else:
                print(f"\n✗ 识别失败")
                if "details" in result and "error" in result["details"]:
                    print(f"  错误信息: {result['details']['error']}")
            
            return result.get("success", False)
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_recognize(base_url="http://localhost:8000", image_paths=None):
    """测试批量识别接口"""
    print("\n" + "=" * 60)
    print("测试批量识别接口...")
    print("=" * 60)
    
    if image_paths is None:
        # 尝试使用当前目录下的测试图片
        test_images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        if len(test_images) >= 2:
            image_paths = [str(img) for img in test_images[:3]]  # 最多3张
            print(f"使用找到的测试图片: {image_paths}")
        else:
            print("错误: 未找到足够的测试图片（至少2张），请提供图片路径列表")
            return False
    
    try:
        files = []
        for img_path in image_paths:
            if not Path(img_path).exists():
                print(f"警告: 图片文件不存在: {img_path}")
                continue
            files.append(("files", (Path(img_path).name, open(img_path, "rb"), "image/jpeg")))
        
        if not files:
            print("错误: 没有有效的图片文件")
            return False
        
        print(f"上传 {len(files)} 张图片...")
        response = requests.post(
            f"{base_url}/api/v1/recognize_batch",
            files=files,
            timeout=120
        )
        
        # 关闭文件
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        print(f"\n状态码: {response.status_code}")
        result = response.json()
        print(f"\n批量识别结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        success_count = sum(1 for r in result.get("results", []) if r.get("success"))
        print(f"\n成功识别: {success_count}/{result.get('total', 0)}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="车牌识别 API 测试脚本")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="API 服务地址（默认: http://localhost:8000）")
    parser.add_argument("--image", type=str, default=None,
                       help="测试图片路径（单张识别）")
    parser.add_argument("--images", type=str, nargs="+", default=None,
                       help="测试图片路径列表（批量识别）")
    parser.add_argument("--skip-health", action="store_true",
                       help="跳过健康检查")
    parser.add_argument("--skip-single", action="store_true",
                       help="跳过单张识别测试")
    parser.add_argument("--skip-batch", action="store_true",
                       help="跳过批量识别测试")
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    
    print(f"API 服务地址: {base_url}")
    print()
    
    # 健康检查
    if not args.skip_health:
        if not test_health(base_url):
            print("\n健康检查失败，请确保服务已启动")
            sys.exit(1)
    
    # 单张识别测试
    success = True
    if not args.skip_single:
        success = test_recognize(base_url, args.image) and success
    
    # 批量识别测试
    if not args.skip_batch:
        success = test_batch_recognize(base_url, args.images) and success
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

