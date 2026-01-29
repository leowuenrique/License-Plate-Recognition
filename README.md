# 🚗 车牌识别 API 服务

基于关键点检测的高精度车牌识别 RESTful API 服务，采用 YOLO11-pose 进行车牌检测和关键点定位，通过透视变换校正车牌图像，使用 PaddleOCR 进行文本识别。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 核心特性

- 🎯 **关键点检测**：使用 YOLO11-pose 模型同时检测车牌位置和四个角点关键点
- 🔄 **透视变换**：基于关键点进行精确的透视变换，将倾斜车牌校正为矩形
- 🔀 **多角度检测**：支持 0°、90°、180°、270° 多角度旋转检测
- 🧠 **智能文本校正**：采用 V4 专家级融合算法进行文本校正
- ✅ **格式验证**：自动验证识别结果是否符合中国车牌格式规范
- 🛡️ **容错机制**：支持在检测或识别失败时，对整张图片进行识别（可选）
- 🖼️ **图像增强**：可选地将原图和旋转180度的图像拼接，提高识别准确率
- 🚀 **RESTful API**：基于 FastAPI 构建，提供完整的 API 文档和交互式测试界面

## 🏗️ 技术栈

- **Web 框架**: FastAPI
- **检测模型**: YOLO11-pose (Ultralytics)
- **OCR 引擎**: PaddleOCR
- **图像处理**: OpenCV, NumPy
- **配置管理**: Pydantic Settings
- **API 文档**: Swagger UI / ReDoc

## 📁 项目结构

```
├── app/                              # 应用主目录
│   ├── __init__.py                   # 应用包初始化
│   ├── main.py                       # FastAPI 应用入口，应用生命周期管理
│   ├── dependencies.py               # 依赖注入，Pipeline 单例管理
│   │
│   ├── api/                          # API 路由模块
│   │   ├── __init__.py
│   │   └── v1/                       # API 版本 1
│   │       ├── __init__.py           # API 路由注册
│   │       └── endpoints/            # API 端点
│   │           ├── __init__.py
│   │           └── recognition.py    # 车牌识别端点
│   │
│   ├── core/                         # 核心业务逻辑模块
│   │   ├── __init__.py
│   │   ├── pipeline.py               # 识别流水线主类（LicensePlatePipelineKP）
│   │   ├── detector.py               # YOLO11-pose 检测器（LicensePlatePoseDetector）
│   │   ├── rec.py                    # PaddleOCR 识别器（PaddleOCRRecognition）
│   │   └── utils.py                  # 工具函数（透视变换、文本校正、格式验证等）
│   │
│   └── config/                       # 配置管理模块
│       ├── __init__.py
│       └── settings.py               # 应用配置（基于 Pydantic Settings）
│
├── models/                           # 模型文件目录
│   ├── det/                          # 检测模型
│   │   └── yolo11n-pose-LPR-best.pt # YOLO11-pose 车牌检测模型
│   └── rec/                          # OCR 识别模型
│       └── PP-OCRv5_server_rec_infer/
│           ├── inference.json        # 模型配置
│           ├── inference.pdiparams  # 模型参数
│           └── inference.yml         # 模型元数据
│
├── tests/                            # 测试文件目录
│   └── test_api.py                   # API 测试脚本
│
├── docs/                             # 文档目录
│
├── LRP/                              # 旧版本代码（已迁移到 app/core/，可删除）
│   ├── detector.py
│   ├── pipeline.py
│   ├── rec.py
│   └── utils.py
│
├── scripts/                          # 旧脚本（已迁移到 app/，可删除）
│   └── api.py
│
├── run.py                            # 应用启动脚本
├── requirements.txt                  # Python 依赖列表
├── .env.example                      # 环境变量配置示例
├── .gitignore                        # Git 忽略规则
└── README.md                         # 项目说明文档
```

### 目录说明

- **`app/`** - 应用核心代码，采用模块化设计
  - `main.py` - FastAPI 应用入口，配置中间件和路由
  - `dependencies.py` - 依赖注入，管理 Pipeline 实例（单例模式）
  - `api/` - RESTful API 路由，支持版本化管理
  - `core/` - 核心业务逻辑，包含检测、识别、文本校正等
  - `config/` - 配置管理，支持环境变量和默认值

- **`models/`** - 模型文件存储目录
  - `det/` - 车牌检测模型（YOLO11-pose）
  - `rec/` - OCR 识别模型（PaddleOCR）

- **`tests/`** - 单元测试和集成测试

- **`docs/`** - 项目文档

- **`LRP/`** 和 **`scripts/`** - 旧版本代码，已迁移到新结构，可删除

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA (可选，用于 GPU 加速)

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd LPR4_kp
```

2. **创建虚拟环境**

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置环境变量（可选）**

```bash
cp .env.example .env
# 编辑 .env 文件，根据需要修改配置
```

5. **启动服务**

```bash
# 方式 1：使用启动脚本
python run.py

# 方式 2：使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 方式 3：开发模式（自动重载）
uvicorn app.main:app --reload
```

6. **访问 API 文档**

服务启动后，访问以下地址：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/health

## 📚 API 文档

### 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | API 信息 |
| GET | `/health` | 健康检查 |
| POST | `/api/v1/recognize` | 车牌识别 |

### 识别单张图片

**请求示例**

```bash
curl -X POST "http://localhost:8000/api/v1/recognize" \
  -F "file=@image.jpg" \
  -F "angles=0,90,180,270" \
  -F "use_best_detection=true"
```

**Python 示例**

```python
import requests

url = "http://localhost:8000/api/v1/recognize"
with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "angles": "0,90,180,270",
        "use_best_detection": "true"
    }
    response = requests.post(url, files=files, data=data)
    result = response.json()
    print(result)
```

**响应格式**

```json
{
  "success": true,
  "text": "京A12345",
  "raw_text": "京A12345",
  "confidence": 0.95,
  "ocr_confidence": 0.98,
  "is_valid_plate": true,
  "detection_count": 1,
  "warped_image_available": true,
  "details": {
    "texts": ["京A12345"],
    "scores": [0.98],
    "detection_score": 0.95,
    "corrected": false,
    "original_text": "京A12345",
    "is_valid_plate": true,
    "fallback_used": false
  }
}
```

完整的 API 文档请访问 http://localhost:8000/docs 查看交互式文档。

## ⚙️ 配置说明

### 环境变量

项目支持通过环境变量进行配置。创建 `.env` 文件（参考 `.env.example`）：

```bash
# 服务器配置
HOST=0.0.0.0
PORT=8000

# 检测器配置
DETECTOR_MODEL_PATH=models/det/yolo11n-pose-LPR-best.pt
DETECTOR_CONF_THRESHOLD=0.25
DETECTOR_USE_GPU=true

# OCR 配置
OCR_USE_GPU=true
OCR_USE_TEXTLINE_ORIENTATION=false

# 其他配置
ENABLE_LOGGING=true
LOG_LEVEL=INFO
```

### 配置优先级

1. 环境变量（`.env` 文件）
2. `app/config/settings.py` 中的默认值

所有配置项说明请参考 `app/config/settings.py`。

## 💻 开发指南

### 项目结构说明

- `app/core/` - 核心业务逻辑，可独立测试
- `app/api/` - API 路由，版本化管理
- `app/config/` - 配置管理，支持环境变量
- `app/dependencies.py` - 依赖注入，单例模式管理 Pipeline

### 添加新的 API 端点

1. 在 `app/api/v1/endpoints/` 创建新文件
2. 定义路由和端点函数
3. 在 `app/api/v1/__init__.py` 中注册路由

### 运行测试

```bash
pytest tests/
```

### 代码规范

- 使用类型提示（Type Hints）
- 遵循 PEP 8 代码风格
- 使用 Pydantic 进行数据验证


## 📊 性能指标

- **检测速度**: ~100-200ms/张（GPU）
- **识别速度**: ~200-500ms/张（GPU）
- **准确率**: >95%（标准测试集）
- **支持格式**: JPG, PNG, JPEG

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR 识别引擎
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架

## 📞 联系方式

如有问题或建议，请通过 Issue 联系。

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
