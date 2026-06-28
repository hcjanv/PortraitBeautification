# 人像美颜

本项目是一个本地运行的人像美颜工具。前端使用 Vite + Vue 3 + TypeScript 提供图片上传、参数调节、前后对比、放大缩放预览和结果下载；后端使用 FastAPI + OpenCV + MediaPipe Face Landmarker 对人像图片做美颜、磨皮、瘦脸、五官微调和局部修复。

## 功能

- 上传单张 JPG、PNG、WebP 人像图片。
- 参数化处理：
  - 美颜：白平衡、提亮、肤色和清晰度微调。
  - 磨皮：肤色区域平滑，尽量保留发丝与背景细节。
  - 瘦脸：基于 MediaPipe Face Landmarker 的脸颊、颧骨、下颌局部变形。
  - 祛痘：自动淡化小痘印和局部瑕疵。
  - 大眼：局部放大眼部，控制五官自然度。
  - 瘦鼻：收窄鼻翼和鼻头，避免鼻梁漂移。
  - 小嘴：轻微收窄唇宽和唇高，让五官比例更协调。
  - 法令纹：局部提亮和平滑鼻唇沟阴影。
  - 白牙：检测口腔亮色区域后降低黄调。
  - 虚化：保护人物主体，柔化背景干扰。
- 处理结果支持前后对比拖拽线。
- 原图和结果图都支持放大预览、缩放、1:1 像素查看、拖拽平移。
- 处理结果放大预览支持前后对比拖拽。
- 处理完成后可手动下载图片，不会自动强制下载。
- 如果用户离开当前页面，处理完成后会尝试发送浏览器通知；点击通知可回到结果区域。

## 技术栈

### 前端

- Node.js 22.19.0
- npm
- Vue 3.5.39
- Vite 8.1.0
- TypeScript 6.0.3
- vue-tsc 3.3.5
- lucide-vue-next 1.0.0

### 后端

- Python 3.13
- FastAPI 0.138.1
- uvicorn 0.49.0
- OpenCV
- MediaPipe 0.10.35
- Pillow
- NumPy
- python-multipart

后端依赖本地模型文件：

```text
backend/models/face_landmarker.task
```

该模型用于 MediaPipe Face Landmarker 高精度人脸关键点检测。缺少模型时，部分依赖脸部网格的功能会降级或提示检测失败，效果会明显变差。

## 项目结构

```text
PortraitBeautification
├─ backend
│  ├─ main.py                  # FastAPI 接口
│  ├─ image_processor.py       # 图片处理逻辑
│  ├─ requirements.txt         # 后端依赖
│  └─ models
│     └─ face_landmarker.task  # MediaPipe Face Landmarker 模型
├─ frontend
│  ├─ public
│  │  └─ notification-sw.js    # 处理完成通知 Service Worker
│  ├─ src
│  │  ├─ App.vue               # 主界面和交互
│  │  ├─ style.css             # 样式
│  │  └─ images                # 测试图片
│  ├─ package.json
│  └─ package-lock.json
├─ start-backend.ps1           # 后端启动脚本
└─ README.md
```

## 安装

### 1. 后端

```powershell
cd F:\PortraitBeautification
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

如果 PowerShell 不允许激活虚拟环境，先执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. 前端

```powershell
cd F:\PortraitBeautification\frontend
npm install
```

如果本机 npm registry 指向过期的 `registry.npm.taobao.org`，安装前执行：

```powershell
npm config set registry https://registry.npmjs.org/
```

## 启动

### 1. 启动后端

推荐使用项目脚本：

```powershell
cd F:\PortraitBeautification
.\start-backend.ps1
```

默认地址：

```text
http://127.0.0.1:8000
```

也可以指定端口：

```powershell
.\start-backend.ps1 -Port 8010
```

手动启动方式：

```powershell
cd F:\PortraitBeautification
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. 启动前端

打开另一个 PowerShell：

```powershell
cd F:\PortraitBeautification\frontend
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

如果后端不是 `http://127.0.0.1:8000`，启动前端前设置：

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8010"
npm run dev
```

## 使用

1. 打开 `http://127.0.0.1:5173`。
2. 上传 JPG、PNG 或 WebP 人像图片。
3. 在“处理参数”里勾选需要的功能并调整强度。
4. 点击“开始处理”。
5. 等待后端返回图片。
6. 在“处理结果”区域拖动分割线查看修图前后差异。
7. 点击右上角放大按钮进入结果预览，可继续缩放、拖拽、1:1 查看细节，并拖动分割线对比前后。
8. 确认效果后点击“下载图片”。

默认参数：

| 功能 | 默认状态 | 默认强度 |
| --- | --- | --- |
| 美颜 | 勾选 | 60 |
| 磨皮 | 勾选 | 55 |
| 瘦脸 | 勾选 | 15 |
| 祛痘 | 勾选 | 45 |
| 大眼 | 勾选 | 15 |
| 瘦鼻 | 勾选 | 15 |
| 小嘴 | 勾选 | 15 |
| 法令纹 | 勾选 | 35 |
| 白牙 | 不勾选 | 45 |
| 虚化 | 不勾选 | 45 |

## 预览交互

- 原图预览：点击左侧原图可打开放大预览。
- 结果预览：点击结果图右上角放大按钮可打开前后对比预览。
- 缩放：使用工具栏放大/缩小按钮，或鼠标滚轮。
- 1:1：按原始像素比例查看细节，不经过 canvas 重采样。
- 拖拽：图片放大后可拖拽平移。
- 重置：恢复为默认适配窗口大小。
- 前后对比：拖动分割线查看修图前后差异。

## 浏览器通知

点击“开始处理”时，前端会在浏览器支持的情况下请求通知权限。处理耗时较长且用户离开当前页面时，处理完成后会发送“图片已处理完成”通知。

- 点击通知会回到当前应用页面并定位到处理结果区域。
- 如果通知权限被拒绝，页面不会弹系统通知，但会在用户离开页面时把标题改为“图片已处理完成 · 人像美颜”作为提醒。
- 本地 `127.0.0.1` 属于浏览器安全上下文，可使用通知和 Service Worker。

## 接口

### `GET /health`

返回：

```json
{
  "status": "ok"
}
```

### `POST /api/process`

请求类型：`multipart/form-data`

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image` | File | 必填 | JPG、PNG、WebP 图片 |
| `beauty` | boolean | `true` | 美颜 |
| `smoothSkin` | boolean | `true` | 磨皮 |
| `slimFace` | boolean | `true` | 瘦脸 |
| `blemishRepair` | boolean | `false` | 祛痘 |
| `bigEyes` | boolean | `false` | 大眼 |
| `slimNose` | boolean | `false` | 瘦鼻 |
| `smallMouth` | boolean | `false` | 小嘴 |
| `softenSmileLines` | boolean | `false` | 法令纹 |
| `teethWhitening` | boolean | `false` | 白牙 |
| `backgroundBlur` | boolean | `false` | 背景虚化 |
| `beautyStrength` | number | `60` | 0-100 |
| `smoothStrength` | number | `55` | 0-100 |
| `slimStrength` | number | `35` | 0-100 |
| `blemishStrength` | number | `45` | 0-100 |
| `bigEyeStrength` | number | `35` | 0-100 |
| `slimNoseStrength` | number | `35` | 0-100 |
| `smallMouthStrength` | number | `25` | 0-100 |
| `smileLineStrength` | number | `35` | 0-100 |
| `teethStrength` | number | `45` | 0-100 |
| `backgroundBlurStrength` | number | `45` | 0-100 |
| `outputFormat` | string | `jpg` | 支持 `jpg`、`jpeg`、`png`、`webp` |
| `jpegQuality` | number | `95` | 60-100 |

返回图片二进制，响应头包含：

- `Content-Disposition`：输出文件名。
- `X-Output-Bytes`：输出字节数。
- `X-Processing-Message`：处理提示，例如是否使用 FaceLandmarker。

## 输出规则

- 默认输出 JPG。
- 默认质量为 95。
- 输出文件名形如 `beautified_<原文件名>.jpg`。
- 后端会尽量将输出控制在 30MB 以内；超出时会降低 JPG 质量或缩放后重新编码。

## 开发和检查

前端构建检查：

```powershell
cd F:\PortraitBeautification\frontend
npm run build
```

后端语法检查：

```powershell
cd F:\PortraitBeautification
.\.venv\Scripts\python.exe -m compileall backend
```

测试接口：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

## 常见问题

### 前端无法连接后端

确认后端正在运行：

```text
http://127.0.0.1:8000
```

如果改过后端端口，需要设置 `VITE_API_BASE_URL` 后重新启动前端。

### 上传图片报错

仅支持 MIME 类型为 `image/jpeg`、`image/png`、`image/webp` 的文件。文件为空、损坏或浏览器识别不到正确 MIME 类型时，接口会返回 400。

### 瘦脸、五官调整效果差

这类功能依赖 `backend/models/face_landmarker.task` 和清晰正脸。如果模型缺失、脸部被遮挡、侧脸角度过大、照片过暗或脸部太小，FaceLandmarker 可能检测失败，后端会跳过或降级相关处理。

### 处理速度慢

处理耗时与图片尺寸、开启功能数量、机器性能有关。瘦脸、大眼、瘦鼻、小嘴、法令纹、白牙、背景虚化都需要脸部关键点检测，开启越多耗时越长。

### 浏览器通知没有出现

检查浏览器是否允许此站点通知。部分内嵌浏览器或系统通知设置可能会拦截通知；这种情况下页面标题提醒仍会生效。
