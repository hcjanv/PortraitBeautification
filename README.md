# 人像美颜

本项目是一个本地运行的人像美颜工具：前端用 Vite + Vue 3 + TypeScript 上传图片和设置参数，后端用 FastAPI + OpenCV + MediaPipe 处理图片，完成后浏览器自动下载结果图。

## 环境
- Node.js 22.19.0
- npm 10.9.3
- Python 3.13
- Windows PowerShell

前端依赖锁定为兼容 Node.js 22.19.0 的版本：`vite@8.1.0`、`@vitejs/plugin-vue@6.0.7`、`vue@3.5.39`、`typescript@6.0.3`、`vue-tsc@3.3.5`、`lucide-vue-next@1.0.0`。

如果本机 npm registry 指向过期的 `registry.npm.taobao.org`，安装前执行：

```powershell
npm config set registry https://registry.npmjs.org/
```

本项目的 `frontend/.npmrc` 也已经指向官方 registry。

## 安装

### 后端
```powershell
cd F:\PortraitBeautification
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

### 前端
```powershell
cd F:\PortraitBeautification\frontend
npm install
```

## 启动

打开第一个 PowerShell：

```powershell
cd F:\PortraitBeautification
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

打开第二个 PowerShell：

```powershell
cd F:\PortraitBeautification\frontend
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

## 使用
1. 上传 JPG、PNG 或 WebP 人像图片。
2. 勾选 `美颜`、`磨皮`、`瘦脸`，调整强度滑杆。
3. 点击 `开始处理`。
4. 后端返回图片后，浏览器会自动下载 `beautified_<原文件名>.jpg`，页面也会显示结果预览和输出大小。

## 接口

### `GET /health`
返回：

```json
{
  "status": "ok"
}
```

### `POST /api/process`
`multipart/form-data` 字段：

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `image` | File | 必填 |
| `beauty` | boolean | `true` |
| `smoothSkin` | boolean | `true` |
| `slimFace` | boolean | `true` |
| `beautyStrength` | number | `60` |
| `smoothStrength` | number | `55` |
| `slimStrength` | number | `35` |
| `outputFormat` | string | `jpg` |
| `jpegQuality` | number | `95` |

返回图片二进制，响应头包含：
- `Content-Disposition`：下载文件名
- `X-Output-Bytes`：输出大小
- `X-Processing-Message`：处理提示

## 常见问题

### PowerShell 不允许激活虚拟环境
执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 瘦脸没有生效
后端会优先尝试 MediaPipe Face Mesh；如果当前 MediaPipe 包没有暴露旧版 FaceMesh 接口，会自动改用 OpenCV 正脸检测兜底。图片没有清晰正脸、侧脸角度过大或检测失败时，后端会跳过瘦脸并在响应头中返回提示。

### 输出文件太大
默认输出 JPG，质量从 95 开始，如果超过 30MB 会自动降低质量重新编码。

### 前端无法连接后端
确认后端运行在：

```text
http://127.0.0.1:8000
```

如果修改了后端地址，可在启动前端前设置：

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```
