# 人像美颜项目创建计划

## Summary
- 在 `F:\PortraitBeautification` 新建独立项目
- 技术路线按你的选择使用“本地后端”：前端上传图片和选择功能，FastAPI 后端用 OpenCV/MediaPipe 处理图片，完成后浏览器自动下载。
- 采用分阶段交付：第一阶段完成可用核心版；`docx/阶段计划.md` 记录后续功能，使用 `[]` 和 `[×]` 标记进度。

## Key Changes
- 前端：Vite + Vue 3 + TypeScript，锁定兼容当前 Node 22.19.0 的最新版本（截至 2026-06-27 npm latest）：`vite@8.1.0`、`@vitejs/plugin-vue@6.0.7`、`vue@3.5.39`、`typescript@6.0.3`、`vue-tsc@3.3.5`、`lucide-vue-next@1.0.0`。
- 后端：Python 3.13 + FastAPI + OpenCV + MediaPipe，依赖包含 `fastapi==0.138.1`、`uvicorn==0.49.0`、`python-multipart`、`pillow`、`numpy`、`opencv-python`、`mediapipe==0.10.35`。
- 因本机 npm 当前 registry 指向过期的 `registry.npm.taobao.org`，README 中写明安装前执行：
  `npm config set registry https://registry.npmjs.org/`

## Implementation
- 项目结构：
  - `frontend/`：上传页面、功能勾选、强度滑杆、处理状态、原图/结果预览、自动下载逻辑。
  - `backend/`：FastAPI 服务、图片处理 pipeline、人脸检测/关键点、导出压缩控制。
  - `docx/阶段计划.md`：阶段清单。
  - `README.md`：环境、依赖、安装、启动、使用、常见问题。
- 前端页面功能：
  - 上传 JPG/PNG/WebP。
  - 勾选 `美颜`、`磨皮`、`瘦脸`。
  - 每项提供强度滑杆，默认：美颜 60、磨皮 55、瘦脸 35。
  - 点击“开始处理”后调用后端；成功后用 `Content-Disposition` 文件名自动下载。
  - 显示处理进度、错误提示、原图信息、输出文件大小。
- 后端接口：
  - `GET /health`：健康检查。
  - `POST /api/process`：multipart 上传，字段包含 `image`、`beauty`、`smoothSkin`、`slimFace`、`beautyStrength`、`smoothStrength`、`slimStrength`、`outputFormat=jpg`、`jpegQuality=95`。
  - 返回处理后的图片二进制，默认文件名 `beautified_<原文件名>.jpg`。
- 图片处理逻辑：
  - 美颜：自动白平衡、轻微提亮、肤色优化、对比和清晰度微调。
  - 磨皮：肤色区域 mask + 双边滤波/高斯融合，保留头发、衣服和背景细节。
  - 瘦脸：MediaPipe Face Mesh 获取脸部关键点，对脸颊和下颌做局部 warping；检测不到脸时跳过瘦脸并返回可读错误信息。
  - 输出控制：保留较高清晰度，JPEG 自动压缩到 30MB 以内。
- 阶段计划文档：
  - 第一阶段完成后标记 `[×]`：项目骨架、上传、三项功能、自动下载、README。
  - 后续保留 `[]`：批量处理、祛痘、亮眼、美白牙齿、背景虚化、局部预览、参数预设、历史记录、AI增强接口。

## Test Plan
- 后端：用至少 1 张人像 JPG 测试三种功能单独/组合处理，确认返回图片可打开且小于 30MB。
- 前端：测试上传、勾选、滑杆、处理按钮禁用状态、错误提示、自动下载。
- 集成：启动后端和前端，浏览器访问页面，上传 `001/` 中图片完成一次完整处理。
- 文档：按 README 从零执行安装和启动命令，确认步骤完整可复现。

## Assumptions
- 第一版只处理单张图片；批量处理放入 `docx/阶段计划.md` 后续阶段。
- 默认输出 JPG，质量 95，并自动降质量保证小于 30MB。
- 项目本地运行，不做账号、云存储或外部 AI 计费接口。
- README 使用中文，命令兼容 Windows PowerShell。
