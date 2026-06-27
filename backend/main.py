from __future__ import annotations

import logging
import re
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .image_processor import ImageProcessingError, ProcessOptions, process_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

app = FastAPI(title="Portrait Beautification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Output-Bytes", "X-Processing-Message"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process")
async def process_image(
    image: UploadFile = File(...),
    beauty: bool = Form(True),
    smoothSkin: bool = Form(True),
    slimFace: bool = Form(True),
    blemishRepair: bool = Form(False),
    bigEyes: bool = Form(False),
    slimNose: bool = Form(False),
    softenSmileLines: bool = Form(False),
    teethWhitening: bool = Form(False),
    backgroundBlur: bool = Form(False),
    beautyStrength: int = Form(60),
    smoothStrength: int = Form(55),
    slimStrength: int = Form(35),
    blemishStrength: int = Form(45),
    bigEyeStrength: int = Form(35),
    slimNoseStrength: int = Form(35),
    smileLineStrength: int = Form(35),
    teethStrength: int = Form(45),
    backgroundBlurStrength: int = Form(45),
    outputFormat: str = Form("jpg"),
    jpegQuality: int = Form(95),
) -> Response:
    validate_upload(image)

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    options = ProcessOptions(
        beauty=beauty,
        smooth_skin=smoothSkin,
        slim_face=slimFace,
        blemish_repair=blemishRepair,
        big_eyes=bigEyes,
        slim_nose=slimNose,
        soften_smile_lines=softenSmileLines,
        teeth_whitening=teethWhitening,
        background_blur=backgroundBlur,
        beauty_strength=clamp_int(beautyStrength, 0, 100),
        smooth_strength=clamp_int(smoothStrength, 0, 100),
        slim_strength=clamp_int(slimStrength, 0, 100),
        blemish_strength=clamp_int(blemishStrength, 0, 100),
        big_eye_strength=clamp_int(bigEyeStrength, 0, 100),
        slim_nose_strength=clamp_int(slimNoseStrength, 0, 100),
        smile_line_strength=clamp_int(smileLineStrength, 0, 100),
        teeth_strength=clamp_int(teethStrength, 0, 100),
        background_blur_strength=clamp_int(backgroundBlurStrength, 0, 100),
        output_format=outputFormat,
        jpeg_quality=clamp_int(jpegQuality, 60, 100),
    )

    try:
        result = process_upload(data, image.filename or "portrait.jpg", options)
    except ImageProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="图片处理失败，请降低图片尺寸后重试。") from exc

    return Response(
        content=result.data,
        media_type=result.media_type,
        headers={
            "Content-Disposition": build_content_disposition(result.filename),
            "X-Output-Bytes": str(result.output_bytes),
            "X-Processing-Message": quote(result.message),
        },
    )


def validate_upload(image: UploadFile) -> None:
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG 或 WebP 图片。")


def build_content_disposition(filename: str) -> str:
    ascii_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "beautified.jpg"
    return f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{quote(filename)}'


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
