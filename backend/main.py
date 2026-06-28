from __future__ import annotations

import logging
import os
import re
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .image_processor import ImageProcessingError, ProcessOptions, process_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "https://portraitbeautification.netlify.app",
]
DEFAULT_ALLOWED_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1):\d+$"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def parse_allowed_origins(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in value.split(",") if origin.strip()]


def parse_upload_limit(value: str | None) -> int:
    if not value:
        return MAX_UPLOAD_BYTES
    try:
        parsed = int(value)
    except ValueError:
        return MAX_UPLOAD_BYTES
    return parsed if parsed > 0 else MAX_UPLOAD_BYTES


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", DEFAULT_ALLOWED_ORIGIN_REGEX).strip() or None
UPLOAD_LIMIT_BYTES = parse_upload_limit(os.getenv("MAX_UPLOAD_BYTES"))

app = FastAPI(title="Portrait Beautification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
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
    smallMouth: bool = Form(False),
    softenSmileLines: bool = Form(False),
    teethWhitening: bool = Form(False),
    backgroundBlur: bool = Form(False),
    beautyStrength: int = Form(60),
    smoothStrength: int = Form(55),
    slimStrength: int = Form(35),
    blemishStrength: int = Form(45),
    bigEyeStrength: int = Form(35),
    slimNoseStrength: int = Form(35),
    smallMouthStrength: int = Form(25),
    smileLineStrength: int = Form(35),
    teethStrength: int = Form(45),
    backgroundBlurStrength: int = Form(45),
    outputFormat: str = Form("jpg"),
    jpegQuality: int = Form(95),
) -> Response:
    validate_upload(image)

    data = await image.read()
    if len(data) > UPLOAD_LIMIT_BYTES:
        raise_upload_too_large()

    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    options = ProcessOptions(
        beauty=beauty,
        smooth_skin=smoothSkin,
        slim_face=slimFace,
        blemish_repair=blemishRepair,
        big_eyes=bigEyes,
        slim_nose=slimNose,
        small_mouth=smallMouth,
        soften_smile_lines=softenSmileLines,
        teeth_whitening=teethWhitening,
        background_blur=backgroundBlur,
        beauty_strength=clamp_int(beautyStrength, 0, 100),
        smooth_strength=clamp_int(smoothStrength, 0, 100),
        slim_strength=clamp_int(slimStrength, 0, 100),
        blemish_strength=clamp_int(blemishStrength, 0, 100),
        big_eye_strength=clamp_int(bigEyeStrength, 0, 100),
        slim_nose_strength=clamp_int(slimNoseStrength, 0, 100),
        small_mouth_strength=clamp_int(smallMouthStrength, 0, 100),
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
    if image.size is not None and image.size > UPLOAD_LIMIT_BYTES:
        raise_upload_too_large()


def raise_upload_too_large() -> None:
    limit_mb = UPLOAD_LIMIT_BYTES / (1024 * 1024)
    raise HTTPException(status_code=413, detail=f"图片不能超过 {limit_mb:g} MB。")


def build_content_disposition(filename: str) -> str:
    ascii_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "beautified.jpg"
    return f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{quote(filename)}'


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
