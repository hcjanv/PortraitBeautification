from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
import threading
import time
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_OUTPUT_BYTES = 30 * 1024 * 1024
MAX_FACE_DETECT_SIDE = 1280
MAX_SMOOTH_SIDE = 1400
FACE_LANDMARKER_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
OutputFormat = Literal["jpg", "jpeg", "png", "webp"]
LOGGER = logging.getLogger("portrait_beautification.processor")

FACE_OVAL_INDICES = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

LEFT_SLIM_CONTROLS = (
    (234, 0.42, 1.14),
    (93, 0.62, 1.12),
    (132, 0.86, 1.08),
    (58, 0.96, 1.02),
    (172, 0.94, 0.98),
    (136, 0.78, 0.92),
    (150, 0.58, 0.86),
    (149, 0.44, 0.80),
    (176, 0.30, 0.74),
    (148, 0.18, 0.68),
)
RIGHT_SLIM_CONTROLS = (
    (454, 0.42, 1.14),
    (323, 0.62, 1.12),
    (361, 0.86, 1.08),
    (288, 0.96, 1.02),
    (397, 0.94, 0.98),
    (365, 0.78, 0.92),
    (379, 0.58, 0.86),
    (378, 0.44, 0.80),
    (400, 0.30, 0.74),
    (377, 0.18, 0.68),
)

_FACE_LANDMARKER_LOCK = threading.Lock()
_FACE_LANDMARKER: object | None = None
_FACE_LANDMARKER_INIT_ERROR: str | None = None


@dataclass(frozen=True)
class ProcessOptions:
    beauty: bool = True
    smooth_skin: bool = True
    slim_face: bool = True
    beauty_strength: int = 60
    smooth_strength: int = 55
    slim_strength: int = 35
    output_format: OutputFormat = "jpg"
    jpeg_quality: int = 95


@dataclass(frozen=True)
class ProcessResult:
    data: bytes
    filename: str
    media_type: str
    output_bytes: int
    message: str


@dataclass(frozen=True)
class FaceMesh:
    points: np.ndarray
    source: str


@dataclass(frozen=True)
class WarpControl:
    point: tuple[float, float]
    delta: tuple[float, float]
    radius: float


class ImageProcessingError(ValueError):
    pass


def process_upload(file_bytes: bytes, original_filename: str, options: ProcessOptions) -> ProcessResult:
    started = time.perf_counter()
    image = decode_image(file_bytes)
    messages: list[str] = []
    LOGGER.info("decoded image filename=%s size=%sx%s", original_filename, image.shape[1], image.shape[0])

    if options.beauty:
        step_started = time.perf_counter()
        image = apply_beauty(image, options.beauty_strength)
        LOGGER.info("beauty step completed in %.2fs", time.perf_counter() - step_started)

    if options.smooth_skin:
        step_started = time.perf_counter()
        image = apply_smooth_skin(image, options.smooth_strength)
        LOGGER.info("smooth skin step completed in %.2fs", time.perf_counter() - step_started)

    if options.slim_face:
        step_started = time.perf_counter()
        image, slim_message = apply_slim_face(image, options.slim_strength)
        messages.append(slim_message)
        LOGGER.info("slim face step completed in %.2fs message=%s", time.perf_counter() - step_started, slim_message)

    step_started = time.perf_counter()
    output_format = normalize_output_format(options.output_format)
    data, media_type = encode_image(image, output_format, options.jpeg_quality)
    LOGGER.info("encode step completed in %.2fs output_bytes=%s", time.perf_counter() - step_started, len(data))
    filename = build_output_filename(original_filename, output_format)
    message = "；".join([item for item in messages if item]) or "处理完成"
    LOGGER.info("process completed in %.2fs", time.perf_counter() - started)

    return ProcessResult(
        data=data,
        filename=filename,
        media_type=media_type,
        output_bytes=len(data),
        message=message,
    )


def decode_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ImageProcessingError("无法读取图片，请确认文件是有效的 JPG、PNG 或 WebP。") from exc

    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def apply_beauty(image: np.ndarray, strength: int) -> np.ndarray:
    amount = clamp01(strength / 100)
    if amount <= 0:
        return image

    balanced = gray_world_white_balance(image)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe_limit = 1.05 + amount * 0.9
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    brightness = 1.0 + amount * 0.045
    contrast = 1.0 + amount * 0.04
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=255 * (brightness - 1))

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    sharpened = cv2.addWeighted(enhanced, 1.0 + amount * 0.16, blur, -amount * 0.16, 0)

    return blend_images(image, sharpened, 0.70 * amount)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_means.mean())
    scale = gray_mean / np.maximum(channel_means, 1.0)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_smooth_skin(image: np.ndarray, strength: int) -> np.ndarray:
    amount = clamp01(strength / 100)
    if amount <= 0:
        return image

    mask = build_skin_mask(image)
    if mask.max() == 0:
        return image

    work_image, scale = resize_for_work(image, MAX_SMOOTH_SIDE)
    sigma_color = 16 + int(amount * 48)
    sigma_space = 8 + int(amount * 16)
    diameter = make_odd(7 + int(amount * 6))
    filtered = cv2.bilateralFilter(work_image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    if scale != 1.0:
        filtered = cv2.resize(filtered, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    soft = cv2.GaussianBlur(image, (0, 0), 2.0 + amount * 2.6)
    smooth = cv2.addWeighted(filtered, 0.76, soft, 0.24, 0)

    base_blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    detail = image.astype(np.float32) - base_blur.astype(np.float32)
    detail_amount = 0.14 + amount * 0.10
    smooth = np.clip(smooth.astype(np.float32) + detail * detail_amount, 0, 255).astype(np.uint8)

    mask_float = (mask.astype(np.float32) / 255.0)[:, :, None] * (0.26 + amount * 0.38)
    result = image.astype(np.float32) * (1.0 - mask_float) + smooth.astype(np.float32) * mask_float
    return np.clip(result, 0, 255).astype(np.uint8)


def build_skin_mask(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 180, 135]))
    hsv_mask = cv2.inRange(hsv, np.array([0, 18, 48]), np.array([25, 210, 255]))
    skin_mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.GaussianBlur(skin_mask, (0, 0), 5)


def apply_slim_face(image: np.ndarray, strength: int) -> tuple[np.ndarray, str]:
    amount = clamp01(strength / 100)
    if amount <= 0:
        return image, "瘦脸强度为 0，已跳过"

    face_mesh, mesh_message = detect_face_mesh(image)
    if face_mesh is not None:
        result = apply_mesh_slim_face(image, face_mesh.points, amount)
        return result, "处理完成（FaceLandmarker 高精度瘦脸）"

    face_points = detect_face_points_with_opencv(image)
    if face_points is not None:
        result = apply_opencv_slim_face(image, face_points, amount)
        return result, f"处理完成（OpenCV 低精度瘦脸：{mesh_message}）"

    return image, f"{mesh_message}；未检测到人脸，已跳过瘦脸"


def detect_face_mesh(image: np.ndarray) -> tuple[FaceMesh | None, str]:
    detector, detector_message = get_face_landmarker()
    if detector is None:
        return None, detector_message

    try:
        import mediapipe as mp
    except Exception as exc:  # noqa: BLE001
        message = f"MediaPipe 导入失败：{exc}"
        LOGGER.warning(message)
        return None, message

    detect_image, detect_scale = resize_for_work(image, MAX_FACE_DETECT_SIDE)
    rgb = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _FACE_LANDMARKER_LOCK:
            results = detector.detect(mp_image)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        message = f"FaceLandmarker 检测失败：{exc}"
        LOGGER.warning(message)
        return None, message

    if not results.face_landmarks:
        return None, "FaceLandmarker 未检测到人脸"

    height, width = detect_image.shape[:2]
    landmarks = results.face_landmarks[0]
    points = np.array(
        [
            (
                float(np.clip(point.x * width / detect_scale, 0, image.shape[1] - 1)),
                float(np.clip(point.y * height / detect_scale, 0, image.shape[0] - 1)),
            )
            for point in landmarks
        ],
        dtype=np.float32,
    )
    return FaceMesh(points=points, source="FaceLandmarker"), "FaceLandmarker 可用"


def get_face_landmarker() -> tuple[object | None, str]:
    global _FACE_LANDMARKER, _FACE_LANDMARKER_INIT_ERROR

    if _FACE_LANDMARKER is not None:
        return _FACE_LANDMARKER, "FaceLandmarker 可用"
    if _FACE_LANDMARKER_INIT_ERROR is not None:
        return None, _FACE_LANDMARKER_INIT_ERROR

    with _FACE_LANDMARKER_LOCK:
        if _FACE_LANDMARKER is not None:
            return _FACE_LANDMARKER, "FaceLandmarker 可用"
        if _FACE_LANDMARKER_INIT_ERROR is not None:
            return None, _FACE_LANDMARKER_INIT_ERROR

        if not FACE_LANDMARKER_MODEL.exists():
            _FACE_LANDMARKER_INIT_ERROR = f"未找到 FaceLandmarker 模型：{FACE_LANDMARKER_MODEL}"
            LOGGER.warning(_FACE_LANDMARKER_INIT_ERROR)
            return None, _FACE_LANDMARKER_INIT_ERROR

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
            LOGGER.info("FaceLandmarker initialized model=%s", FACE_LANDMARKER_MODEL)
            return _FACE_LANDMARKER, "FaceLandmarker 可用"
        except Exception as exc:  # noqa: BLE001
            _FACE_LANDMARKER_INIT_ERROR = f"FaceLandmarker 初始化失败：{exc}"
            LOGGER.exception(_FACE_LANDMARKER_INIT_ERROR)
            return None, _FACE_LANDMARKER_INIT_ERROR


def apply_mesh_slim_face(image: np.ndarray, points: np.ndarray, amount: float) -> np.ndarray:
    height, width = image.shape[:2]
    face_oval = points[list(FACE_OVAL_INDICES)]
    min_x, min_y = np.min(face_oval, axis=0)
    max_x, max_y = np.max(face_oval, axis=0)
    face_width = float(max(max_x - min_x, 1.0))
    face_height = float(max(max_y - min_y, 1.0))

    if face_width < 20 or face_height < 20:
        return image

    center_x = float(np.mean([points[1][0], points[152][0], np.mean(face_oval[:, 0])]))
    amount_curve = float(np.power(amount, 0.78))
    base_shift = face_width * (0.010 + amount_curve * 0.078)
    base_shift = float(min(base_shift, face_width * 0.096, face_height * 0.070))
    base_radius = face_width * (0.19 + amount_curve * 0.12)

    controls: list[WarpControl] = []
    for index, move_scale, radius_scale in LEFT_SLIM_CONTROLS + RIGHT_SLIM_CONTROLS:
        point = points[index]
        y_norm = clamp01((float(point[1]) - min_y) / face_height)
        contour_weight = smoothstep(0.30, 0.43, y_norm)
        cheekbone_boost = 1.0 + 0.22 * (1.0 - smoothstep(0.48, 0.68, y_norm))
        jaw_soften = 1.0 - smoothstep(0.86, 1.00, y_norm) * 0.42
        direction = 1.0 if point[0] < center_x else -1.0
        dx = direction * base_shift * move_scale * contour_weight * cheekbone_boost * jaw_soften
        if abs(dx) < 0.25:
            continue
        controls.append(
            WarpControl(
                point=(float(point[0]), float(point[1])),
                delta=(float(dx), 0.0),
                radius=float(base_radius * radius_scale),
            )
        )

    chin = points[152]
    chin_lift = -base_shift * 0.08 * amount_curve
    if abs(chin_lift) >= 0.25:
        controls.append(
            WarpControl(
                point=(float(chin[0]), float(chin[1])),
                delta=(0.0, float(chin_lift)),
                radius=float(base_radius * 0.72),
            )
        )

    if not controls:
        return image

    margin_x = int(face_width * (0.38 + amount_curve * 0.10))
    margin_top = int(face_height * 0.16)
    margin_bottom = int(face_height * 0.22)
    left = max(int(min_x) - margin_x, 0)
    right = min(int(max_x) + margin_x, width - 1)
    top = max(int(min_y) - margin_top, 0)
    bottom = min(int(max_y) + margin_bottom, height - 1)

    if left >= right or top >= bottom:
        return image

    roi = image[top : bottom + 1, left : right + 1]
    grid_x, grid_y = np.meshgrid(
        np.arange(left, right + 1, dtype=np.float32),
        np.arange(top, bottom + 1, dtype=np.float32),
    )

    disp_x = np.zeros_like(grid_x, dtype=np.float32)
    disp_y = np.zeros_like(grid_y, dtype=np.float32)
    total_weight = np.zeros_like(grid_x, dtype=np.float32)

    for control in controls:
        cx, cy = control.point
        dx, dy = control.delta
        radius_square = max(control.radius * control.radius, 1.0)
        distance_square = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        mask = distance_square < radius_square
        if not np.any(mask):
            continue

        weight = np.zeros_like(grid_x, dtype=np.float32)
        t = 1.0 - distance_square[mask] / radius_square
        weight[mask] = t * t * (3.0 - 2.0 * t)
        disp_x += weight * dx
        disp_y += weight * dy
        total_weight += weight

    active = total_weight > 1.0
    disp_x[active] /= total_weight[active]
    disp_y[active] /= total_weight[active]

    protection = build_feature_protection_mask(points, (left, top), roi.shape[:2], face_width, face_height)
    protect_factor = 1.0 - protection * 0.88
    disp_x *= protect_factor
    disp_y *= protect_factor

    influence = build_face_influence_mask(points, (left, top), roi.shape[:2], face_width)
    disp_x *= influence
    disp_y *= influence

    max_displacement = face_width * (0.026 + amount_curve * 0.082)
    magnitude = np.sqrt(disp_x * disp_x + disp_y * disp_y)
    capped = magnitude > max_displacement
    if np.any(capped):
        cap_scale = max_displacement / np.maximum(magnitude[capped], 1e-6)
        disp_x[capped] *= cap_scale
        disp_y[capped] *= cap_scale

    map_x = (grid_x - disp_x - left).astype(np.float32)
    map_y = (grid_y - disp_y - top).astype(np.float32)
    warped_roi = cv2.remap(
        roi,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    result = image.copy()
    result[top : bottom + 1, left : right + 1] = warped_roi
    return result


def build_feature_protection_mask(
    points: np.ndarray,
    offset: tuple[int, int],
    shape: tuple[int, int],
    face_width: float,
    face_height: float,
) -> np.ndarray:
    left, top = offset
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)

    def draw_ellipse(indices: tuple[int, ...], axis_x: float, axis_y: float) -> None:
        center = np.mean(points[list(indices)], axis=0)
        cx = int(round(float(center[0]) - left))
        cy = int(round(float(center[1]) - top))
        axes = (max(2, int(round(axis_x))), max(2, int(round(axis_y))))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)

    draw_ellipse((33, 133, 159, 145), face_width * 0.12, face_height * 0.055)
    draw_ellipse((362, 263, 386, 374), face_width * 0.12, face_height * 0.055)
    draw_ellipse((1, 2, 98, 327, 168), face_width * 0.11, face_height * 0.15)
    draw_ellipse((61, 291, 13, 14, 17), face_width * 0.18, face_height * 0.075)

    sigma = max(3.0, face_width * 0.022)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return np.clip(mask, 0.0, 1.0)


def build_face_influence_mask(
    points: np.ndarray,
    offset: tuple[int, int],
    shape: tuple[int, int],
    face_width: float,
) -> np.ndarray:
    left, top = offset
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    oval = points[list(FACE_OVAL_INDICES)].copy()
    oval[:, 0] -= left
    oval[:, 1] -= top
    hull = cv2.convexHull(np.round(oval).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    dilate_size = make_odd(max(9, int(round(face_width * 0.08))))
    blur_sigma = max(4.0, face_width * 0.035)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask_float = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), blur_sigma)
    return np.clip(mask_float, 0.0, 1.0)


def apply_opencv_slim_face(
    image: np.ndarray,
    face_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    amount: float,
) -> np.ndarray:
    left_cheek, right_cheek, left_jaw, right_jaw = face_points
    center_x = int((left_cheek[0] + right_cheek[0]) / 2)
    face_width = max(abs(right_cheek[0] - left_cheek[0]), 1)

    safe_amount = amount * 0.42
    move = face_width * (0.006 + safe_amount * 0.04)
    radius = face_width * (0.14 + safe_amount * 0.09)
    warped = image.copy()
    warped = local_translate_warp(warped, left_cheek, (int(left_cheek[0] + move), left_cheek[1]), radius)
    warped = local_translate_warp(warped, right_cheek, (int(right_cheek[0] - move), right_cheek[1]), radius)

    lower_move = min(move * 0.28, abs(center_x - left_jaw[0]) * 0.10)
    lower_radius = radius * 0.55
    warped = local_translate_warp(warped, left_jaw, (int(left_jaw[0] + lower_move), left_jaw[1]), lower_radius)
    warped = local_translate_warp(warped, right_jaw, (int(right_jaw[0] - lower_move), right_jaw[1]), lower_radius)
    return warped


def detect_face_points_with_opencv(
    image: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    detect_image, detect_scale = resize_for_work(image, MAX_FACE_DETECT_SIDE)
    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None

    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda rect: rect[2] * rect[3])

    def scale_point(px: float, py: float) -> tuple[int, int]:
        return int(px / detect_scale), int(py / detect_scale)

    left_cheek = scale_point(x + width * 0.17, y + height * 0.56)
    right_cheek = scale_point(x + width * 0.83, y + height * 0.56)
    left_jaw = scale_point(x + width * 0.30, y + height * 0.84)
    right_jaw = scale_point(x + width * 0.70, y + height * 0.84)
    return left_cheek, right_cheek, left_jaw, right_jaw


def local_translate_warp(
    image: np.ndarray,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
    radius: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    radius = max(radius, 1.0)
    radius_square = radius * radius
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    left = max(int(start_point[0] - radius), 0)
    right = min(int(start_point[0] + radius), width - 1)
    top = max(int(start_point[1] - radius), 0)
    bottom = min(int(start_point[1] + radius), height - 1)

    if left >= right or top >= bottom:
        return image

    roi = image[top : bottom + 1, left : right + 1]
    grid_x, grid_y = np.meshgrid(
        np.arange(left, right + 1, dtype=np.float32),
        np.arange(top, bottom + 1, dtype=np.float32),
    )
    distance_square = (grid_x - start_point[0]) ** 2 + (grid_y - start_point[1]) ** 2
    mask = distance_square < radius_square

    map_x = grid_x.copy()
    map_y = grid_y.copy()

    ratio = np.zeros_like(distance_square, dtype=np.float32)
    influence = radius_square - distance_square[mask]
    ratio[mask] = (influence / (influence + dx * dx + dy * dy)) ** 2
    map_x[mask] = grid_x[mask] - dx * ratio[mask]
    map_y[mask] = grid_y[mask] - dy * ratio[mask]

    map_x -= left
    map_y -= top
    warped_roi = cv2.remap(
        roi,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    result = image.copy()
    result[top : bottom + 1, left : right + 1] = warped_roi
    return result


def encode_image(image: np.ndarray, output_format: OutputFormat, jpeg_quality: int) -> tuple[bytes, str]:
    output_format = normalize_output_format(output_format)
    quality = int(np.clip(jpeg_quality, 60, 100))

    if output_format in {"jpg", "jpeg"}:
        return encode_jpeg_under_limit(image, quality), "image/jpeg"

    if output_format == "png":
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        media_type = "image/png"
    else:
        ok, encoded = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, quality])
        media_type = "image/webp"

    if not ok:
        raise ImageProcessingError("图片导出失败。")

    data = encoded.tobytes()
    if len(data) > MAX_OUTPUT_BYTES and output_format != "png":
        return encode_jpeg_under_limit(image, quality), "image/jpeg"
    return data, media_type


def encode_jpeg_under_limit(image: np.ndarray, start_quality: int) -> bytes:
    last_data = b""
    current = image

    for _ in range(6):
        quality = int(np.clip(start_quality, 60, 100))

        while quality >= 60:
            ok, encoded = cv2.imencode(
                ".jpg",
                current,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                    1,
                ],
            )
            if not ok:
                raise ImageProcessingError("JPEG 导出失败。")

            last_data = encoded.tobytes()
            if len(last_data) <= MAX_OUTPUT_BYTES:
                return last_data
            quality -= 5

        height, width = current.shape[:2]
        if max(height, width) <= 1600:
            break

        scale = (MAX_OUTPUT_BYTES / max(len(last_data), 1)) ** 0.5 * 0.92
        scale = float(np.clip(scale, 0.55, 0.9))
        current = cv2.resize(
            current,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    return last_data


def normalize_output_format(output_format: str) -> OutputFormat:
    normalized = output_format.lower().lstrip(".")
    if normalized not in {"jpg", "jpeg", "png", "webp"}:
        return "jpg"
    return normalized  # type: ignore[return-value]


def build_output_filename(original_filename: str, output_format: OutputFormat) -> str:
    stem = Path(original_filename or "portrait").stem.strip() or "portrait"
    extension = "jpg" if output_format == "jpeg" else output_format
    return f"beautified_{stem}.{extension}"


def blend_images(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    alpha = clamp01(alpha)
    result = base.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def resize_for_work(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0

    scale = max_side / longest
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
