<script setup lang="ts">
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileImage,
  ImageIcon,
  LoaderCircle,
  Maximize2,
  RotateCcw,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Wand2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-vue-next";
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

type FeatureKey = "beauty" | "smoothSkin" | "slimFace";

interface FeatureState {
  key: FeatureKey;
  label: string;
  description: string;
  enabled: boolean;
  strength: number;
  min: number;
  max: number;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const fileInput = ref<HTMLInputElement | null>(null);
const selectedFile = ref<File | null>(null);
const originalUrl = ref("");
const resultUrl = ref("");
const outputFileName = ref("");
const outputSize = ref("");
const processingMessage = ref("");
const errorMessage = ref("");
const isDragging = ref(false);
const isProcessing = ref(false);
const progress = ref(0);
const progressTimer = ref<number | null>(null);
const previewDialog = ref<{ src: string; title: string } | null>(null);
const previewViewport = ref<HTMLDivElement | null>(null);
const previewScale = ref(1);
const previewOffset = reactive({ x: 0, y: 0 });
const isPanningPreview = ref(false);
const panStart = reactive({ x: 0, y: 0, offsetX: 0, offsetY: 0 });

const MIN_PREVIEW_SCALE = 1;
const MAX_PREVIEW_SCALE = 6;
const PREVIEW_SCALE_STEP = 0.25;
const KEYBOARD_PAN_STEP = 42;

const features = reactive<FeatureState[]>([
  {
    key: "beauty",
    label: "美颜",
    description: "白平衡、提亮、肤色和清晰度微调",
    enabled: true,
    strength: 60,
    min: 0,
    max: 100,
  },
  {
    key: "smoothSkin",
    label: "磨皮",
    description: "肤色区域平滑，尽量保留发丝与背景细节",
    enabled: true,
    strength: 55,
    min: 0,
    max: 100,
  },
  {
    key: "slimFace",
    label: "瘦脸",
    description: "检测面部关键点后做轻量局部变形",
    enabled: true,
    strength: 35,
    min: 0,
    max: 80,
  },
]);

const canProcess = computed(() => {
  return Boolean(selectedFile.value && features.some((feature) => feature.enabled) && !isProcessing.value);
});

const progressLabel = computed(() => {
  if (!progress.value) {
    return "未开始";
  }

  if (isProcessing.value && progress.value >= 94) {
    return "后端处理中";
  }

  return `${progress.value}%`;
});

const originalInfo = computed(() => {
  if (!selectedFile.value) {
    return "";
  }

  return `${selectedFile.value.name} · ${formatBytes(selectedFile.value.size)}`;
});

const previewScaleLabel = computed(() => `${Math.round(previewScale.value * 100)}%`);

const canZoomInPreview = computed(() => previewScale.value < MAX_PREVIEW_SCALE);
const canZoomOutPreview = computed(() => previewScale.value > MIN_PREVIEW_SCALE);
const canResetPreviewZoom = computed(() => {
  return previewScale.value !== MIN_PREVIEW_SCALE || previewOffset.x !== 0 || previewOffset.y !== 0;
});

const previewImageStyle = computed(() => ({
  transform: `translate3d(${previewOffset.x}px, ${previewOffset.y}px, 0) scale(${previewScale.value})`,
}));

function openFilePicker() {
  fileInput.value?.click();
}

function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) {
    setSelectedFile(file);
  }
}

function handleDrop(event: DragEvent) {
  event.preventDefault();
  isDragging.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    setSelectedFile(file);
  }
}

function setSelectedFile(file: File) {
  errorMessage.value = "";
  processingMessage.value = "";
  outputFileName.value = "";
  outputSize.value = "";
  closePreview();
  revokeResultUrl();

  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    selectedFile.value = null;
    revokeOriginalUrl();
    errorMessage.value = "请上传 JPG、PNG 或 WebP 图片。";
    return;
  }

  selectedFile.value = file;
  revokeOriginalUrl();
  originalUrl.value = URL.createObjectURL(file);
}

async function processImage() {
  if (!selectedFile.value || !canProcess.value) {
    return;
  }

  isProcessing.value = true;
  errorMessage.value = "";
  processingMessage.value = "";
  outputFileName.value = "";
  outputSize.value = "";
  progress.value = 12;
  startProgressPulse();
  closePreview();
  revokeResultUrl();

  const formData = new FormData();
  formData.append("image", selectedFile.value);

  for (const feature of features) {
    formData.append(feature.key, String(feature.enabled));
  }

  formData.append("beautyStrength", String(findFeature("beauty").strength));
  formData.append("smoothStrength", String(findFeature("smoothSkin").strength));
  formData.append("slimStrength", String(findFeature("slimFace").strength));
  formData.append("outputFormat", "jpg");
  formData.append("jpegQuality", "95");

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 180_000);

  try {
    const response = await fetch(`${API_BASE}/api/process`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      const detail = await readErrorDetail(response);
      throw new Error(detail || `处理失败，HTTP ${response.status}`);
    }

    const blob = await response.blob();
    const filename = readFilename(response.headers.get("Content-Disposition")) ?? "beautified.jpg";
    const message = response.headers.get("X-Processing-Message");

    outputFileName.value = filename;
    outputSize.value = formatBytes(blob.size);
    resultUrl.value = URL.createObjectURL(blob);
    processingMessage.value = message ? decodeURIComponent(message) : "处理完成";
    progress.value = 100;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      errorMessage.value = "处理超时，请降低图片尺寸或关闭部分功能后重试。";
    } else {
      errorMessage.value = error instanceof Error ? error.message : "处理失败，请重试。";
    }
    progress.value = 0;
  } finally {
    window.clearTimeout(timeoutId);
    isProcessing.value = false;
    stopProgressPulse();
  }
}

function startProgressPulse() {
  stopProgressPulse();
  progressTimer.value = window.setInterval(() => {
    if (progress.value < 82) {
      progress.value += 6;
    } else if (progress.value < 94) {
      progress.value += 1;
    }
  }, 380);
}

function stopProgressPulse() {
  if (progressTimer.value !== null) {
    window.clearInterval(progressTimer.value);
    progressTimer.value = null;
  }
}

function findFeature(key: FeatureKey) {
  const feature = features.find((item) => item.key === key);
  if (!feature) {
    throw new Error(`Feature ${key} not found`);
  }
  return feature;
}

function readFilename(contentDisposition: string | null) {
  if (!contentDisposition) {
    return null;
  }

  const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    return decodeURIComponent(utf8Match[1]);
  }

  const asciiMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
  return asciiMatch?.[1] ?? null;
}

async function readErrorDetail(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const data = (await response.json()) as { detail?: string };
    return data.detail;
  }
  return response.text();
}

function openPreview(src: string, title: string) {
  resetPreviewZoom();
  previewDialog.value = { src, title };
}

function closePreview() {
  previewDialog.value = null;
  resetPreviewZoom();
}

function zoomPreviewIn() {
  setPreviewScale(previewScale.value + PREVIEW_SCALE_STEP);
}

function zoomPreviewOut() {
  setPreviewScale(previewScale.value - PREVIEW_SCALE_STEP);
}

function resetPreviewZoom() {
  previewScale.value = MIN_PREVIEW_SCALE;
  previewOffset.x = 0;
  previewOffset.y = 0;
  isPanningPreview.value = false;
}

function setPreviewScale(nextScale: number, anchor?: { clientX: number; clientY: number }) {
  const oldScale = previewScale.value;
  const newScale = clamp(nextScale, MIN_PREVIEW_SCALE, MAX_PREVIEW_SCALE);

  if (newScale === oldScale) {
    return;
  }

  if (anchor && previewViewport.value) {
    const rect = previewViewport.value.getBoundingClientRect();
    const anchorX = anchor.clientX - rect.left - rect.width / 2;
    const anchorY = anchor.clientY - rect.top - rect.height / 2;

    previewOffset.x = anchorX - ((anchorX - previewOffset.x) * newScale) / oldScale;
    previewOffset.y = anchorY - ((anchorY - previewOffset.y) * newScale) / oldScale;
  }

  previewScale.value = newScale;
  constrainPreviewOffset();
}

function constrainPreviewOffset() {
  if (!previewViewport.value || previewScale.value <= MIN_PREVIEW_SCALE) {
    previewOffset.x = 0;
    previewOffset.y = 0;
    return;
  }

  const rect = previewViewport.value.getBoundingClientRect();
  const maxX = (rect.width * (previewScale.value - MIN_PREVIEW_SCALE)) / 2;
  const maxY = (rect.height * (previewScale.value - MIN_PREVIEW_SCALE)) / 2;

  previewOffset.x = clamp(previewOffset.x, -maxX, maxX);
  previewOffset.y = clamp(previewOffset.y, -maxY, maxY);
}

function handlePreviewWheel(event: WheelEvent) {
  if (!previewDialog.value) {
    return;
  }

  event.preventDefault();
  const direction = event.deltaY > 0 ? -PREVIEW_SCALE_STEP : PREVIEW_SCALE_STEP;
  setPreviewScale(previewScale.value + direction, { clientX: event.clientX, clientY: event.clientY });
}

function togglePreviewZoom(event: MouseEvent) {
  if (previewScale.value > MIN_PREVIEW_SCALE) {
    resetPreviewZoom();
    return;
  }

  setPreviewScale(2, { clientX: event.clientX, clientY: event.clientY });
}

function startPreviewPan(event: PointerEvent) {
  if (previewScale.value <= MIN_PREVIEW_SCALE || event.button !== 0) {
    return;
  }

  event.preventDefault();
  isPanningPreview.value = true;
  panStart.x = event.clientX;
  panStart.y = event.clientY;
  panStart.offsetX = previewOffset.x;
  panStart.offsetY = previewOffset.y;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function movePreviewPan(event: PointerEvent) {
  if (!isPanningPreview.value) {
    return;
  }

  previewOffset.x = panStart.offsetX + event.clientX - panStart.x;
  previewOffset.y = panStart.offsetY + event.clientY - panStart.y;
  constrainPreviewOffset();
}

function endPreviewPan(event: PointerEvent) {
  if (!isPanningPreview.value) {
    return;
  }

  isPanningPreview.value = false;
  const target = event.currentTarget as HTMLElement;
  if (target.hasPointerCapture(event.pointerId)) {
    target.releasePointerCapture(event.pointerId);
  }
}

function panPreviewBy(deltaX: number, deltaY: number) {
  if (previewScale.value <= MIN_PREVIEW_SCALE) {
    return;
  }

  previewOffset.x += deltaX;
  previewOffset.y += deltaY;
  constrainPreviewOffset();
}

function downloadResult() {
  if (!resultUrl.value) {
    return;
  }

  const link = document.createElement("a");
  link.href = resultUrl.value;
  link.download = outputFileName.value || "beautified.jpg";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function handlePreviewKeydown(event: KeyboardEvent) {
  if (!previewDialog.value) {
    return;
  }

  if (event.key === "Escape") {
    closePreview();
    return;
  }

  if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    zoomPreviewIn();
    return;
  }

  if (event.key === "-" || event.key === "_") {
    event.preventDefault();
    zoomPreviewOut();
    return;
  }

  if (event.key === "0") {
    event.preventDefault();
    resetPreviewZoom();
    return;
  }

  if (event.key === "ArrowLeft") {
    event.preventDefault();
    panPreviewBy(KEYBOARD_PAN_STEP, 0);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    panPreviewBy(-KEYBOARD_PAN_STEP, 0);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    panPreviewBy(0, KEYBOARD_PAN_STEP);
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    panPreviewBy(0, -KEYBOARD_PAN_STEP);
  }
}

function resetAll() {
  selectedFile.value = null;
  outputFileName.value = "";
  outputSize.value = "";
  processingMessage.value = "";
  errorMessage.value = "";
  progress.value = 0;
  closePreview();
  revokeOriginalUrl();
  revokeResultUrl();
  if (fileInput.value) {
    fileInput.value.value = "";
  }
}

function revokeOriginalUrl() {
  if (originalUrl.value) {
    URL.revokeObjectURL(originalUrl.value);
    originalUrl.value = "";
  }
}

function revokeResultUrl() {
  if (resultUrl.value) {
    URL.revokeObjectURL(resultUrl.value);
    resultUrl.value = "";
  }
}

function formatBytes(bytes: number) {
  if (bytes === 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

onMounted(() => {
  window.addEventListener("keydown", handlePreviewKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handlePreviewKeydown);
  stopProgressPulse();
  closePreview();
  revokeOriginalUrl();
  revokeResultUrl();
});
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">Portrait Beautification</p>
        <h1>人像美颜</h1>
      </div>
      <button class="ghost-button" type="button" :disabled="isProcessing" @click="resetAll">
        <RefreshCw :size="18" />
        重置
      </button>
    </header>

    <section class="workspace" aria-label="图片处理工作台">
      <div class="panel upload-panel">
        <div
          class="dropzone"
          :class="{ 'is-dragging': isDragging, 'has-file': selectedFile }"
          role="button"
          tabindex="0"
          @click="openFilePicker"
          @keydown.enter.prevent="openFilePicker"
          @keydown.space.prevent="openFilePicker"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop="handleDrop"
        >
          <input
            ref="fileInput"
            class="file-input"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            @change="handleFileChange"
          />
          <Upload :size="30" />
          <div>
            <strong>{{ selectedFile ? "更换图片" : "上传图片" }}</strong>
            <span>JPG、PNG、WebP</span>
          </div>
        </div>

        <div class="info-strip" :class="{ muted: !selectedFile }">
          <FileImage :size="18" />
          <span>{{ originalInfo || "等待选择单张人像图片" }}</span>
        </div>

        <button
          v-if="originalUrl"
          class="preview-frame preview-action"
          type="button"
          aria-label="放大查看原图"
          @click="openPreview(originalUrl, '原图预览')"
        >
          <img :src="originalUrl" alt="原图预览" />
          <span class="preview-hint">
            <Maximize2 :size="16" />
            放大查看
          </span>
        </button>
        <div v-else class="preview-frame">
          <div class="empty-preview">
            <ImageIcon :size="40" />
            <span>原图预览</span>
          </div>
        </div>
      </div>

      <div class="panel controls-panel">
        <div class="panel-heading">
          <SlidersHorizontal :size="20" />
          <h2>处理参数</h2>
        </div>

        <div class="feature-list">
          <label v-for="feature in features" :key="feature.key" class="feature-row">
            <input v-model="feature.enabled" type="checkbox" />
            <span class="feature-copy">
              <span class="feature-title">{{ feature.label }}</span>
              <span>{{ feature.description }}</span>
            </span>
            <span class="strength-value">{{ feature.strength }}</span>
            <input
              v-model.number="feature.strength"
              class="range"
              type="range"
              :min="feature.min"
              :max="feature.max"
              :disabled="!feature.enabled"
            />
          </label>
        </div>

        <button class="primary-button" type="button" :disabled="!canProcess" @click="processImage">
          <LoaderCircle v-if="isProcessing" class="spin" :size="19" />
          <Wand2 v-else :size="19" />
          {{ isProcessing ? "处理中" : "开始处理" }}
        </button>

        <div class="progress-wrap" aria-live="polite">
          <div class="progress-track">
          <div class="progress-bar" :style="{ width: `${progress}%` }" />
          </div>
          <span>{{ progressLabel }}</span>
        </div>

        <p v-if="errorMessage" class="message error">
          <AlertCircle :size="18" />
          {{ errorMessage }}
        </p>
        <p v-else-if="processingMessage" class="message success">
          <CheckCircle2 :size="18" />
          {{ processingMessage }}
        </p>
      </div>

      <div class="panel result-panel">
        <div class="panel-heading">
          <Sparkles :size="20" />
          <h2>处理结果</h2>
        </div>

        <button
          v-if="resultUrl"
          class="preview-frame preview-action result-frame"
          type="button"
          aria-label="放大查看处理结果"
          @click="openPreview(resultUrl, '处理结果预览')"
        >
          <img :src="resultUrl" alt="处理结果预览" />
          <span class="preview-hint">
            <Maximize2 :size="16" />
            放大查看
          </span>
        </button>
        <div v-else class="preview-frame result-frame">
          <div class="empty-preview">
            <Download :size="40" />
            <span>处理完成后可预览</span>
          </div>
        </div>

        <div class="result-meta">
          <span>{{ outputFileName || "暂无输出文件" }}</span>
          <strong>{{ outputSize || "--" }}</strong>
        </div>

        <button v-if="resultUrl" class="download-button" type="button" @click="downloadResult">
          <Download :size="18" />
          下载图片
        </button>
      </div>
    </section>

    <Teleport to="body">
      <div v-if="previewDialog" class="lightbox" role="dialog" aria-modal="true" @click.self="closePreview">
        <div class="lightbox-surface">
          <div class="lightbox-bar">
            <strong>{{ previewDialog.title }}</strong>
            <div class="lightbox-tools">
              <button
                class="icon-button"
                type="button"
                aria-label="缩小图片"
                title="缩小图片"
                :disabled="!canZoomOutPreview"
                @click="zoomPreviewOut"
              >
                <ZoomOut :size="19" />
              </button>
              <span class="zoom-readout" aria-live="polite">{{ previewScaleLabel }}</span>
              <button
                class="icon-button"
                type="button"
                aria-label="放大图片"
                title="放大图片"
                :disabled="!canZoomInPreview"
                @click="zoomPreviewIn"
              >
                <ZoomIn :size="19" />
              </button>
              <button
                class="icon-button"
                type="button"
                aria-label="重置缩放"
                title="重置缩放"
                :disabled="!canResetPreviewZoom"
                @click="resetPreviewZoom"
              >
                <RotateCcw :size="18" />
              </button>
              <button class="icon-button" type="button" aria-label="关闭预览" title="关闭预览" @click="closePreview">
                <X :size="20" />
              </button>
            </div>
          </div>
          <div
            ref="previewViewport"
            class="lightbox-viewport"
            :class="{ 'is-pannable': previewScale > 1, 'is-panning': isPanningPreview }"
            @wheel="handlePreviewWheel"
            @dblclick="togglePreviewZoom"
            @pointerdown="startPreviewPan"
            @pointermove="movePreviewPan"
            @pointerup="endPreviewPan"
            @pointercancel="endPreviewPan"
          >
            <img
              class="lightbox-image"
              :src="previewDialog.src"
              :alt="previewDialog.title"
              :style="previewImageStyle"
              draggable="false"
            />
          </div>
        </div>
      </div>
    </Teleport>
  </main>
</template>
