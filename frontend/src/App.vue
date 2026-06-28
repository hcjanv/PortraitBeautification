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
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue";

type FeatureKey =
  | "beauty"
  | "smoothSkin"
  | "slimFace"
  | "blemishRepair"
  | "bigEyes"
  | "slimNose"
  | "smallMouth"
  | "softenSmileLines"
  | "teethWhitening"
  | "backgroundBlur";

interface FeatureState {
  key: FeatureKey;
  label: string;
  description: string;
  enabled: boolean;
  strength: number;
  min: number;
  max: number;
}

type FeatureDefaults = Pick<FeatureState, "enabled" | "strength">;
type PreviewDialog =
  | { mode: "single"; src: string; title: string }
  | { mode: "compare"; beforeSrc: string; afterSrc: string; title: string };

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const strengthFieldByFeature: Record<FeatureKey, string> = {
  beauty: "beautyStrength",
  smoothSkin: "smoothStrength",
  slimFace: "slimStrength",
  blemishRepair: "blemishStrength",
  bigEyes: "bigEyeStrength",
  slimNose: "slimNoseStrength",
  smallMouth: "smallMouthStrength",
  softenSmileLines: "smileLineStrength",
  teethWhitening: "teethStrength",
  backgroundBlur: "backgroundBlurStrength",
};

const featureDefaultsByKey: Record<FeatureKey, FeatureDefaults> = {
  beauty: { enabled: true, strength: 60 },
  smoothSkin: { enabled: true, strength: 55 },
  slimFace: { enabled: true, strength: 15 },
  blemishRepair: { enabled: true, strength: 45 },
  bigEyes: { enabled: true, strength: 15 },
  slimNose: { enabled: true, strength: 15 },
  smallMouth: { enabled: true, strength: 15 },
  softenSmileLines: { enabled: true, strength: 35 },
  teethWhitening: { enabled: false, strength: 45 },
  backgroundBlur: { enabled: false, strength: 45 },
};

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
const previewDialog = ref<PreviewDialog | null>(null);
const previewViewport = ref<HTMLDivElement | null>(null);
const previewScale = ref(1);
const previewFitScale = ref(1);
const previewOffset = reactive({ x: 0, y: 0 });
const previewNaturalSize = reactive({ width: 0, height: 0 });
const isPanningPreview = ref(false);
const isDraggingCompare = ref(false);
const panStart = reactive({ x: 0, y: 0, offsetX: 0, offsetY: 0 });
const compareValue = ref(50);
const resultPanel = ref<HTMLElement | null>(null);

const MIN_PREVIEW_SCALE = 0.5;
const MIN_PREVIEW_FIT_SCALE = 0.02;
const MAX_PREVIEW_SCALE = 10;
const DEFAULT_PREVIEW_SCALE = 1;
const PREVIEW_SCALE_STEP = 0.25;
const KEYBOARD_PAN_STEP = 42;
const ZOOM_EPSILON = 0.005;
const COMPLETION_NOTIFICATION_TAG = "portrait-beautification-complete";
const SHOW_RESULT_MESSAGE = "portrait-beautification:show-result";
const NOTIFICATION_WORKER_PATH = "/notification-sw.js";
const defaultDocumentTitle = document.title;

let notificationRegistration: ServiceWorkerRegistration | null = null;
let activeCompletionNotification: Notification | null = null;

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
    description: "脸颊、颧骨和下颌的高精度局部变形",
    enabled: true,
    strength: 15,
    min: 0,
    max: 80,
  },
  {
    key: "blemishRepair",
    label: "祛痘",
    description: "自动淡化小痘印和局部瑕疵",
    enabled: true,
    strength: 45,
    min: 0,
    max: 100,
  },
  {
    key: "bigEyes",
    label: "大眼",
    description: "局部放大眼部，控制五官自然度",
    enabled: true,
    strength: 15,
    min: 0,
    max: 80,
  },
  {
    key: "slimNose",
    label: "瘦鼻",
    description: "收窄鼻翼和鼻头，避免鼻梁漂移",
    enabled: true,
    strength: 15,
    min: 0,
    max: 80,
  },
  {
    key: "smallMouth",
    label: "小嘴",
    description: "轻微收窄唇宽和唇高，让五官比例更协调",
    enabled: true,
    strength: 15,
    min: 0,
    max: 80,
  },
  {
    key: "softenSmileLines",
    label: "法令纹",
    description: "局部提亮和平滑鼻唇沟阴影",
    enabled: true,
    strength: 35,
    min: 0,
    max: 100,
  },
  {
    key: "teethWhitening",
    label: "白牙",
    description: "检测口腔亮色区域后降低黄调",
    enabled: false,
    strength: 45,
    min: 0,
    max: 100,
  },
  {
    key: "backgroundBlur",
    label: "虚化",
    description: "保护人物主体，柔化背景干扰",
    enabled: false,
    strength: 45,
    min: 0,
    max: 100,
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
const previewSingleDialog = computed(() => {
  return previewDialog.value?.mode === "single" ? previewDialog.value : null;
});
const previewCompareDialog = computed(() => {
  return previewDialog.value?.mode === "compare" ? previewDialog.value : null;
});
const previewActualSizeScale = computed(() => {
  if (!previewNaturalSize.width || previewFitScale.value <= 0) {
    return DEFAULT_PREVIEW_SCALE;
  }

  return 1 / previewFitScale.value;
});
const previewMaxScale = computed(() => Math.max(MAX_PREVIEW_SCALE, previewActualSizeScale.value));
const previewRenderedScale = computed(() => previewFitScale.value * previewScale.value);

const canZoomInPreview = computed(() => previewScale.value < previewMaxScale.value - ZOOM_EPSILON);
const canZoomOutPreview = computed(() => previewScale.value > MIN_PREVIEW_SCALE + ZOOM_EPSILON);
const isPreviewPannable = computed(() => previewScale.value > DEFAULT_PREVIEW_SCALE + ZOOM_EPSILON);
const canUseActualSizePreview = computed(() => {
  return Boolean(previewNaturalSize.width && Math.abs(previewScale.value - previewActualSizeScale.value) > ZOOM_EPSILON);
});
const canResetPreviewZoom = computed(() => {
  return (
    Math.abs(previewScale.value - DEFAULT_PREVIEW_SCALE) > ZOOM_EPSILON ||
    previewOffset.x !== 0 ||
    previewOffset.y !== 0
  );
});

const previewImageStyle = computed(() => ({
  width: previewNaturalSize.width ? `${previewNaturalSize.width}px` : "auto",
  height: previewNaturalSize.height ? `${previewNaturalSize.height}px` : "auto",
  left: previewNaturalSize.width ? `${-previewNaturalSize.width / 2}px` : "0",
  top: previewNaturalSize.height ? `${-previewNaturalSize.height / 2}px` : "0",
  transform: `scale(${previewRenderedScale.value})`,
}));

const previewAnchorStyle = computed(() => ({
  transform: `translate3d(${previewOffset.x}px, ${previewOffset.y}px, 0)`,
}));

const compareAfterStyle = computed(() => ({
  clipPath: `inset(0 ${100 - compareValue.value}% 0 0)`,
}));

const lightboxCompareAfterStyle = computed(() => ({
  clipPath: `inset(0 ${100 - compareValue.value}% 0 0)`,
}));

const compareDividerStyle = computed(() => ({
  left: `${compareValue.value}%`,
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
  compareValue.value = 50;
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

  const notificationReady = prepareCompletionNotification();

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
    formData.append(strengthFieldByFeature[feature.key], String(feature.strength));
  }

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
    compareValue.value = 50;
    progress.value = 100;
    markCompletionInTitleIfAway();
    void notificationReady.then(() => showCompletionNotification(filename)).catch(markCompletionInTitleIfAway);
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
  openPreviewDialog({ mode: "single", src, title });
}

function openResultPreview() {
  if (!originalUrl.value || !resultUrl.value) {
    return;
  }

  openPreviewDialog({
    mode: "compare",
    beforeSrc: originalUrl.value,
    afterSrc: resultUrl.value,
    title: "处理结果预览",
  });
}

function openPreviewDialog(dialog: PreviewDialog) {
  previewNaturalSize.width = 0;
  previewNaturalSize.height = 0;
  previewFitScale.value = 1;
  resetPreviewZoom();
  previewDialog.value = dialog;
  void nextTick(() => refreshPreviewFitScale(true));
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
  previewScale.value = DEFAULT_PREVIEW_SCALE;
  previewOffset.x = 0;
  previewOffset.y = 0;
  isPanningPreview.value = false;
  isDraggingCompare.value = false;
}

function zoomPreviewActualSize() {
  setPreviewScale(previewActualSizeScale.value);
}

function handlePreviewImageLoad(event: Event) {
  const image = event.target as HTMLImageElement;
  previewNaturalSize.width = image.naturalWidth;
  previewNaturalSize.height = image.naturalHeight;
  void nextTick(() => refreshPreviewFitScale(true));
}

function calculatePreviewFitScale() {
  if (!previewViewport.value || !previewNaturalSize.width || !previewNaturalSize.height) {
    return 1;
  }

  const rect = previewViewport.value.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return 1;
  }

  const scale = Math.min(rect.width / previewNaturalSize.width, rect.height / previewNaturalSize.height, 1);
  return clamp(scale, MIN_PREVIEW_FIT_SCALE, 1);
}

function refreshPreviewFitScale(forceDefault = false) {
  previewFitScale.value = calculatePreviewFitScale();

  if (forceDefault) {
    resetPreviewZoom();
    return;
  }

  constrainPreviewOffset();
}

function setPreviewScale(nextScale: number, anchor?: { clientX: number; clientY: number }) {
  const oldScale = previewScale.value;
  const newScale = clamp(nextScale, MIN_PREVIEW_SCALE, previewMaxScale.value);

  if (Math.abs(newScale - oldScale) <= ZOOM_EPSILON) {
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
  if (
    !previewViewport.value ||
    !previewNaturalSize.width ||
    !previewNaturalSize.height
  ) {
    previewOffset.x = 0;
    previewOffset.y = 0;
    return;
  }

  const rect = previewViewport.value.getBoundingClientRect();
  const renderedWidth = previewNaturalSize.width * previewRenderedScale.value;
  const renderedHeight = previewNaturalSize.height * previewRenderedScale.value;
  const maxX = Math.max(0, (renderedWidth - rect.width) / 2);
  const maxY = Math.max(0, (renderedHeight - rect.height) / 2);

  previewOffset.x = maxX > 0 ? clamp(previewOffset.x, -maxX, maxX) : 0;
  previewOffset.y = maxY > 0 ? clamp(previewOffset.y, -maxY, maxY) : 0;
}

function handlePreviewWheel(event: WheelEvent) {
  if (!previewDialog.value) {
    return;
  }

  event.preventDefault();
  const direction = event.deltaY > 0 ? -1 : 1;
  const nextScale = previewScale.value + direction * PREVIEW_SCALE_STEP;

  setPreviewScale(nextScale, { clientX: event.clientX, clientY: event.clientY });
}

function togglePreviewZoom(event: MouseEvent) {
  if (Math.abs(previewScale.value - DEFAULT_PREVIEW_SCALE) > ZOOM_EPSILON) {
    resetPreviewZoom();
    return;
  }

  setPreviewScale(previewActualSizeScale.value, { clientX: event.clientX, clientY: event.clientY });
}

function startPreviewPan(event: PointerEvent) {
  if (previewScale.value <= DEFAULT_PREVIEW_SCALE + ZOOM_EPSILON || event.button !== 0) {
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

function setCompareFromClientX(clientX: number) {
  if (!previewViewport.value) {
    return;
  }

  const rect = previewViewport.value.getBoundingClientRect();
  if (rect.width <= 0) {
    return;
  }

  compareValue.value = Math.round(clamp(((clientX - rect.left) / rect.width) * 100, 0, 100));
}

function startCompareDrag(event: PointerEvent) {
  event.preventDefault();
  event.stopPropagation();
  isDraggingCompare.value = true;
  setCompareFromClientX(event.clientX);
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function moveCompareDrag(event: PointerEvent) {
  if (!isDraggingCompare.value) {
    return;
  }

  event.preventDefault();
  setCompareFromClientX(event.clientX);
}

function endCompareDrag(event: PointerEvent) {
  if (!isDraggingCompare.value) {
    return;
  }

  isDraggingCompare.value = false;
  const target = event.currentTarget as HTMLElement;
  if (target.hasPointerCapture(event.pointerId)) {
    target.releasePointerCapture(event.pointerId);
  }
}

function handleCompareKeydown(event: KeyboardEvent) {
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    compareValue.value = clamp(compareValue.value - 1, 0, 100);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    compareValue.value = clamp(compareValue.value + 1, 0, 100);
  } else if (event.key === "Home") {
    event.preventDefault();
    compareValue.value = 0;
  } else if (event.key === "End") {
    event.preventDefault();
    compareValue.value = 100;
  }
}

function panPreviewBy(deltaX: number, deltaY: number) {
  if (previewScale.value <= DEFAULT_PREVIEW_SCALE + ZOOM_EPSILON) {
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

function canUseBrowserNotifications() {
  return window.isSecureContext && "Notification" in window;
}

async function prepareCompletionNotification() {
  if (!canUseBrowserNotifications()) {
    return false;
  }

  if (Notification.permission === "granted") {
    await ensureNotificationWorker();
    return true;
  }

  if (Notification.permission === "denied") {
    return false;
  }

  try {
    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
      return false;
    }

    await ensureNotificationWorker();
    return true;
  } catch {
    return false;
  }
}

async function ensureNotificationWorker() {
  if (!("serviceWorker" in navigator) || !window.isSecureContext) {
    return null;
  }

  try {
    if (!notificationRegistration) {
      await navigator.serviceWorker.register(NOTIFICATION_WORKER_PATH);
      notificationRegistration = await navigator.serviceWorker.ready;
    }

    return notificationRegistration;
  } catch {
    notificationRegistration = null;
    return null;
  }
}

async function showCompletionNotification(filename: string) {
  if (!shouldShowCompletionNotification()) {
    return;
  }

  markCompletionInTitleIfAway();

  if (!canUseBrowserNotifications() || Notification.permission !== "granted") {
    return;
  }

  const body = filename ? `${filename} 已处理完成，点击查看效果。` : "图片已处理完成，点击查看效果。";
  const options: NotificationOptions = {
    body,
    tag: COMPLETION_NOTIFICATION_TAG,
    requireInteraction: true,
    data: { url: buildResultUrl(), messageType: SHOW_RESULT_MESSAGE },
  };

  const registration = await ensureNotificationWorker();
  if (registration) {
    try {
      await registration.showNotification("图片已处理完成", options);
      return;
    } catch {
      // Fall through to the page-level Notification API.
    }
  }

  closeActiveCompletionNotification();
  activeCompletionNotification = new Notification("图片已处理完成", options);
  activeCompletionNotification.onclick = () => {
    closeActiveCompletionNotification();
    void focusResultPanel();
  };
}

function shouldShowCompletionNotification() {
  return document.visibilityState !== "visible" || !document.hasFocus();
}

function markCompletionInTitleIfAway() {
  if (shouldShowCompletionNotification()) {
    document.title = "图片已处理完成 · 人像美颜";
  }
}

function restoreDocumentTitle() {
  document.title = defaultDocumentTitle;
}

function closeActiveCompletionNotification() {
  if (activeCompletionNotification) {
    activeCompletionNotification.close();
    activeCompletionNotification = null;
  }
}

function buildResultUrl() {
  return `${window.location.origin}${window.location.pathname}${window.location.search}#result-panel`;
}

async function focusResultPanel() {
  closeActiveCompletionNotification();
  restoreDocumentTitle();
  window.focus();

  if (window.location.hash !== "#result-panel") {
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#result-panel`);
  }

  await nextTick();
  resultPanel.value?.scrollIntoView({ behavior: "smooth", block: "center" });
  resultPanel.value?.focus({ preventScroll: true });
}

function handleCompletionNotificationMessage(event: MessageEvent) {
  if (event.data?.type === SHOW_RESULT_MESSAGE) {
    void focusResultPanel();
  }
}

function handlePageFocus() {
  restoreDocumentTitle();
}

function handleVisibilityChange() {
  if (document.visibilityState === "visible") {
    restoreDocumentTitle();
  }
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

function handleWindowResize() {
  if (!previewDialog.value) {
    return;
  }

  refreshPreviewFitScale();
}

function resetAll() {
  selectedFile.value = null;
  outputFileName.value = "";
  outputSize.value = "";
  processingMessage.value = "";
  errorMessage.value = "";
  progress.value = 0;
  compareValue.value = 50;
  resetFeatureDefaults();
  closePreview();
  revokeOriginalUrl();
  revokeResultUrl();
  if (fileInput.value) {
    fileInput.value.value = "";
  }
}

function resetFeatureDefaults() {
  for (const feature of features) {
    const defaults = featureDefaultsByKey[feature.key];
    feature.enabled = defaults.enabled;
    feature.strength = defaults.strength;
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
  window.addEventListener("resize", handleWindowResize);
  window.addEventListener("focus", handlePageFocus);
  document.addEventListener("visibilitychange", handleVisibilityChange);
  navigator.serviceWorker?.addEventListener("message", handleCompletionNotificationMessage);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handlePreviewKeydown);
  window.removeEventListener("resize", handleWindowResize);
  window.removeEventListener("focus", handlePageFocus);
  document.removeEventListener("visibilitychange", handleVisibilityChange);
  navigator.serviceWorker?.removeEventListener("message", handleCompletionNotificationMessage);
  closeActiveCompletionNotification();
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

      <div id="result-panel" ref="resultPanel" class="panel result-panel" tabindex="-1">
        <div class="panel-heading">
          <Sparkles :size="20" />
          <h2>处理结果</h2>
        </div>

        <div v-if="resultUrl" class="compare-frame result-frame">
          <img class="compare-image" :src="originalUrl" alt="修图前" draggable="false" />
          <div class="compare-after" :style="compareAfterStyle">
            <img class="compare-image" :src="resultUrl" alt="修图后" draggable="false" />
          </div>
          <div class="compare-divider" :style="compareDividerStyle">
            <span />
          </div>
          <input
            v-model.number="compareValue"
            class="compare-slider"
            type="range"
            min="0"
            max="100"
            aria-label="拖拽查看前后对比"
          />
          <button
            class="preview-icon-button"
            type="button"
            aria-label="放大查看处理结果"
            title="放大查看处理结果"
            @click="openResultPreview"
          >
            <Maximize2 :size="17" />
          </button>
        </div>
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
                class="icon-button actual-size-button"
                type="button"
                aria-label="按原图像素查看"
                title="按原图像素查看"
                :disabled="!canUseActualSizePreview"
                @click="zoomPreviewActualSize"
              >
                1:1
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
            :class="{ 'is-pannable': isPreviewPannable, 'is-panning': isPanningPreview }"
            @wheel="handlePreviewWheel"
            @dblclick="togglePreviewZoom"
            @pointerdown="startPreviewPan"
            @pointermove="movePreviewPan"
            @pointerup="endPreviewPan"
            @pointercancel="endPreviewPan"
          >
            <div v-if="previewSingleDialog" class="lightbox-image-anchor" :style="previewAnchorStyle">
              <img
                class="lightbox-image"
                :src="previewSingleDialog.src"
                :alt="previewSingleDialog.title"
                :style="previewImageStyle"
                draggable="false"
                @load="handlePreviewImageLoad"
              />
            </div>
            <template v-else-if="previewCompareDialog">
              <div class="lightbox-compare-layer">
                <div class="lightbox-image-anchor" :style="previewAnchorStyle">
                  <img
                    class="lightbox-image"
                    :src="previewCompareDialog.beforeSrc"
                    alt="修图前"
                    :style="previewImageStyle"
                    draggable="false"
                    @load="handlePreviewImageLoad"
                  />
                </div>
              </div>
              <div class="lightbox-compare-layer lightbox-compare-after" :style="lightboxCompareAfterStyle">
                <div class="lightbox-image-anchor" :style="previewAnchorStyle">
                  <img
                    class="lightbox-image"
                    :src="previewCompareDialog.afterSrc"
                    alt="修图后"
                    :style="previewImageStyle"
                    draggable="false"
                  />
                </div>
              </div>
              <div
                class="compare-divider lightbox-compare-divider"
                :style="compareDividerStyle"
                role="slider"
                tabindex="0"
                aria-label="拖拽查看前后对比"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="compareValue"
                @keydown="handleCompareKeydown"
                @pointerdown="startCompareDrag"
                @pointermove="moveCompareDrag"
                @pointerup="endCompareDrag"
                @pointercancel="endCompareDrag"
              >
                <span />
              </div>
            </template>
          </div>
        </div>
      </div>
    </Teleport>
  </main>
</template>
