import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const PHONE_CLASS_ID = 67;

const FEATURE_MEAN = [
  0.5381992334717343, 0.692648546790163, 0.07948696901948259, 0.031646758224209516,
  648.3677659214309, -35.40995455126158, 28.769947780261898, 0.2797828169913766,
];
const FEATURE_SCALE = [
  0.15490431488635853, 0.2297531086955083, 0.066732169187364, 0.037331699259384024,
  476.44254697205355, 19.722409382463397, 96.85572689994291, 0.4488924061595902,
];

const SCORE_BY_LABEL = {
  Focused: 100,
  Slouching: 70,
  'Leaning on Desk': 45,
  'Looking Away': 30,
  'Using Phone': 20,
  Absence: 5,
};

let postureSession = null;
let yoloSession = null;

let frameWidth = 640;
let frameHeight = 480;
let latestLandmarks = null;
let latestFaceLandmarks = null;

let flushIntervalMs = 2000;
let sampledFps = 5;
let maxHistory = 6;
let yoloCheckInterval = 2;
let phoneConfidenceThreshold = 0.35;

let lastFlushAt = Date.now();
let yoloCounter = 0;
let labelHistory = [];
let currentSmoothScore = 100;
let lastLabel = 'Focused';

self.onmessage = async (event) => {
  const { type } = event.data;
  if (type === 'init') {
    await initWorker(event.data);
    return;
  }

  if (type !== 'process_landmarks') return;

  latestLandmarks = event.data.landmarks;
  latestFaceLandmarks = event.data.faceLandmarks;
  if (event.data.width) frameWidth = event.data.width;
  if (event.data.height) frameHeight = event.data.height;

  const frame = event.data.frame || null;
  try {
    const result = await classify(frame);
    if (!result) return;
    lastLabel = result.label;
    const alpha = result.label === 'Using Phone' ? 0.6 : 0.3;
    currentSmoothScore = (currentSmoothScore * (1 - alpha)) + (result.score * alpha);
    await flushScores(Date.now());
  } catch (error) {
    self.postMessage({
      type: 'worker_error',
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    if (frame && typeof frame.close === 'function') frame.close();
  }
};

async function initWorker(config) {
  const {
    modelUrl,
    yoloUrl = '/models/yolo26s.onnx',
    flushIntervalMs: interval = 2000,
    sampledFps: fps = 5,
    maxHistory: historyWindow = 6,
    yoloCheckInterval: phoneEveryN = 2,
    phoneConfidenceThreshold: phoneThreshold = 0.35,
  } = config;

  flushIntervalMs = interval;
  sampledFps = fps;
  maxHistory = historyWindow;
  yoloCheckInterval = Math.max(1, phoneEveryN);
  phoneConfidenceThreshold = phoneThreshold;
  lastFlushAt = Date.now();

  try {
    postureSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  } catch (error) {
    self.postMessage({
      type: 'worker_error',
      message: `Cannot load posture ONNX (${modelUrl}): ${error instanceof Error ? error.message : String(error)}`,
    });
    return;
  }

  try {
    yoloSession = await ort.InferenceSession.create(yoloUrl, { executionProviders: ['wasm'] });
  } catch (error) {
    yoloSession = null;
    self.postMessage({
      type: 'worker_error',
      message: `Phone detector disabled. Failed to load ${yoloUrl}: ${error instanceof Error ? error.message : String(error)}`,
    });
  }

  self.postMessage({ type: 'worker_ready', yoloEnabled: Boolean(yoloSession) });
}

async function classify(frame) {
  if (!postureSession) return null;

  if (!latestLandmarks || latestLandmarks.length === 0) {
    pushLabel('Absence');
    return { label: 'Absence', score: SCORE_BY_LABEL.Absence };
  }

  const visibilityNose = latestLandmarks[0]?.visibility ?? 0;
  const visibilityLeftShoulder = latestLandmarks[11]?.visibility ?? 0;
  const visibilityRightShoulder = latestLandmarks[12]?.visibility ?? 0;
  if (visibilityNose < 0.2 || (visibilityLeftShoulder < 0.2 && visibilityRightShoulder < 0.2)) {
    pushLabel('Absence');
    return { label: 'Absence', score: SCORE_BY_LABEL.Absence };
  }

  let label = await predictPosture();

  yoloCounter += 1;
  const shouldCheckPhone = Boolean(yoloSession && frame && (yoloCounter % yoloCheckInterval === 0));
  if (shouldCheckPhone) {
    const hasPhone = await detectPhone(frame);
    if (hasPhone) {
      label = 'Using Phone';
    }
  }

  pushLabel(label);
  const smoothLabel = label === 'Using Phone' ? label : getMode(labelHistory);
  return { label: smoothLabel, score: SCORE_BY_LABEL[smoothLabel] ?? 55 };
}

function pushLabel(label) {
  labelHistory.push(label);
  if (labelHistory.length > maxHistory) labelHistory.shift();
}

async function predictPosture() {
  const features = extractFeatures();
  const scaled = features.map((value, idx) => (value - FEATURE_MEAN[idx]) / FEATURE_SCALE[idx]);
  const tensor = new ort.Tensor('float32', Float32Array.from(scaled), [1, 8]);
  const output = await postureSession.run({ [postureSession.inputNames[0]]: tensor });

  const labelOutput = output.output_label
    || output[postureSession.outputNames.find((name) => name.includes('label'))]
    || output[postureSession.outputNames[0]];

  const raw = labelOutput?.data?.[0];
  return mapLabel(raw);
}

async function detectPhone(frame) {
  const canvas = new OffscreenCanvas(640, 640);
  const context = canvas.getContext('2d');
  context.drawImage(frame, 0, 0, 640, 640);

  const image = context.getImageData(0, 0, 640, 640);
  const chw = new Float32Array(3 * 640 * 640);
  for (let i = 0; i < 640 * 640; i += 1) {
    chw[i] = image.data[i * 4] / 255;
    chw[i + 640 * 640] = image.data[(i * 4) + 1] / 255;
    chw[i + (2 * 640 * 640)] = image.data[(i * 4) + 2] / 255;
  }

  const input = new ort.Tensor('float32', chw, [1, 3, 640, 640]);
  const output = await yoloSession.run({ [yoloSession.inputNames[0]]: input });
  const primary = output[yoloSession.outputNames[0]];
  const data = primary?.data;
  const dims = primary?.dims || [];
  if (!data) return false;

  // Exported YOLO from this project returns [1, 300, 6]: [x1, y1, x2, y2, conf, cls]
  if (dims.length === 3 && dims[2] === 6) {
    for (let i = 0; i < dims[1]; i += 1) {
      const base = i * 6;
      const conf = data[base + 4];
      const cls = data[base + 5];
      if (conf >= phoneConfidenceThreshold && Math.round(cls) === PHONE_CLASS_ID) return true;
    }
    return false;
  }

  // Fallback for [1, classes+4, num_boxes] style.
  if (dims.length === 3 && dims[1] >= 84) {
    const numBoxes = dims[2];
    const classOffset = (4 + PHONE_CLASS_ID) * numBoxes;
    for (let i = 0; i < numBoxes; i += 1) {
      if (data[classOffset + i] >= phoneConfidenceThreshold) return true;
    }
  }
  return false;
}

function extractFeatures() {
  const w = frameWidth;
  const h = frameHeight;
  const toPx = (lm) => ({ x: (1 - lm.x) * w, y: lm.y * h, z: lm.z * w });
  const dist = (p1, p2) => Math.hypot(p2.x - p1.x, p2.y - p1.y);

  const nose = toPx(latestLandmarks[0]);
  const leftShoulder = toPx(latestLandmarks[11]);
  const rightShoulder = toPx(latestLandmarks[12]);
  const leftEar = toPx(latestLandmarks[7]);
  const rightEar = toPx(latestLandmarks[8]);
  const leftWrist = toPx(latestLandmarks[15]);
  const rightWrist = toPx(latestLandmarks[16]);

  const midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
  const midEar = { x: (leftEar.x + rightEar.x) / 2, y: (leftEar.y + rightEar.y) / 2 };
  const shoulderWidth = dist(leftShoulder, rightShoulder) || 1;

  const neckRatio = Math.abs(midShoulder.y - midEar.y) / shoulderWidth;
  const forwardLeanZ = ((latestLandmarks[11].z + latestLandmarks[12].z) / 2) - latestLandmarks[0].z;
  const shoulderTiltRatio = Math.abs(leftShoulder.y - rightShoulder.y) / shoulderWidth;
  const headTiltRatio = Math.abs(leftEar.y - rightEar.y) / shoulderWidth;

  let minHandToFace = 999;
  for (const wrist of [leftWrist, rightWrist]) {
    minHandToFace = Math.min(minHandToFace, Math.min(dist(wrist, leftEar), dist(wrist, nose)) / shoulderWidth);
  }

  let poseX = 0;
  let poseY = 0;
  if (latestFaceLandmarks) {
    const faceNose = latestFaceLandmarks[1];
    const faceLeftEye = latestFaceLandmarks[33];
    const faceRightEye = latestFaceLandmarks[263];
    const lx = (1 - faceLeftEye.x) * w;
    const lz = faceLeftEye.z * w;
    const rx = (1 - faceRightEye.x) * w;
    const rz = faceRightEye.z * w;
    poseY = (Math.atan2(rz - lz, rx - lx) * 180) / Math.PI;

    const dy = (faceNose.y * h) - ((faceLeftEye.y * h + faceRightEye.y * h) / 2);
    const dz = (faceNose.z * w) - ((faceLeftEye.z * w + faceRightEye.z * w) / 2);
    poseX = (Math.atan2(dz, dy) * 180) / Math.PI;
  }

  const wristElevated = (leftWrist.y < midShoulder.y + shoulderWidth * 0.5 || rightWrist.y < midShoulder.y + shoulderWidth * 0.5) ? 1 : 0;
  return [neckRatio, forwardLeanZ, shoulderTiltRatio, headTiltRatio, minHandToFace, poseX, poseY, wristElevated];
}

function mapLabel(raw) {
  const normalized = String(raw ?? '');
  const byIndex = {
    0: 'Focused',
    1: 'Leaning on Desk',
    2: 'Looking Away',
    3: 'Slouching',
  };
  if (/^\d+$/.test(normalized)) {
    return byIndex[Number(normalized)] || 'Focused';
  }
  if (normalized === 'Slouched') return 'Slouching';
  if (normalized in SCORE_BY_LABEL) return normalized;
  return 'Focused';
}

function getMode(values) {
  if (!values.length) return 'Focused';
  const counts = new Map();
  let best = values[values.length - 1];
  let bestCount = 0;
  for (const value of values) {
    const count = (counts.get(value) || 0) + 1;
    counts.set(value, count);
    if (count > bestCount) {
      best = value;
      bestCount = count;
    }
  }
  return best;
}

async function flushScores(now) {
  if (now - lastFlushAt < flushIntervalMs) return;
  self.postMessage({
    type: 'score_update',
    averageScore: Math.round(currentSmoothScore),
    status: lastLabel,
    sampledFps,
    sampleCount: Math.max(1, labelHistory.length),
    cameraOn: true,
  });
  lastFlushAt = now;
}
