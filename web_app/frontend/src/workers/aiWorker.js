import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const PHONE_CLASS_ID = 67;

const FEATURE_MEAN = [
  0.5912850397877984,
  0.6386924137931034,
  0.13853071618037135,
  0.08592355437665783,
  2.9888225994694957,
  -33.39181946949602,
  0.24842403183023978,
  0.3819628647214854,
  0.4033495490716181,
  -0.04561442970822282,
  -0.571819151193634,
  -0.07412790450928383,
  -0.4676516710875332,
  33.39181946949602,
  119.03989936339522,
  0.7119363395225464,
  0.9743484350132626,
  0.9830701326259946,
  0.9778890716180372,
  0.294037400530504,
  0.22856976127320958,
  0.7740053050397878,
  0.49124668435013263,
];

const FEATURE_SCALE = [
  1.6221677061560418,
  0.3845630444715387,
  0.19982810638548598,
  0.3261975880635139,
  2.1237192721460363,
  23.82927372115119,
  141.6092543769391,
  0.4858675073466445,
  0.17622064008087374,
  0.8358520209822504,
  1.629131340162789,
  1.1544485076305526,
  1.5008452511670842,
  23.82927372115119,
  76.69905474805392,
  0.45286089253741474,
  0.09118692367480632,
  0.03396455640848045,
  0.075234210764983,
  0.3249106583369266,
  0.2696136536936384,
  0.8604872655590918,
  0.4999233735935277,
];

const SCORE_BY_LABEL = {
  'Focused': 100,
  'Slouching': 70,
  'Leaning on Desk': 45,
  'Looking Away': 30,
  'Using Phone': 20,
  'Absence': 5,
};

let postureSession = null;
let yoloSession = null;

let frameWidth = 640;
let frameHeight = 480;
let latestLandmarks = null;
let latestFaceLandmarks = null;

let flushIntervalMs = 500;
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
    flushIntervalMs: interval = 500,
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

  const poseAbsent = !latestLandmarks || latestLandmarks.length === 0;
  const faceAbsent = !latestFaceLandmarks || latestFaceLandmarks.length === 0;

  if (poseAbsent && faceAbsent) {
    pushLabel('Absence');
    return { label: 'Absence', score: SCORE_BY_LABEL.Absence };
  }

  if (poseAbsent) {
    pushLabel('Focused');
    return { label: 'Focused', score: SCORE_BY_LABEL.Focused };
  }

  const visibilityNose = latestLandmarks[0]?.visibility ?? 0;
  const visibilityLeftShoulder = latestLandmarks[11]?.visibility ?? 0;
  const visibilityRightShoulder = latestLandmarks[12]?.visibility ?? 0;
  if (visibilityNose < 0.2 || (visibilityLeftShoulder < 0.2 && visibilityRightShoulder < 0.2)) {
    if (!faceAbsent) {
      pushLabel('Focused');
      return { label: 'Focused', score: SCORE_BY_LABEL.Focused };
    }
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
  const tensor = new ort.Tensor('float32', Float32Array.from(scaled), [1, 23]);
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
  const midpoint = (p1, p2) => ({ x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 });

  const noseLm = latestLandmarks[0];
  const lShoulderLm = latestLandmarks[11];
  const rShoulderLm = latestLandmarks[12];
  const lEarLm = latestLandmarks[7];
  const rEarLm = latestLandmarks[8];
  const lWristLm = latestLandmarks[15];
  const rWristLm = latestLandmarks[16];

  const nose = toPx(noseLm);
  const lShoulder = toPx(lShoulderLm);
  const rShoulder = toPx(rShoulderLm);
  const lEar = toPx(lEarLm);
  const rEar = toPx(rEarLm);
  const lWrist = toPx(lWristLm);
  const rWrist = toPx(rWristLm);

  const shoulderWidth = dist(lShoulder, rShoulder) || 1.0;
  const midShoulder = midpoint(lShoulder, rShoulder);
  const midEar = midpoint(lEar, rEar);
  const midShoulderZ = (lShoulderLm.z * w + rShoulderLm.z * w) / 2.0;

  const neckRatio = Math.abs(midShoulder.y - midEar.y) / shoulderWidth;
  const forwardLeanZ = midShoulderZ - nose.z;
  const shoulderTiltRatio = Math.abs(lShoulder.y - rShoulder.y) / shoulderWidth;
  const headTiltRatio = Math.abs(lEar.y - rEar.y) / shoulderWidth;

  const chestLevel = midShoulder.y + (shoulderWidth * 0.5);

  let minHandToFace = 999.0;
  let wristElevated = false;
  let visibleWristCount = 0;

  const wrists = [
    { lm: lWristLm, px: lWrist },
    { lm: rWristLm, px: rWrist }
  ];

  for (const wrist of wrists) {
    if (wrist.lm.visibility > 0.2) {
      visibleWristCount += 1;
      const distFace = Math.min(dist(wrist.px, lEar), dist(wrist.px, nose));
      minHandToFace = Math.min(minHandToFace, distFace / shoulderWidth);
      if (wrist.px.y < chestLevel) {
        wristElevated = true;
      }
    }
  }

  let poseX = 0;
  let poseY = 0;
  let faceDetected = 0;

  if (latestFaceLandmarks && latestFaceLandmarks.length > 0) {
    faceDetected = 1;
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

  const headOffsetXRatio = (midEar.x - midShoulder.x) / shoulderWidth;
  const headOffsetYRatio = (midEar.y - midShoulder.y) / shoulderWidth;
  const noseShoulderXRatio = (nose.x - midShoulder.x) / shoulderWidth;
  const noseShoulderYRatio = (nose.y - midShoulder.y) / shoulderWidth;

  // Trả về chính xác 23 đặc trưng (Features)
  return [
    neckRatio, forwardLeanZ, shoulderTiltRatio, headTiltRatio, minHandToFace, poseX, poseY, wristElevated ? 1.0 : 0.0,
    shoulderWidth / Math.max(w, 1.0),
    headOffsetXRatio, headOffsetYRatio,
    noseShoulderXRatio, noseShoulderYRatio,
    Math.abs(poseX), Math.abs(poseY),
    faceDetected,
    noseLm.visibility,
    Math.min(lShoulderLm.visibility, rShoulderLm.visibility),
    Math.min(lEarLm.visibility, rEarLm.visibility),
    lWristLm.visibility, rWristLm.visibility,
    visibleWristCount,
    visibleWristCount > 0 ? 1.0 : 0.0
  ];
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
