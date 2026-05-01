import * as ort from 'onnxruntime-web';

console.log('[AI Worker] Multi-model module loaded.');
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const FEATURE_MEAN = [
  0.5381992334717343, 0.692648546790163, 0.07948696901948259, 0.031646758224209516,
  648.3677659214309, -35.40995455126158, 28.769947780261898, 0.2797828169913766
];
const FEATURE_SCALE = [
  0.15490431488635853, 0.2297531086955083, 0.066732169187364, 0.037331699259384024,
  476.44254697205355, 19.722409382463397, 96.85572689994291, 0.4488924061595902
];

const SCORE_BY_LABEL = {
  'Focused': 100,
  'Slouching': 60,
  'Leaning on Desk': 45,
  'Looking Away': 30,
  'Using Phone': 20,
  'Absence': 5
};

let postureSession = null;
let yoloSession = null;
let currentSmoothScore = 100;
let labelHistory = [];
const MAX_HISTORY = 25; // 5 seconds of smoothing

let lastFlushAt = Date.now();
let flushIntervalMs = 4000;
let yoloCounter = 0;

let latestLandmarks = null;
let latestFaceLandmarks = null;
let frameWidth = 640;
let frameHeight = 480;

self.onmessage = async (event) => {
  const { type, landmarks, faceLandmarks, width, height, frame } = event.data;

  if (type === 'init') {
    await initWorker(event.data);
  } else if (type === 'process_landmarks') {
    latestLandmarks = landmarks;
    latestFaceLandmarks = faceLandmarks;
    if (width) frameWidth = width;
    if (height) frameHeight = height;

    const result = await classify(frame);
    if (result) {
      currentSmoothScore = (currentSmoothScore * 0.9) + (result.score * 0.1);
      self.postMessage({ 
        type: 'inference_result', 
        label: result.label, 
        score: Math.round(currentSmoothScore) 
      });
      await flushScores(Date.now(), result.label);
    }
    if (frame && frame.close) frame.close();
  }
};

async function initWorker(config) {
  try {
    const { modelUrl, yoloUrl = '/models/yolo26s.onnx', flushIntervalMs: interval } = config;
    if (interval) flushIntervalMs = interval;
    postureSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    try { yoloSession = await ort.InferenceSession.create(yoloUrl, { executionProviders: ['wasm'] }); } catch (e) { }
    self.postMessage({ type: 'worker_ready' });
  } catch (error) { }
}

async function classify(frame) {
  if (!postureSession) return null;

  // Immediate Absence if no landmarks detected at all
  if (!latestLandmarks || latestLandmarks.length === 0) {
    labelHistory.push('Absence');
    if (labelHistory.length > MAX_HISTORY) labelHistory.shift();
    const smoothLabel = getMode(labelHistory);
    return { label: smoothLabel, score: SCORE_BY_LABEL[smoothLabel] };
  }

  let rawLabel = 'Focused';
  const vNose = latestLandmarks[0]?.visibility ?? 0;
  const vLS = latestLandmarks[11]?.visibility ?? 0;
  const vRS = latestLandmarks[12]?.visibility ?? 0;

  // Broad Absence check based on key landmark visibility
  if (vNose < 0.2 || (vLS < 0.2 && vRS < 0.2)) {
    rawLabel = 'Absence';
  } else {
    yoloCounter++;
    if (yoloSession && frame && yoloCounter % 10 === 0) {
      const hasPhone = await detectPhone(frame);
      if (hasPhone) rawLabel = 'Using Phone';
      else rawLabel = await predictPosture();
    } else {
      rawLabel = await predictPosture();
    }
  }

  labelHistory.push(rawLabel);
  if (labelHistory.length > MAX_HISTORY) labelHistory.shift();
  const smoothLabel = getMode(labelHistory);
  return { label: smoothLabel, score: SCORE_BY_LABEL[smoothLabel] || 55 };
}

async function predictPosture() {
  try {
    const features = extractFeatures();
    const scaled = features.map((v, i) => (v - FEATURE_MEAN[i]) / FEATURE_SCALE[i]);
    const tensor = new ort.Tensor('float32', Float32Array.from(scaled), [1, 8]);
    const output = await postureSession.run({ [postureSession.inputNames[0]]: tensor });
    const rawIndex = output[postureSession.outputNames[0]]?.data?.[0];
    return mapLabel(String(rawIndex));
  } catch (e) { return 'Absence'; }
}

function getMode(arr) {
  const counts = {};
  let maxCount = 0;
  let mode = arr[0];
  for (const val of arr) {
    counts[val] = (counts[val] || 0) + 1;
    if (counts[val] > maxCount) { maxCount = counts[val]; mode = val; }
  }
  return mode;
}

async function detectPhone(frame) {
  try {
    const canvas = new OffscreenCanvas(640, 640);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(frame, 0, 0, 640, 640);
    const imgData = ctx.getImageData(0, 0, 640, 640);
    const floatData = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
      floatData[i] = imgData.data[i * 4] / 255.0;
      floatData[i + 640 * 640] = imgData.data[i * 4 + 1] / 255.0;
      floatData[i + 2 * 640 * 640] = imgData.data[i * 4 + 2] / 255.0;
    }
    const tensor = new ort.Tensor('float32', floatData, [1, 3, 640, 640]);
    const output = await yoloSession.run({ [yoloSession.inputNames[0]]: tensor });
    const detections = output[yoloSession.outputNames[0]].data;
    for (let i = 0; i < 8400; i++) {
      if (detections[71 * 8400 + i] > 0.5) return true;
    }
  } catch (e) { }
  return false;
}

function extractFeatures() {
  const w = frameWidth; const h = frameHeight;
  const getPx = (lm) => ({ x: (1.0 - lm.x) * w, y: lm.y * h, z: lm.z * w });
  const dist = (p1, p2) => Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
  const nose = getPx(latestLandmarks[0]);
  const lS = getPx(latestLandmarks[11]);
  const rS = getPx(latestLandmarks[12]);
  const lE = getPx(latestLandmarks[7]);
  const rE = getPx(latestLandmarks[8]);
  const lW = getPx(latestLandmarks[15]);
  const rW = getPx(latestLandmarks[16]);
  const midS = { x: (lS.x + rS.x) / 2, y: (lS.y + rS.y) / 2 };
  const midE = { x: (lE.x + rE.x) / 2, y: (lE.y + rE.y) / 2 };
  const sW = dist(lS, rS) || 1;
  const neck_ratio = Math.abs(midS.y - midE.y) / sW;
  const forward_lean_z = ((latestLandmarks[11].z + latestLandmarks[12].z) / 2) - latestLandmarks[0].z;
  const shoulder_tilt_ratio = Math.abs(lS.y - rS.y) / sW;
  const head_tilt_ratio = Math.abs(lE.y - rE.y) / sW;
  let min_h2f = 999.0;
  [lW, rW].forEach(w => {
    min_h2f = Math.min(min_h2f, Math.min(dist(w, lE), dist(w, nose)) / sW);
  });
  let px = 0, py = 0;
  if (latestFaceLandmarks) {
    const fN = latestFaceLandmarks[1], fL = latestFaceLandmarks[33], fR = latestFaceLandmarks[263];
    const lx = (1.0 - fL.x) * w, lz = fL.z * w;
    const rx = (1.0 - fR.x) * w, rz = fR.z * w;
    py = (Math.atan2(rz - lz, rx - lx) * 180) / Math.PI;
    const dy = (fN.y * h) - ((fL.y * h + fR.y * h) / 2), dz = (fN.z * w) - ((fL.z * w + fR.z * w) / 2);
    px = (Math.atan2(dz, dy) * 180) / Math.PI;
  }
  return [neck_ratio, forward_lean_z, shoulder_tilt_ratio, head_tilt_ratio, min_h2f, px, py, (lW.y < midS.y + sW * 0.5 || rW.y < midS.y + sW * 0.5) ? 1 : 0];
}

function mapLabel(raw) {
  const m = { '0': 'Focused', '1': 'Slouching', '2': 'Leaning on Desk', '3': 'Looking Away' };
  return m[raw] || raw;
}

async function flushScores(now, currentLabel) {
  const interval = (now - lastFlushAt < 10000) ? 1000 : flushIntervalMs;
  if (now - lastFlushAt < interval) return;
  
  self.postMessage({ 
    type: 'score_update', 
    averageScore: Math.round(currentSmoothScore), 
    status: currentLabel, // Use the actual posture label
    sampledFps: 5, 
    sampleCount: 5 
  });
  lastFlushAt = now;
}
