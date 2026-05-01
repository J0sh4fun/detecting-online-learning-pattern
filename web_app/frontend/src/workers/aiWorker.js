import * as ort from 'onnxruntime-web';

const FEATURE_MEAN = [
  0.5381992334717343,
  0.692648546790163,
  0.07948696901948259,
  0.031646758224209516,
  648.3677659214309,
  -35.40995455126158,
  28.769947780261898,
  0.2797828169913766,
];

const FEATURE_SCALE = [
  0.15490431488635853,
  0.2297531086955083,
  0.066732169187364,
  0.037331699259384024,
  476.44254697205355,
  19.722409382463397,
  96.85572689994291,
  0.4488924061595902,
];

const SCORE_BY_LABEL = {
  Focused: 95,
  Slouched: 70,
  Slouching: 70,
  'Looking Away': 62,
  'Leaning on Desk': 45,
  'Using Phone': 20,
  Absence: 5,
};

let targetFps = 1;
let flushIntervalMs = 4000;
let lastFlushAt = Date.now();
let lastInferenceAt = 0;
let nextVerifyAt = Date.now() + randomVerifyDelayMs();
let modelSession = null;
let modelInputName = 'float_input';
let scoreBuffer = [];
let labelBuffer = [];
let scratchCanvas = new OffscreenCanvas(160, 120);
let scratchContext = scratchCanvas.getContext('2d', { willReadFrequently: true });

function randomVerifyDelayMs() {
  const min = 5 * 60 * 1000;
  const max = 10 * 60 * 1000;
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function average(values) {
  if (!values.length) return 100;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function majorityLabel(labels) {
  if (!labels.length) return 'Focused';
  const counts = new Map();
  for (const label of labels) {
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  let winner = labels[labels.length - 1];
  let winnerCount = 0;
  for (const [label, count] of counts.entries()) {
    if (count > winnerCount) {
      winner = label;
      winnerCount = count;
    }
  }
  return winner;
}

function displayLabel(label) {
  if (label === 'Slouching') return 'Slouched';
  return label || 'Focused';
}

async function frameToBase64(frame, quality = 0.72) {
  const canvas = new OffscreenCanvas(frame.width, frame.height);
  const context = canvas.getContext('2d');
  context.drawImage(frame, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return `data:image/jpeg;base64,${btoa(binary)}`;
}

function extractFeatures(frame) {
  scratchContext.drawImage(frame, 0, 0, scratchCanvas.width, scratchCanvas.height);
  const { data, width, height } = scratchContext.getImageData(0, 0, scratchCanvas.width, scratchCanvas.height);
  const luminance = new Float32Array(width * height);

  let total = 0;
  let top = 0;
  let bottom = 0;
  let left = 0;
  let right = 0;
  let center = 0;
  let edge = 0;
  let topLeft = 0;
  let topRight = 0;
  let centerCount = 0;
  let edgeCount = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const rgb = idx * 4;
      const lum = 0.2126 * data[rgb] + 0.7152 * data[rgb + 1] + 0.0722 * data[rgb + 2];
      luminance[idx] = lum;
      total += lum;

      if (y < height / 2) top += lum;
      else bottom += lum;
      if (x < width / 2) left += lum;
      else right += lum;
      if (x < width / 2 && y < height / 2) topLeft += lum;
      if (x >= width / 2 && y < height / 2) topRight += lum;

      const isCenter = x > width * 0.25 && x < width * 0.75 && y > height * 0.2 && y < height * 0.8;
      if (isCenter) {
        center += lum;
        centerCount += 1;
      } else {
        edge += lum;
        edgeCount += 1;
      }
    }
  }

  const n = width * height;
  const topMean = top / (n / 2);
  const bottomMean = bottom / (n / 2);
  const leftMean = left / (n / 2);
  const rightMean = right / (n / 2);
  const topLeftMean = topLeft / (n / 4);
  const topRightMean = topRight / (n / 4);
  const centerMean = centerCount ? center / centerCount : total / n;
  const edgeMean = edgeCount ? edge / edgeCount : total / n;

  let variance = 0;
  const globalMean = total / n;
  for (let i = 0; i < luminance.length; i += 1) {
    const diff = luminance[i] - globalMean;
    variance += diff * diff;
  }
  variance /= n;

  const neckRatio = clamp((topMean + 1) / (bottomMean + 1), 0, 3);
  const forwardLeanZ = clamp((centerMean - edgeMean) / 60, -2, 2);
  const shoulderTiltRatio = clamp(Math.abs(leftMean - rightMean) / 255, 0, 1);
  const headTiltRatio = clamp(Math.abs(topLeftMean - topRightMean) / 255, 0, 1);
  const handToFaceRatio = clamp(Math.sqrt(variance) * 12, 0, 2000);
  const poseX = clamp(((leftMean - rightMean) / 255) * 90, -90, 90);
  const poseY = clamp(((topMean - bottomMean) / 255) * 90, -90, 90);
  const wristElevated = topMean > bottomMean ? 1 : 0;

  return [
    neckRatio,
    forwardLeanZ,
    shoulderTiltRatio,
    headTiltRatio,
    handToFaceRatio,
    poseX,
    poseY,
    wristElevated,
  ];
}

function standardizeFeatures(features) {
  return features.map((value, idx) => (value - FEATURE_MEAN[idx]) / FEATURE_SCALE[idx]);
}

async function inferLabel(frame) {
  if (!modelSession) {
    return { label: 'Focused', score: 95 };
  }

  const features = extractFeatures(frame);
  const scaled = standardizeFeatures(features);
  const tensor = new ort.Tensor('float32', Float32Array.from(scaled), [1, 8]);
  const output = await modelSession.run({ [modelInputName]: tensor });
  const rawLabel = output.output_label?.data?.[0] || 'Focused';
  const label = displayLabel(String(rawLabel));
  const score = SCORE_BY_LABEL[label] ?? 55;
  return { label, score };
}

async function flushScores(now) {
  if (!scoreBuffer.length || now - lastFlushAt < flushIntervalMs) {
    return;
  }
  const averageScore = average(scoreBuffer);
  const status = majorityLabel(labelBuffer);
  const sampleCount = scoreBuffer.length;
  scoreBuffer = [];
  labelBuffer = [];
  lastFlushAt = now;
  self.postMessage({
    type: 'score_update',
    averageScore,
    status,
    sampleCount,
    sampledFps: targetFps,
    cameraOn: true,
  });
}

async function handleFrame(frame) {
  const now = Date.now();
  const minInterval = 1000 / Math.max(0.2, targetFps);
  try {
    if (now - lastInferenceAt >= minInterval) {
      const { label, score } = await inferLabel(frame);
      scoreBuffer.push(score);
      labelBuffer.push(label);
      lastInferenceAt = now;
    }

    await flushScores(now);

    if (now >= nextVerifyAt) {
      self.postMessage({
        type: 'verify_frame',
        frameBase64: await frameToBase64(frame, 0.65),
      });
      nextVerifyAt = now + randomVerifyDelayMs();
    }
  } catch (error) {
    self.postMessage({
      type: 'worker_error',
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    frame.close();
  }
}

async function initWorker({ targetFps: fps, flushIntervalMs: flushMs, modelUrl }) {
  targetFps = fps || 1;
  flushIntervalMs = flushMs || 4000;
  lastFlushAt = Date.now();
  scoreBuffer = [];
  labelBuffer = [];

  try {
    modelSession = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    modelInputName = modelSession.inputNames[0] || 'float_input';
    self.postMessage({ type: 'worker_ready' });
  } catch (error) {
    modelSession = null;
    self.postMessage({
      type: 'worker_error',
      message: `Failed to load ONNX model (${modelUrl}): ${error instanceof Error ? error.message : String(error)}`,
    });
  }
}

self.onmessage = async (event) => {
  const { type } = event.data;
  if (type === 'init') {
    await initWorker(event.data);
  }
  if (type === 'frame' && event.data.frame) {
    await handleFrame(event.data.frame);
  }
};

