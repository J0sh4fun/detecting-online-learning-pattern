import * as ort from 'onnxruntime-web';

let session = null;
let targetFps = 1;
let flushIntervalMs = 4000;
let lastFlushAt = Date.now();
let nextVerifyAt = Date.now() + randomVerifyDelayMs();
let sampleBuffer = [];
let previousGray = null;

function randomVerifyDelayMs() {
  const min = 5 * 60 * 1000;
  const max = 10 * 60 * 1000;
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values, avg) {
  if (!values.length) return 0;
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function inferStatus(score) {
  if (score >= 80) return 'Focused';
  if (score >= 60) return 'Slightly Distracted';
  if (score >= 40) return 'Distracted';
  return 'Low Concentration';
}

function clampScore(value) {
  return Math.max(0, Math.min(100, value));
}

function frameToFeatureVector(frame) {
  const canvas = new OffscreenCanvas(64, 64);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(frame, 0, 0, 64, 64);

  const imageData = context.getImageData(0, 0, 64, 64).data;
  const gray = new Float32Array(64 * 64);
  for (let i = 0; i < gray.length; i += 1) {
    const index = i * 4;
    gray[i] = 0.2126 * imageData[index] + 0.7152 * imageData[index + 1] + 0.0722 * imageData[index + 2];
  }

  const brightness = mean(Array.from(gray)) / 255;
  const contrast = std(Array.from(gray), brightness * 255) / 255;

  let motion = 0;
  if (previousGray) {
    for (let i = 0; i < gray.length; i += 1) {
      motion += Math.abs(gray[i] - previousGray[i]);
    }
    motion = motion / gray.length / 255;
  }
  previousGray = gray;

  const centerIndex = (32 * 64) + 32;
  const centerLum = gray[centerIndex] / 255;
  const upperLum = gray[(16 * 64) + 32] / 255;
  const lowerLum = gray[(48 * 64) + 32] / 255;
  const leftLum = gray[(32 * 64) + 16] / 255;
  const rightLum = gray[(32 * 64) + 48] / 255;

  return [
    brightness,
    contrast,
    motion,
    Math.abs(upperLum - lowerLum),
    Math.abs(leftLum - rightLum),
    upperLum - centerLum,
    lowerLum - centerLum,
    motion > 0.18 ? 1 : 0,
    centerLum < 0.08 ? 1 : 0,
  ];
}

async function runModel(features) {
  if (!session) {
    return clampScore(60 + (features[0] * 20) - (features[2] * 45));
  }
  try {
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 9]);
    const result = await session.run({ [inputName]: inputTensor });
    const output = result[outputName];
    if (!output || !output.data || !output.data.length) return 60;
    const value = Number(output.data[0]);
    return clampScore(Number.isFinite(value) ? value : 60);
  } catch {
    return clampScore(60 + (features[0] * 20) - (features[2] * 45));
  }
}

async function handleFrame(frame) {
  const featureVector = frameToFeatureVector(frame);
  const score = await runModel(featureVector);
  sampleBuffer.push(score);

  const now = Date.now();
  if (now - lastFlushAt >= flushIntervalMs && sampleBuffer.length > 0) {
    const averageScore = clampScore(mean(sampleBuffer));
    const payload = {
      type: 'aggregate',
      averageScore,
      status: inferStatus(averageScore),
      sampleCount: sampleBuffer.length,
      fps: targetFps,
      cameraOn: true,
    };
    sampleBuffer = [];
    lastFlushAt = now;
    self.postMessage(payload);
  }

  if (now >= nextVerifyAt) {
    const verifyCanvas = new OffscreenCanvas(frame.width, frame.height);
    const verifyContext = verifyCanvas.getContext('2d');
    verifyContext.drawImage(frame, 0, 0);
    const verifyBlob = await verifyCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.65 });
    const buffer = await verifyBlob.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i += 1) {
      binary += String.fromCharCode(bytes[i]);
    }
    self.postMessage({
      type: 'verify_frame',
      clientScore: clampScore(mean(sampleBuffer) || score),
      frameBase64: `data:image/jpeg;base64,${btoa(binary)}`,
    });
    nextVerifyAt = now + randomVerifyDelayMs();
  }

  frame.close();
}

self.onmessage = async (event) => {
  const { type } = event.data;
  if (type === 'init') {
    targetFps = event.data.targetFps || 1;
    flushIntervalMs = event.data.flushIntervalMs || 4000;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    try {
      session = await ort.InferenceSession.create(event.data.modelUrl, {
        executionProviders: ['wasm'],
      });
    } catch {
      session = null;
    }
  }
  if (type === 'frame' && event.data.frame) {
    await handleFrame(event.data.frame);
  }
};

