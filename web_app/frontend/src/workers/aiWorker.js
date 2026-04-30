let targetFps = 1;
let flushIntervalMs = 4000;
let lastFlushAt = Date.now();
let nextVerifyAt = Date.now() + randomVerifyDelayMs();

function randomVerifyDelayMs() {
  const min = 5 * 60 * 1000;
  const max = 10 * 60 * 1000;
  return Math.floor(Math.random() * (max - min + 1)) + min;
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

async function handleFrame(frame) {
  const now = Date.now();
  if (now - lastFlushAt >= flushIntervalMs) {
    self.postMessage({
      type: 'score_frame',
      frameBase64: await frameToBase64(frame, 0.72),
      fps: targetFps,
      cameraOn: true,
    });
    lastFlushAt = now;
  }

  if (now >= nextVerifyAt) {
    self.postMessage({
      type: 'verify_frame',
      frameBase64: await frameToBase64(frame, 0.65),
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
  }
  if (type === 'frame' && event.data.frame) {
    await handleFrame(event.data.frame);
  }
};

