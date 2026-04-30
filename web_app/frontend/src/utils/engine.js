export function calculateDistance(p1, p2) {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

export function getMidpoint(p1, p2) {
  return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
}

export function estimateHeadPose(faceLandmarks, frameWidth, frameHeight) {
  const nose = faceLandmarks[1];
  const leftEye = faceLandmarks[33];
  const rightEye = faceLandmarks[263];

  // Apply horizontal flip to match training (width inversion)
  const lx = (1.0 - leftEye.x) * frameWidth, ly = leftEye.y * frameHeight, lz = leftEye.z * frameWidth;
  const rx = (1.0 - rightEye.x) * frameWidth, ry = rightEye.y * frameHeight, rz = rightEye.z * frameWidth;
  const ny = nose.y * frameHeight, nz = nose.z * frameWidth;

  const dx = rx - lx;
  const dz = rz - lz;
  const poseY = (Math.atan2(dz, dx) * 180) / Math.PI;

  const midEyeY = (ly + ry) / 2;
  const midEyeZ = (lz + rz) / 2;
  const dy = ny - midEyeY;
  const dzPitch = nz - midEyeZ;
  const poseX = (Math.atan2(dzPitch, dy) * 180) / Math.PI;

  return { poseX, poseY, poseZ: 0.0 };
}

export function extractFeatures(landmarks, faceLandmarks, w, h) {
  // Flip the X coordinate because the ML model was trained on flipped frames (cv2.flip(frame, 1))
  const getLandmarkPx = (lm) => ({ x: (1.0 - lm.x) * w, y: lm.y * h, z: lm.z });

  const nose = getLandmarkPx(landmarks[0]); // NOSE
  const lShoulder = getLandmarkPx(landmarks[11]); // LEFT_SHOULDER
  const rShoulder = getLandmarkPx(landmarks[12]); // RIGHT_SHOULDER
  const lEar = getLandmarkPx(landmarks[7]); // LEFT_EAR
  const rEar = getLandmarkPx(landmarks[8]); // RIGHT_EAR
  const lWrist = getLandmarkPx(landmarks[15]); // LEFT_WRIST
  const rWrist = getLandmarkPx(landmarks[16]); // RIGHT_WRIST

  const midShoulderZ = (lShoulder.z + rShoulder.z) / 2;
  const midShoulder = getMidpoint(lShoulder, rShoulder);
  const midEar = getMidpoint(lEar, rEar);
  const shoulderWidth = calculateDistance(lShoulder, rShoulder) || 1;

  const neckRatio = Math.abs(midShoulder.y - midEar.y) / shoulderWidth;
  const forwardLeanZ = midShoulderZ - nose.z;
  
  const shoulderTiltRatio = Math.abs(lShoulder.y - rShoulder.y) / shoulderWidth;
  const headTiltRatio = Math.abs(lEar.y - rEar.y) / shoulderWidth;

  const chestLevel = midShoulder.y + (shoulderWidth * 0.5);
  let wristElevated = false;
  let minHandToFace = 999.0;

  const wrists = [];
  if (landmarks[15].visibility > 0.2) wrists.push(lWrist);
  if (landmarks[16].visibility > 0.2) wrists.push(rWrist);

  for (const wristPx of wrists) {
    const distEar = calculateDistance(wristPx, lEar);
    const distNose = calculateDistance(wristPx, nose);
    const distFace = Math.min(distEar, distNose);
    minHandToFace = Math.min(minHandToFace, distFace / shoulderWidth);
    if (wristPx.y < chestLevel) {
      wristElevated = true;
    }
  }

  let poseX = 0, poseY = 0;
  if (faceLandmarks && faceLandmarks.length > 0) {
    const pose = estimateHeadPose(faceLandmarks, w, h);
    poseX = pose.poseX;
    poseY = pose.poseY;
  }

  return {
    neck_ratio: neckRatio,
    forward_lean_z: forwardLeanZ,
    shoulder_tilt_ratio: shoulderTiltRatio,
    head_tilt_ratio: headTiltRatio,
    hand_to_face_ratio: minHandToFace,
    pose_x: poseX,
    pose_y: poseY,
    wrist_elevated: wristElevated,
    
    // extra info for heuristics
    visibility: {
      nose: landmarks[0].visibility,
      l_shoulder: landmarks[11].visibility,
      r_shoulder: landmarks[12].visibility,
      l_wrist: landmarks[15].visibility,
      r_wrist: landmarks[16].visibility
    }
  };
}
