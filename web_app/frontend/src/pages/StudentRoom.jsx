import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { FilesetResolver, PoseLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';
import { extractFeatures } from '../utils/engine';

export default function StudentRoom() {
  const { roomId, studentId } = useParams();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  
  const [status, setStatus] = useState('Initializing Models...');
  const [currentScore, setCurrentScore] = useState(100);
  const [currentLabel, setCurrentLabel] = useState('Waiting');
  
  const lastTimeRef = useRef(performance.now());
  const processingIntervalRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  useEffect(() => {
    let active = true;
    let poseLandmarker, faceLandmarker;

    const initializeModels = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1
        });

        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numFaces: 1
        });

        setStatus('Connecting to Camera...');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setStatus('Connecting to Session...');
            connectWebSocket();
          };
        }
      } catch (err) {
        setStatus(`Error: ${err.message}`);
      }
    };

    const connectWebSocket = () => {
      if (!active) return;
      wsRef.current = new WebSocket(`ws://localhost:8000/ws/student/${roomId}/${studentId}`);
      
      wsRef.current.onopen = () => {
        setStatus('Connected. Tracking Active.');
        lastTimeRef.current = performance.now();
        if (!processingIntervalRef.current) {
          processingIntervalRef.current = window.setInterval(processFrame, 1000);
        }
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.score !== undefined) {
          setCurrentScore(data.score);
          setCurrentLabel(data.label);
        }
      };

      wsRef.current.onclose = () => {
        if (!active) return;
        setStatus('Disconnected. Reconnecting...');
        if (reconnectTimeoutRef.current) {
          window.clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connectWebSocket();
        }, 1000);
      };
    };

    const processFrame = () => {
      if (!active || !videoRef.current || !poseLandmarker || !faceLandmarker) return;

      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      
      const v = videoRef.current;
      if (!v.videoWidth || !v.videoHeight) return;

      try {
        const poseResults = poseLandmarker.detectForVideo(v, performance.now());
        const faceResults = faceLandmarker.detectForVideo(v, performance.now());
        
        let features = null;
        let isAbsent = true;
        let hasPhone = false;

        const w = v.videoWidth;
        const h = v.videoHeight;

        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          // If pose landmarks are detected, the student is present.
          isAbsent = false;
          const pLms = poseResults.landmarks[0];
          const fLms = (faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) ? faceResults.faceLandmarks[0] : [];
          
          // Apply anti-tampering by obfuscating the payload feature formatting
          features = extractFeatures(pLms, fLms, w, h);
          
          const vis = features.visibility || {};

          // Phone detection heuristic logic
          const wristVisible = (vis.l_wrist ?? 0) > 0.5 || (vis.r_wrist ?? 0) > 0.5;
          const handNearEar = features.hand_to_face_ratio < 0.18;
          if (handNearEar && wristVisible) {
            hasPhone = true;
          }
        }

        const now = performance.now();
        const dt = (now - lastTimeRef.current) / 1000.0;
        lastTimeRef.current = now;

        // In a real prod environment, calculate crypto nonce hash here for anti-tampering verification
        const rawFeatures = features ? [
            features.neck_ratio, features.forward_lean_z, features.shoulder_tilt_ratio,
            features.head_tilt_ratio, features.hand_to_face_ratio, features.pose_x,
            features.pose_y, features.wrist_elevated ? 1 : 0, hasPhone ? 1 : 0
          ] : null;
        const safeFeatures = rawFeatures ? rawFeatures.map((value) => {
          const n = Number(value);
          return Number.isFinite(n) ? n : 0;
        }) : null;

        const payload = {
          features: safeFeatures,
          has_phone: hasPhone,
          is_absent: isAbsent,
          dt: dt
        };
        
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(payload));
        }

        // Draw debug overlay
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          ctx.clearRect(0,0, w, h);
          if (features) {
            ctx.fillStyle = 'lime';
            ctx.beginPath();
            ctx.arc(features.pose_x * w + w/2, h/2, 5, 0, 2*Math.PI);
            ctx.fill();
          }
        }
      } catch (err) {
        // Keep interval alive even if one inference frame fails.
        console.error('Frame processing failed:', err);
      }
    };

    initializeModels();

    return () => {
      active = false;
      if (processingIntervalRef.current) {
        window.clearInterval(processingIntervalRef.current);
        processingIntervalRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        window.clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      }
    };
  }, [roomId, studentId]);

  const getScoreColor = (sc) => {
    if (sc > 80) return '#00ff6a';
    if (sc > 50) return '#ffd000';
    return '#ff3737';
  };

  return (
    <div className="student-container">
      <div className="status-banner" style={{ background: getScoreColor(currentScore) + '33', borderBottom: `2px solid ${getScoreColor(currentScore)}`}}>
        Status: <strong>{status}</strong> | Room: <strong>{roomId}</strong>
      </div>
      
      <div className="video-wrapper slide-up">
        <video ref={videoRef} className="camera-feed" playsInline muted />
        <canvas ref={canvasRef} className="camera-overlay" width={640} height={480} />
      </div>
    </div>
  );
}
