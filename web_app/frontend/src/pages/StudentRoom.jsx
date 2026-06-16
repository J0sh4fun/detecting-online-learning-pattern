import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, useLocalParticipant, useRoomContext, ControlBar } from '@livekit/components-react';
import { RoomEvent, Track } from 'livekit-client';
import { Hand } from 'lucide-react';
import '@livekit/components-styles';
import { getSession } from '../lib/sessionStore';

function getCameraTrackItems(room, { includeLocal = true, participantFilter = () => true } = {}) {
  const participants = [
    ...(includeLocal ? [room.localParticipant] : []),
    ...Array.from(room.remoteParticipants.values()),
  ];

  return participants.flatMap((participant) => (
    Array.from(participant.trackPublications.values())
      .filter((publication) => (publication.source === Track.Source.Camera || publication.source === Track.Source.ScreenShare) && publication.track && participantFilter(participant))
      .map((publication) => ({
        id: publication.trackSid || publication.sid || `${participant.identity}-${publication.trackName || publication.source}`,
        participant,
        publication,
        track: publication.track,
      }))
  ));
}

function useCameraTrackItems(options) {
  const room = useRoomContext();
  const [items, setItems] = useState(() => getCameraTrackItems(room, options));

  useEffect(() => {
    const refresh = () => {
      for (const participant of room.remoteParticipants.values()) {
        for (const publication of participant.trackPublications.values()) {
          if ((publication.source === Track.Source.Camera || publication.source === Track.Source.ScreenShare) && !publication.track && typeof publication.setSubscribed === 'function') {
            publication.setSubscribed(true);
          }
        }
      }
      setItems(getCameraTrackItems(room, options));
    };

    refresh();
    room
      .on(RoomEvent.ParticipantConnected, refresh)
      .on(RoomEvent.ParticipantDisconnected, refresh)
      .on(RoomEvent.TrackPublished, refresh)
      .on(RoomEvent.TrackUnpublished, refresh)
      .on(RoomEvent.TrackSubscribed, refresh)
      .on(RoomEvent.TrackUnsubscribed, refresh)
      .on(RoomEvent.LocalTrackPublished, refresh)
      .on(RoomEvent.LocalTrackUnpublished, refresh)
      .on(RoomEvent.ConnectionStateChanged, refresh);

    return () => {
      room
        .off(RoomEvent.ParticipantConnected, refresh)
        .off(RoomEvent.ParticipantDisconnected, refresh)
        .off(RoomEvent.TrackPublished, refresh)
        .off(RoomEvent.TrackUnpublished, refresh)
        .off(RoomEvent.TrackSubscribed, refresh)
        .off(RoomEvent.TrackUnsubscribed, refresh)
        .off(RoomEvent.LocalTrackPublished, refresh)
        .off(RoomEvent.LocalTrackUnpublished, refresh)
        .off(RoomEvent.ConnectionStateChanged, refresh);
    };
  }, [room, options]);

  return items;
}

function AttachedVideo({ track }) {
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !track) return undefined;

    track.attach(video);
    video.play().catch(() => undefined);

    return () => {
      track.detach(video);
    };
  }, [track]);

  return <video ref={videoRef} muted playsInline />;
}

function StudentAiPipeline({ session, roomId, studentId, setError, setRoomClosedData }) {
  const { cameraTrack, isCameraEnabled } = useLocalParticipant();
  const workerRef = useRef(null);
  const wsRef = useRef(null);
  const hiddenVideoRef = useRef(null);
  const captureTimerRef = useRef(null);
  const lastScoreRef = useRef(100);
  const lastStatusRef = useRef('Focused');
  const cameraEnabledRef   = useRef(true);   // Optimistic: assume camera on (LiveKit initialises asynchronously)
  const cameraWasActiveRef  = useRef(false);  // Becomes true once camera confirmed active

  const sendScoreUpdate = useCallback((cameraOn = cameraEnabledRef.current) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({
      token: session.session_token,
      average_score: cameraOn ? lastScoreRef.current : 0,
      status: cameraOn ? lastStatusRef.current : 'Camera Off',
      camera_on: cameraOn,
      sampled_fps: cameraOn ? 5.0 : 0.0,
      sample_count: 1,
      client_sent_at: Date.now() / 1000,
    }));
  }, [session.session_token]);

  useEffect(() => {
    cameraEnabledRef.current = isCameraEnabled;

    if (isCameraEnabled) {
      // Camera is confirmed active – mark it and broadcast immediately
      cameraWasActiveRef.current = true;
      sendScoreUpdate(true);
    } else if (cameraWasActiveRef.current) {
      // Camera was active before and user explicitly turned it off – broadcast the change
      sendScoreUpdate(false);
    }
    // If camera was never confirmed active (still initialising), do nothing:
    // ws.onopen will send the correct state once we know it.
  }, [isCameraEnabled, sendScoreUpdate]);

  useEffect(() => {
    let disposed = false;
    const worker = new Worker(new URL('../workers/aiWorker.js', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    const pose = new window.Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({ modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5 });

    const faceMesh = new window.FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5 });

    // lastPoseResults and lastFaceResults are updated by their respective
    // model callbacks. Both are captured AFTER both awaits in captureFrame
    // complete, which guarantees pose.onResults has had a chance to fire
    // before we dispatch to the worker — eliminating the race condition.
    let lastPoseResults = null;
    let lastFaceResults = null;
    pose.onResults((results) => { lastPoseResults = results.poseLandmarks; });
    faceMesh.onResults((results) => { lastFaceResults = results.multiFaceLandmarks?.[0] ?? null; });

    const captureFrame = async () => {
      if (disposed) return;
      if (hiddenVideoRef.current && hiddenVideoRef.current.readyState >= 2) {
        try {
          // Send to both models sequentially on the SAME frame.
          // After both awaits return, both onResults callbacks above have
          // already been called and lastPoseResults / lastFaceResults are
          // up to date for this frame. Only then do we dispatch to the worker.
          await pose.send({ image: hiddenVideoRef.current });
          await faceMesh.send({ image: hiddenVideoRef.current });

          if (workerRef.current) {
            let frameBitmap = null;
            try {
              frameBitmap = await createImageBitmap(hiddenVideoRef.current);
            } catch {
              // Skip ImageBitmap creation on transient errors.
            }
            workerRef.current.postMessage({
              type: 'process_landmarks',
              landmarks: lastPoseResults,
              faceLandmarks: lastFaceResults,
              width: hiddenVideoRef.current?.videoWidth || 640,
              height: hiddenVideoRef.current?.videoHeight || 480,
              frame: frameBitmap,
            }, frameBitmap ? [frameBitmap] : []);
          }
        } catch {
          // Keep loop alive on transient frame processing errors.
        }
      }
      captureTimerRef.current = setTimeout(captureFrame, 200); // 5 FPS
    };

    worker.postMessage({
      type: 'init',
      modelUrl: import.meta.env.VITE_POSTURE_MODEL_URL || '/models/best_posture_model.onnx',
      yoloUrl: import.meta.env.VITE_PHONE_MODEL_URL || '/models/yolo26s.onnx',
      flushIntervalMs: 2000,
      sampledFps: 5,
      yoloCheckInterval: 2,
      maxHistory: 6,
    });

    captureTimerRef.current = setTimeout(captureFrame, 200);

    worker.onmessage = (event) => {
      const payload = event.data;
      if (payload.type === 'score_update' && wsRef.current?.readyState === WebSocket.OPEN) {
        lastScoreRef.current = payload.averageScore;
        lastStatusRef.current = payload.status;
        const cameraOn = cameraEnabledRef.current && (payload.cameraOn ?? true);
        wsRef.current.send(JSON.stringify({
          token: session.session_token,
          average_score: cameraOn ? payload.averageScore : 0,
          status: cameraOn ? payload.status : 'Camera Off',
          camera_on: cameraOn,
          sampled_fps: cameraOn ? (payload.sampledFps ?? 5.0) : 0.0,
          sample_count: payload.sampleCount ?? 1,
          client_sent_at: Date.now() / 1000,
        }));
      }

      if (payload.type === 'worker_error') {
        setError(`AI worker: ${payload.message}`);
      }
    };

    const wsBase = (import.meta.env.VITE_API_WS_BASE || 'ws://localhost:8000').replace(/\/$/, '');
    const ws = new WebSocket(`${wsBase}/ws/student/${roomId}/${encodeURIComponent(studentId)}`);
    wsRef.current = ws;
    ws.onopen = () => {
      // Send initial state; if camera hasn't been confirmed active yet, use optimistic true.
      // The isCameraEnabled effect will correct this as soon as LiveKit confirms camera state.
      sendScoreUpdate(cameraWasActiveRef.current ? cameraEnabledRef.current : true);
    };
    ws.onerror = () => setError(`Score WebSocket failed (${wsBase})`);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'room_closed') {
          setRoomClosedData({ teacherName: data.teacher_id || 'The teacher' });
        }
      } catch {
        // Ignore parse errors for now
      }
    };
    ws.onclose = () => {
      if (!disposed) setError(`Score WebSocket closed (${wsBase})`);
    };

    return () => {
      disposed = true;
      if (captureTimerRef.current) clearTimeout(captureTimerRef.current);
      if (wsRef.current) wsRef.current.close();
      worker.terminate();
      pose.close();
      faceMesh.close();
    };
  }, [roomId, sendScoreUpdate, session.session_token, studentId, setError, setRoomClosedData]);

  const mediaStreamTrack = cameraTrack?.track?.mediaStreamTrack;

  useEffect(() => {
    if (hiddenVideoRef.current && mediaStreamTrack) {
      const stream = new MediaStream([mediaStreamTrack]);
      hiddenVideoRef.current.srcObject = stream;
      // Chạy video tĩnh tiếng
      hiddenVideoRef.current.play().catch(() => undefined);
    } else if (hiddenVideoRef.current) {
      // Clear video khi tắt cam
      hiddenVideoRef.current.srcObject = null;
    }
  }, [mediaStreamTrack]);

  return (
    <video 
      ref={hiddenVideoRef} 
      muted 
      playsInline 
      style={{ 
        position: 'absolute', 
        width: '1px', 
        height: '1px', 
        opacity: 0, 
        pointerEvents: 'none' 
      }} 
    />
  );
}

function RaiseHandButton() {
  const { localParticipant } = useLocalParticipant();
  let metadata = {};
  if (localParticipant?.metadata) {
    try {
      metadata = JSON.parse(localParticipant.metadata);
    } catch {
      metadata = {};
    }
  }

  const isRaised = metadata.hand_raised === true;

  const toggleHand = () => {
    if (!localParticipant) return;
    const newMetadata = JSON.stringify({ ...metadata, hand_raised: !isRaised });
    localParticipant.setMetadata(newMetadata);
  };

  return (
    <button 
      className={`lk-button raise-hand-btn ${isRaised ? 'active' : ''}`} 
      onClick={toggleHand}
      title={isRaised ? 'Lower Hand' : 'Raise Hand'}
    >
      <Hand size={18} />
      <span>{isRaised ? 'Lower Hand' : 'Raise Hand'}</span>
    </button>
  );
}

function StudentFocusLayout() {
  const tracks = useCameraTrackItems(useMemo(() => ({ includeLocal: true }), []));

  const localTrack = tracks.find(t => t.participant.isLocal && t.publication.source === Track.Source.Camera);
  
  const teacherScreenshare = tracks.find(t => t.participant.identity.startsWith('teacher-') && t.publication.source === Track.Source.ScreenShare);
  const studentScreenshare = tracks.find(t => !t.participant.identity.startsWith('teacher-') && t.publication.source === Track.Source.ScreenShare);
  const teacherCamera = tracks.find(t => t.participant.identity.startsWith('teacher-') && t.publication.source === Track.Source.Camera);

  const mainTrack = teacherScreenshare || studentScreenshare || teacherCamera;

  return (
    <section className="focus-layout">
      {mainTrack ? (
        <article key={mainTrack.id} className="main-view">
          <AttachedVideo track={mainTrack.track} />
          <div className="view-overlay">
            <span>{mainTrack.participant.identity}{mainTrack.publication.source === Track.Source.ScreenShare ? "'s screen" : ""}</span>
          </div>
        </article>
      ) : (
        <div className="empty-main screen-center">
          <p className="muted">Waiting for teacher or presentation...</p>
        </div>
      )}

      {localTrack && (
        <article key={localTrack.id} className="pip-view">
          <AttachedVideo track={localTrack.track} />
        </article>
      )}
    </section>
  );
}

export default function StudentRoom() {
  const { roomId, studentId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [error, setError] = useState('');
  const [roomClosedData, setRoomClosedData] = useState(null);
  const session = useMemo(() => location.state?.session || getSession('student'), [location.state?.session]);

  if (!session) {
    return (
      <main className="screen-center">
        <p>Session expired. Please join the classroom again.</p>
        <button onClick={() => navigate('/')}>Back home</button>
      </main>
    );
  }

  return (
    <main className="student-room-fullscreen">
      <div className="room-info-chip">
        <span>Room <strong>{roomId}</strong></span>
        <span className="muted">·</span>
        <span className="muted">{studentId}</span>
      </div>
      {error && <p className="error-text floating-error">{error}</p>}

      <LiveKitRoom
        token={session.livekit_token}
        serverUrl={session.livekit_url}
        connect
        video
        audio={true}
        onDisconnected={() => navigate('/')}
        onError={(err) => setError(`LiveKit connection failed: ${err?.message || 'Unknown error'}`)}
        onMediaDeviceFailure={(failure, kind) => {
          setError(`Cannot start ${kind || 'media device'}: ${failure || 'permission or device error'}`);
        }}
        className="room-shell"
      >
        <StudentAiPipeline session={session} roomId={roomId} studentId={studentId} setError={setError} setRoomClosedData={setRoomClosedData} />
        <StudentFocusLayout />
        <div className="room-controls">
          <ControlBar variation="minimal" controls={{ microphone: true, camera: true, screenShare: true, chat: false }} />
          <RaiseHandButton />
        </div>
      </LiveKitRoom>

      {roomClosedData && (
        <div className="room-closed-overlay screen-center">
          <div className="panel text-center animate-in">
            <h2 className="error-text">Room Closed</h2>
            <p><strong>{roomClosedData.teacherName}</strong> has closed the classroom.</p>
            <p className="muted">The session has ended and scoring has stopped.</p>
            <button className="mt-4" onClick={() => navigate('/')}>
              Return to main screen
            </button>
          </div>
        </div>
      )}
    </main>
  );
}

