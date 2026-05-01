import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, useLocalParticipant, useRoomContext } from '@livekit/components-react';
import { RoomEvent, Track } from 'livekit-client';
import '@livekit/components-styles';
import { getSession } from '../lib/sessionStore';

function getCameraTrackItems(room, { includeLocal = true, participantFilter = () => true } = {}) {
  const participants = [
    ...(includeLocal ? [room.localParticipant] : []),
    ...Array.from(room.remoteParticipants.values()),
  ];

  return participants.flatMap((participant) => (
    Array.from(participant.trackPublications.values())
      .filter((publication) => publication.source === Track.Source.Camera && publication.track && participantFilter(participant))
      .map((publication) => ({
        id: publication.trackSid || publication.sid || `${participant.identity}-${publication.trackName || 'camera'}`,
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
          if (publication.source === Track.Source.Camera && !publication.track && typeof publication.setSubscribed === 'function') {
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

function StudentAiPipeline({ session, roomId, studentId, setError }) {
  const { cameraTrack } = useLocalParticipant();
  const workerRef = useRef(null);
  const wsRef = useRef(null);
  const hiddenVideoRef = useRef(null);
  const captureTimerRef = useRef(null);
  const lastScoreRef = useRef(100);

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

    let lastPoseResults = null;
    pose.onResults((results) => { lastPoseResults = results.poseLandmarks; });

    const captureFrame = async () => {
      if (disposed) return;
      if (hiddenVideoRef.current && hiddenVideoRef.current.readyState >= 2) {
        try {
          await pose.send({ image: hiddenVideoRef.current });
          await faceMesh.send({ image: hiddenVideoRef.current });
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

    faceMesh.onResults(async (results) => {
      if (workerRef.current) {
        let frameBitmap = null;
        if (hiddenVideoRef.current && hiddenVideoRef.current.readyState >= 2) {
          try {
            frameBitmap = await createImageBitmap(hiddenVideoRef.current);
          } catch {
            // Skip this frame if ImageBitmap creation fails.
          }
        }
        workerRef.current.postMessage({
          type: 'process_landmarks',
          landmarks: lastPoseResults, // Might be null, worker handles it
          faceLandmarks: results.multiFaceLandmarks?.[0],
          width: hiddenVideoRef.current?.videoWidth || 640,
          height: hiddenVideoRef.current?.videoHeight || 480,
          frame: frameBitmap
        }, frameBitmap ? [frameBitmap] : []);
      }
    });
    captureTimerRef.current = setTimeout(captureFrame, 200);

    worker.onmessage = (event) => {
      const payload = event.data;
      if (payload.type === 'score_update' && wsRef.current?.readyState === WebSocket.OPEN) {
        lastScoreRef.current = payload.averageScore;
        wsRef.current.send(JSON.stringify({
          token: session.session_token,
          average_score: payload.averageScore,
          status: payload.status,
          camera_on: payload.cameraOn ?? true,
          sampled_fps: payload.sampledFps ?? 5.0,
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
    ws.onerror = () => setError(`Score WebSocket failed (${wsBase})`);
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
  }, [roomId, session.session_token, studentId, setError]);

  useEffect(() => {
    const mediaStreamTrack = cameraTrack?.track?.mediaStreamTrack;
    if (hiddenVideoRef.current && mediaStreamTrack) {
      const stream = new MediaStream([mediaStreamTrack]);
      hiddenVideoRef.current.srcObject = stream;
      hiddenVideoRef.current.play().catch(() => undefined);
    }
  }, [cameraTrack]);

  return <video ref={hiddenVideoRef} muted playsInline style={{ display: 'none' }} />;
}

function StudentVideoGrid() {
  const tracks = useCameraTrackItems(useMemo(() => ({ includeLocal: true }), []));
  return (
    <section className="student-grid">
      {tracks.length === 0 ? (
        <p className="muted">Waiting for camera stream...</p>
      ) : (
        tracks.map(({ id, participant, track }) => (
          <article key={id} className="student-view">
            <header>{participant.identity}</header>
            <AttachedVideo track={track} />
          </article>
        ))
      )}
    </section>
  );
}

export default function StudentRoom() {
  const { roomId, studentId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [error, setError] = useState('');
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
    <main className="student-layout">
      <header className="panel">
        <h1>Classroom {roomId}</h1>
        <p className="muted">
          Connected as <strong>{studentId}</strong>. AI processing runs in the background.
        </p>
        {error && <p className="error-text">{error}</p>}
      </header>

      <LiveKitRoom
        token={session.livekit_token}
        serverUrl={session.livekit_url}
        connect
        video
        audio={false}
        onError={(err) => setError(`LiveKit connection failed: ${err?.message || 'Unknown error'}`)}
        onMediaDeviceFailure={(failure, kind) => {
          setError(`Cannot start ${kind || 'media device'}: ${failure || 'permission or device error'}`);
        }}
        className="room-shell"
      >
        <StudentAiPipeline session={session} roomId={roomId} studentId={studentId} setError={setError} />
        <StudentVideoGrid />
      </LiveKitRoom>
    </main>
  );
}

