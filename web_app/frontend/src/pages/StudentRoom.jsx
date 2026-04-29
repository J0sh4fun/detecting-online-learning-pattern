import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, TrackLoop, VideoTrack, useLocalParticipant, useTracks } from '@livekit/components-react';
import { Track } from 'livekit-client';
import '@livekit/components-styles';
import { verifyFrame } from '../lib/api';
import { getSession } from '../lib/sessionStore';

function StudentAiPipeline({ session, roomId, studentId, setError }) {
  const { localParticipant } = useLocalParticipant();
  const workerRef = useRef(null);
  const wsRef = useRef(null);
  const hiddenVideoRef = useRef(null);
  const captureTimerRef = useRef(null);

  useEffect(() => {
    let disposed = false;
    const worker = new Worker(new URL('../workers/aiWorker.js', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.postMessage({
      type: 'init',
      modelUrl: import.meta.env.VITE_POSTURE_MODEL_URL || '/models/best_posture_model.onnx',
      targetFps: 1,
      flushIntervalMs: 4000,
    });

    worker.onmessage = async (event) => {
      const payload = event.data;
      if (payload.type === 'aggregate' && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          token: session.session_token,
          average_score: payload.averageScore,
          status: payload.status,
          camera_on: payload.cameraOn,
          sampled_fps: payload.fps,
          sample_count: payload.sampleCount,
          client_sent_at: Date.now() / 1000,
        }));
      }

      if (payload.type === 'verify_frame') {
        try {
          await verifyFrame({
            token: session.session_token,
            roomCode: roomId,
            studentId,
            clientScore: payload.clientScore,
            frameBase64: payload.frameBase64,
          });
        } catch (err) {
          setError(`Verification failed: ${err.message}`);
        }
      }
    };

    const wsBase = (import.meta.env.VITE_API_WS_BASE || 'ws://localhost:8000').replace(/\/$/, '');
    wsRef.current = new WebSocket(`${wsBase}/ws/student/${roomId}/${encodeURIComponent(studentId)}`);
    wsRef.current.onerror = () => {
      if (!disposed) {
        setError(`Score WebSocket disconnected (${wsBase})`);
      }
    };
    wsRef.current.onclose = () => {
      if (!disposed) {
        setError(`Score stream closed (${wsBase})`);
      }
    };

    return () => {
      disposed = true;
      if (captureTimerRef.current) {
        window.clearInterval(captureTimerRef.current);
      }
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        wsRef.current.close();
      }
      worker.terminate();
    };
  }, [roomId, session.session_token, setError, studentId]);

  useEffect(() => {
    const publication = localParticipant.getTrackPublication(Track.Source.Camera);
    const mediaStreamTrack = publication?.track?.mediaStreamTrack;
    const video = hiddenVideoRef.current;
    if (!video || !mediaStreamTrack) return;

    const stream = new MediaStream([mediaStreamTrack]);
    video.srcObject = stream;
    video.play().catch(() => undefined);

    if (captureTimerRef.current) window.clearInterval(captureTimerRef.current);
    captureTimerRef.current = window.setInterval(async () => {
      if (!workerRef.current || !video.videoWidth || !video.videoHeight) {
        return;
      }
      const frame = await createImageBitmap(video);
      workerRef.current.postMessage({ type: 'frame', frame, timestamp: Date.now() }, [frame]);
    }, 1000);
  }, [localParticipant]);

  return <video ref={hiddenVideoRef} muted playsInline style={{ display: 'none' }} />;
}

function StudentVideoGrid() {
  const tracks = useTracks([{ source: Track.Source.Camera, withPlaceholder: true }], { onlySubscribed: false });
  return (
    <section className="student-grid">
      <TrackLoop tracks={tracks}>
        {(trackRef) => (
          <article key={`${trackRef.participant.identity}-${trackRef.source || 'camera'}`} className="student-view">
            <header>{trackRef.participant.identity}</header>
            <VideoTrack trackRef={trackRef} />
          </article>
        )}
      </TrackLoop>
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
        onError={() => setError(`LiveKit connection failed: ${session.livekit_url}`)}
        className="room-shell"
      >
        <StudentAiPipeline session={session} roomId={roomId} studentId={studentId} setError={setError} />
        <StudentVideoGrid />
      </LiveKitRoom>
    </main>
  );
}

