import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, useLocalParticipant, useRoomContext } from '@livekit/components-react';
import { RoomEvent, Track } from 'livekit-client';
import '@livekit/components-styles';
import { scoreFrame, verifyFrame } from '../lib/api';
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

function average(values) {
  if (!values.length) return 100;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function StudentAiPipeline({ session, roomId, studentId, setError }) {
  const { cameraTrack } = useLocalParticipant();
  const workerRef = useRef(null);
  const wsRef = useRef(null);
  const hiddenVideoRef = useRef(null);
  const captureTimerRef = useRef(null);
  const lastScoreRef = useRef(100);
  const scoreHistoryRef = useRef([]);
  const lastSentAtRef = useRef(0);

  useEffect(() => {
    let disposed = false;
    const worker = new Worker(new URL('../workers/aiWorker.js', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.postMessage({
      type: 'init',
      targetFps: 30,
      flushIntervalMs: 250,
    });

    worker.onmessage = async (event) => {
      const payload = event.data;
      if (payload.type === 'score_frame') {
        try {
          const result = await scoreFrame({
            token: session.session_token,
            roomCode: roomId,
            studentId,
            frameBase64: payload.frameBase64,
          });

          scoreHistoryRef.current = [...scoreHistoryRef.current, result.score].slice(-10);
          lastScoreRef.current = average(scoreHistoryRef.current);

          const now = Date.now();
          if (wsRef.current?.readyState === WebSocket.OPEN && now - lastSentAtRef.current >= 750) {
            wsRef.current.send(JSON.stringify({
              token: session.session_token,
              average_score: lastScoreRef.current,
              status: result.status,
              camera_on: payload.cameraOn,
              sampled_fps: payload.fps,
              sample_count: scoreHistoryRef.current.length,
              client_sent_at: now / 1000,
            }));
            lastSentAtRef.current = now;
          }
        } catch (err) {
          setError(`Live scoring failed: ${err.message}`);
        }
      }

      if (payload.type === 'verify_frame') {
        try {
          await verifyFrame({
            token: session.session_token,
            roomCode: roomId,
            studentId,
            clientScore: lastScoreRef.current,
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
    const mediaStreamTrack = cameraTrack?.track?.mediaStreamTrack;
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
    }, 33);
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

