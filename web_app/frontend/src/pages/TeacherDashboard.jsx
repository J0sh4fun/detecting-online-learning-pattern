import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, TrackLoop, VideoTrack, useTracks } from '@livekit/components-react';
import { Track } from 'livekit-client';
import '@livekit/components-styles';
import { endRoom } from '../lib/api';
import { getSession } from '../lib/sessionStore';

const PAGE_SIZE = 16;

function getStudentIdentity(identity = '') {
  return identity.startsWith('student-') ? identity.slice('student-'.length) : identity;
}

function TeacherVideoGrid({ snapshots }) {
  const [page, setPage] = useState(0);
  const tracks = useTracks([{ source: Track.Source.Camera, withPlaceholder: true }], { onlySubscribed: false });
  const studentTracks = useMemo(
    () => tracks.filter((trackRef) => getStudentIdentity(trackRef.participant.identity) !== trackRef.participant.identity),
    [tracks]
  );

  const totalPages = Math.max(1, Math.ceil(studentTracks.length / PAGE_SIZE));
  const pagedTracks = studentTracks.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  useEffect(() => {
    if (page >= totalPages) {
      setPage(Math.max(0, totalPages - 1));
    }
  }, [page, totalPages]);

  return (
    <>
      <div className="grid-toolbar">
        <span>{studentTracks.length} camera streams</span>
        <div className="pager">
          <button onClick={() => setPage((prev) => Math.max(0, prev - 1))} disabled={page === 0}>Prev</button>
          <span>{page + 1} / {totalPages}</span>
          <button onClick={() => setPage((prev) => Math.min(totalPages - 1, prev + 1))} disabled={page + 1 >= totalPages}>Next</button>
        </div>
      </div>

      <div className="teacher-grid">
        <TrackLoop tracks={pagedTracks}>
          {(trackRef) => {
            const studentId = getStudentIdentity(trackRef.participant.identity);
            const score = snapshots[studentId] || null;
            const warning = !score || score.is_warning;
            return (
              <article key={trackRef.participant.identity} className={`student-card ${warning ? 'warning' : ''}`}>
                <header>
                  <strong>{studentId}</strong>
                  <span>{score ? `${Math.round(score.score)}/100` : 'No data'}</span>
                </header>
                <VideoTrack trackRef={trackRef} />
                <footer>
                  <span>{score?.status || 'Waiting...'}</span>
                  {!score?.camera_on && <span className="tag">Camera Off</span>}
                </footer>
              </article>
            );
          }}
        </TrackLoop>
      </div>
    </>
  );
}

export default function TeacherDashboard() {
  const { roomId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [session] = useState(() => location.state?.session || getSession('teacher'));
  const [scores, setScores] = useState({});
  const [report, setReport] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!session?.session_token || !roomId) return;
    let disposed = false;
    const wsBase = (import.meta.env.VITE_API_WS_BASE || 'ws://localhost:8000').replace(/\/$/, '');
    const ws = new WebSocket(`${wsBase}/ws/teacher/${roomId}?token=${encodeURIComponent(session.session_token)}`);

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type !== 'scores_snapshot') return;
      const next = {};
      for (const student of payload.students) {
        next[student.student_id] = student;
      }
      setScores(next);
    };
    ws.onerror = () => {
      if (!disposed) {
        setError(`Cannot subscribe to concentration stream at ${wsBase}.`);
      }
    };
    ws.onclose = () => {
      if (!disposed) {
        setError(`Concentration stream closed (${wsBase}).`);
      }
    };
    return () => {
      disposed = true;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [roomId, session?.session_token]);

  if (!session) {
    return (
      <main className="screen-center">
        <p>Session expired. Please create a new classroom.</p>
        <button onClick={() => navigate('/')}>Back home</button>
      </main>
    );
  }

  async function handleEndClass() {
    try {
      const data = await endRoom({ roomCode: roomId, token: session.session_token });
      setReport(data);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <main className="teacher-layout">
      <header className="teacher-header panel">
        <div>
          <h1>Teacher Dashboard</h1>
          <p className="muted">
            Room <strong>{roomId}</strong> | Invite: <code>{session.invitation_link}</code>
          </p>
        </div>
        <button onClick={handleEndClass}>End class and report</button>
      </header>

      {error && <p className="error-text">{error}</p>}

      <LiveKitRoom
        token={session.livekit_token}
        serverUrl={session.livekit_url}
        connect
        video
        audio={false}
        onError={() => setError(`LiveKit connection failed: ${session.livekit_url}`)}
        className="room-shell"
      >
        <TeacherVideoGrid snapshots={scores} />
      </LiveKitRoom>

      {report && (
        <section className="panel report-panel">
          <h2>Class Report</h2>
          <p>Class average score: <strong>{report.class_average_score}</strong></p>
          <div className="report-list">
            {report.students.map((student) => (
              <article key={student.student_id}>
                <strong>{student.student_id}</strong>
                <span>{student.average_score}</span>
              </article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

