import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, useRoomContext } from '@livekit/components-react';
import { RoomEvent, Track } from 'livekit-client';
import '@livekit/components-styles';
import { endRoom } from '../lib/api';
import { getSession } from '../lib/sessionStore';

const PAGE_SIZE = 16;

function formatUpdateAge(value, now) {
  if (!value) return 'No update';
  const updatedAt = new Date(value).getTime();
  if (!Number.isFinite(updatedAt)) return 'No update';
  const seconds = Math.max(0, Math.round((now - updatedAt) / 1000));
  if (seconds < 2) return 'Just now';
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.floor(seconds / 60)}m ago`;
}

function getStudentIdentity(identity = '') {
  return identity.startsWith('student-') ? identity.slice('student-'.length) : identity;
}

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

function TeacherStatusTable({ students }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="status-panel">
      <table className="status-table">
        <thead>
          <tr>
            <th>Student</th>
            <th>Status</th>
            <th style={{ width: '30%' }}>Focus Level</th>
            <th>Camera</th>
            <th>Last Update</th>
          </tr>
        </thead>
        <tbody>
          {students.length === 0 ? (
            <tr>
              <td colSpan="5" className="empty-cell">Waiting for students to join...</td>
            </tr>
          ) : (
            students.map((student) => {
              const scoreNum = parseFloat(student.score) || 0;
              const isCameraOff = student.camera === 'Off';
              const statusClass = student.status === 'Focused' ? 'success' : student.status === 'Absence' || student.status === 'Using Phone' ? 'error' : 'warning';

              return (
                <tr key={student.studentId}>
                  <td className="bold">{student.studentId}</td>
                  <td>
                    <span className={`status-badge ${statusClass}`}>
                      {student.status}
                    </span>
                  </td>
                  <td>
                    <div className="table-score-container">
                      <div className="table-score-bar">
                        <div 
                          className="bar-fill" 
                          style={{ 
                            width: `${scoreNum}%`,
                            backgroundColor: scoreNum > 70 ? 'var(--success)' : scoreNum > 40 ? 'var(--warning)' : 'var(--error)'
                          }}
                        ></div>
                      </div>
                      <span className="score-label">{Math.round(scoreNum)}%</span>
                    </div>
                  </td>
                  <td>
                    <span className={`camera-tag ${isCameraOff ? 'off' : 'on'}`}>
                      {student.camera}
                    </span>
                  </td>
                  <td className="muted text-sm">{formatUpdateAge(student.lastUpdate, now)}</td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </section>
  );
}

function TeacherVideoGrid({ snapshots }) {
  const [page, setPage] = useState(0);
  const cameraOptions = useMemo(() => ({
    includeLocal: false,
    participantFilter: (participant) => getStudentIdentity(participant.identity) !== participant.identity,
  }), []);
  const studentTracks = useCameraTrackItems(cameraOptions);
  const students = useMemo(() => {
    const byStudentId = new Map();

    for (const { participant } of studentTracks) {
      const studentId = getStudentIdentity(participant.identity);
      byStudentId.set(studentId, {
        studentId,
        status: 'Waiting...',
        score: 'No data',
        camera: 'On',
        lastUpdate: null,
      });
    }

    for (const [studentId, score] of Object.entries(snapshots)) {
      byStudentId.set(studentId, {
        studentId,
        status: score?.status || 'Waiting...',
        score: score ? `${Math.round(score.score)}/100` : 'No data',
        camera: score?.camera_on ? 'On' : 'Off',
        lastUpdate: score?.last_update || null,
      });
    }

    return Array.from(byStudentId.values()).sort((a, b) => a.studentId.localeCompare(b.studentId));
  }, [snapshots, studentTracks]);

  const totalPages = Math.max(1, Math.ceil(studentTracks.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages - 1);
  const pagedTracks = studentTracks.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);

  return (
    <>
      <TeacherStatusTable students={students} />

      <div className="grid-toolbar">
        <span>{studentTracks.length} camera streams</span>
        <div className="pager">
          <button onClick={() => setPage(Math.max(0, currentPage - 1))} disabled={currentPage === 0}>Prev</button>
          <span>{currentPage + 1} / {totalPages}</span>
          <button onClick={() => setPage(Math.min(totalPages - 1, currentPage + 1))} disabled={currentPage + 1 >= totalPages}>Next</button>
        </div>
      </div>

      <div className="teacher-grid">
        {pagedTracks.length === 0 ? (
          <p className="muted">Waiting for student camera streams...</p>
        ) : (
          pagedTracks.map(({ id, participant, track }) => {
            const studentId = getStudentIdentity(participant.identity);
            const score = snapshots[studentId] || null;
            const warning = !score || score.is_warning;
            return (
              <article key={id} className={`student-card ${warning ? 'warning' : ''}`}>
                <header>
                  <strong>{studentId}</strong>
                  <div className="card-score-mini">
                    <div className="mini-bar">
                      <div 
                        className="bar-fill" 
                        style={{ 
                          width: `${Math.round(score?.score || 0)}%`,
                          backgroundColor: (score?.score || 0) > 70 ? 'var(--success)' : (score?.score || 0) > 40 ? 'var(--warning)' : 'var(--error)'
                        }}
                      />
                    </div>
                    <span>{score ? Math.round(score.score) : 0}%</span>
                  </div>
                </header>
                <AttachedVideo track={track} />
                <footer>
                  <span className={`status-tag ${(score?.status || 'Waiting') === 'Focused' ? 'success' : 'warning'}`}>
                    {score?.status || 'Waiting...'}
                  </span>
                  {!score?.camera_on && <span className="tag error">Camera Off</span>}
                </footer>
              </article>
            );
          })
        )}
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
  const [instructorMode, setInstructorMode] = useState(false);

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

  function toggleInstructorMode() {
    setError('');
    setInstructorMode((current) => !current);
  }

  return (
    <main className="teacher-layout">
      <header className="teacher-header panel">
        <div>
          <h1>Teacher Dashboard</h1>
          <p className="muted">
            Room <strong>{roomId}</strong> | Invite: <code>{session.invitation_link}</code>
          </p>
          <p className="muted">
            Mode: <strong>{instructorMode ? 'Instructor (Camera + Mic ON)' : 'Viewer only (Camera + Mic OFF)'}</strong>
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button onClick={toggleInstructorMode}>
            {instructorMode ? 'Switch to viewer-only' : 'Join as instructor'}
          </button>
          <button onClick={handleEndClass}>End class and report</button>
        </div>
      </header>

      {error && <p className="error-text">{error}</p>}

      <LiveKitRoom
        key={`teacher-room-${roomId}-${instructorMode ? 'instructor' : 'viewer'}`}
        token={session.livekit_token}
        serverUrl={session.livekit_url}
        connect
        video={instructorMode}
        audio={instructorMode}
        onError={(err) => setError(`LiveKit connection failed: ${err?.message || 'Unknown error'}`)}
        onMediaDeviceFailure={(failure, kind) => {
          const issue = String(failure || 'permission or device error');
          setError(`Cannot start ${kind || 'media device'}: ${issue}`);
          setInstructorMode(false);
        }}
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

