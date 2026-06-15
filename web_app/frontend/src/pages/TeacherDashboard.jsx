import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, useRoomContext, ControlBar } from '@livekit/components-react';
import { RoomEvent, Track } from 'livekit-client';
import { Hand, PanelRightClose, PanelRightOpen } from 'lucide-react';
import '@livekit/components-styles';
import { endRoom } from '../lib/api';
import { getSession } from '../lib/sessionStore';
import ReportView from './ReportView';
import { VideoTrack } from '@livekit/components-react';

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
      .on(RoomEvent.ConnectionStateChanged, refresh)
      .on(RoomEvent.ParticipantMetadataChanged, refresh);

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
        .off(RoomEvent.ConnectionStateChanged, refresh)
        .off(RoomEvent.ParticipantMetadataChanged, refresh);
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

function getStatusClass(status) {
  if (status === 'Focused') return 'success';
  if (status === 'Absence' || status === 'Using Phone' || status === 'Camera Off') return 'error';
  return 'warning';
}

function CameraOffTile({ label }) {
  return (
    <div className="camera-off-tile" aria-label={label || 'Camera Off'}>
      <div className="camera-off-cross" aria-hidden="true" />
      {label && <span>{label}</span>}
    </div>
  );
}

function getZoomGridClass(count) {
  if (count <= 1) return 'one';
  if (count === 2) return 'two';
  if (count <= 4) return 'four';
  if (count <= 6) return 'six';
  if (count <= 9) return 'nine';
  return 'many';
}

function TeacherStatusTable({ students }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="status-panel status-sidebar-panel">
      <div className="status-panel-heading">
        <h3>Student Status</h3>
        <span>{students.length}</span>
      </div>

      {students.length === 0 ? (
        <p className="empty-cell">Waiting for students to join...</p>
      ) : (
        <div className="status-card-list">
          {students.map((student) => {
            const scoreNum = parseFloat(student.score) || 0;
            const isCameraOff = student.camera === 'Off';
            const statusClass = getStatusClass(student.status);

            return (
              <article className="status-student-card" key={student.studentId}>
                <header>
                  <strong>{student.studentId}</strong>
                  {student.handRaised && <Hand size={14} className="text-warning ml-2" />}
                </header>

                <div className="status-detail-grid">
                  <span className="status-detail-label">Status</span>
                  <span className={`status-badge ${statusClass}`}>{student.status}</span>

                  <span className="status-detail-label">Focus</span>
                  <div className="table-score-container compact">
                    <div className="table-score-bar">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${scoreNum}%`,
                          backgroundColor: scoreNum > 70 ? 'var(--success)' : scoreNum > 40 ? 'var(--warning)' : 'var(--error)'
                        }}
                      />
                    </div>
                    <span className="score-label">{Math.round(scoreNum)}%</span>
                  </div>

                  <span className="status-detail-label">Camera</span>
                  <span className={`camera-tag ${isCameraOff ? 'off' : 'on'}`}>{student.camera}</span>

                  <span className="status-detail-label">Updated</span>
                  <span className="muted text-sm">{formatUpdateAge(student.lastUpdate, now)}</span>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

function TeacherVideoGrid({ snapshots, sidebarOpen }) {
  const [focusedStudentId, setFocusedStudentId] = useState(null);
  const [swapPip, setSwapPip] = useState(false);
  const cameraOptions = useMemo(() => ({
    includeLocal: false,
    participantFilter: (participant) => getStudentIdentity(participant.identity) !== participant.identity,
  }), []);
  const studentTracks = useCameraTrackItems(cameraOptions);
  const students = useMemo(() => {
    const groups = new Map();

    for (const item of studentTracks) {
      const studentId = getStudentIdentity(item.participant.identity);
      if (!groups.has(studentId)) {
        groups.set(studentId, { studentId, participant: item.participant, cameraTrack: null, screenTrack: null, handRaised: false });
      }
      const group = groups.get(studentId);
      if (item.publication.source === Track.Source.Camera) group.cameraTrack = item;
      if (item.publication.source === Track.Source.ScreenShare) group.screenTrack = item;
      
      try {
        const metadata = JSON.parse(item.participant.metadata || '{}');
        group.handRaised = metadata.hand_raised === true;
      } catch {
        group.handRaised = false;
      }
    }

    for (const studentId of Object.keys(snapshots)) {
      if (!groups.has(studentId)) {
        groups.set(studentId, { studentId, participant: null, cameraTrack: null, screenTrack: null, handRaised: false });
      }
    }

    const array = Array.from(groups.values());
    
    return array.map(g => {
      const studentId = g.studentId;
      const score = snapshots[studentId];
      return {
        ...g,
        status: score?.status || 'Waiting...',
        score: score ? `${Math.round(score.score)}/100` : 'No data',
        scoreNum: score?.score || 0,
        camera: score ? (score.camera_on ? 'On' : 'Off') : (g.cameraTrack ? 'On' : 'Unknown'),
        lastUpdate: score?.last_update || null,
        isWarning: !score || score.is_warning
      };
    }).sort((a, b) => {
      if (a.handRaised !== b.handRaised) return a.handRaised ? -1 : 1;
      return a.studentId.localeCompare(b.studentId);
    });
  }, [snapshots, studentTracks]);

  const focusedStudent = focusedStudentId ? students.find(s => s.studentId === focusedStudentId) : null;

  if (focusedStudent) {
    const student = focusedStudent;
    const hasScreen = !!student.screenTrack;
    let mainTrack = swapPip ? student.cameraTrack : (hasScreen ? student.screenTrack : student.cameraTrack);
    let pipTrack = swapPip ? student.screenTrack : (hasScreen ? student.cameraTrack : null);
    if (student.camera === 'Off' && mainTrack === student.cameraTrack) mainTrack = null;
    if (student.camera === 'Off' && pipTrack === student.cameraTrack) pipTrack = null;
    const returnToGrid = () => {
      setFocusedStudentId(null);
      setSwapPip(false);
    };

    return (
      <div className={`focused-container teacher-focused-container ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <section className="focus-layout teacher-focus-layout">
          {mainTrack ? (
            <article className="main-view" onDoubleClick={returnToGrid}>
              <AttachedVideo track={mainTrack.track} />
              <div className="view-overlay">
                <span>{student.studentId} {mainTrack.publication.source === Track.Source.ScreenShare ? "'s screen" : ""}</span>
              </div>
            </article>
          ) : student.camera === 'Off' ? (
            <CameraOffTile label={`${student.studentId} camera off`} />
          ) : (
            <div className="empty-main screen-center">
              <p className="muted">No video available for {student.studentId}</p>
            </div>
          )}

          {pipTrack && (
            <article
              className="pip-view pip-left"
              onClick={() => setSwapPip(!swapPip)}
              onDoubleClick={(event) => {
                event.stopPropagation();
              }}
            >
              <AttachedVideo track={pipTrack.track} />
            </article>
          )}
        </section>
      </div>
    );
  }

  return (
    <div className={`grid-container teacher-grid-container ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <div className={`teacher-grid meeting-grid ${getZoomGridClass(students.length)}`}>
        {students.length === 0 ? (
          <div className="empty-main screen-center">
            <p className="muted">Waiting for students...</p>
          </div>
        ) : (
          students.map((student) => {
            const displayTrack = student.screenTrack || student.cameraTrack;
            
            return (
              <article
                key={student.studentId}
                className={`student-card meeting-tile ${student.handRaised ? 'hand-raised' : ''}`}
                onClick={() => { setFocusedStudentId(student.studentId); setSwapPip(false); }}
              >
                {displayTrack ? (
                  <VideoTrack trackRef={{ participant: student.participant, publication: displayTrack.publication, source: displayTrack.publication.source }} />
                ) : student.camera === 'Off' ? (
                  <CameraOffTile />
                ) : (
                  <div className="video-placeholder" />
                )}
                <div className="tile-nameplate">
                  {student.handRaised && <Hand size={16} />}
                  <span>{student.studentId}</span>
                </div>
              </article>
            );
          })
        )}
      </div>
    </div>
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
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
    <main className="teacher-room-fullscreen">
      <div className="room-info-chip teacher-room-info">
        <span>Room <strong>{roomId}</strong></span>
        <span className="muted">|</span>
        <code>{session.invitation_link}</code>
      </div>
      <button className="teacher-end-button" onClick={handleEndClass}>End class and report</button>
      {error && <p className="error-text floating-error">{error}</p>}

      <LiveKitRoom
        key={`teacher-room-${roomId}`}
        token={session.livekit_token}
        serverUrl={session.livekit_url}
        connect
        video={true}
        audio={true}
        onError={(err) => setError(`LiveKit connection failed: ${err?.message || 'Unknown error'}`)}
        onMediaDeviceFailure={(failure, kind) => {
          const issue = String(failure || 'permission or device error');
          setError(`Cannot start ${kind || 'media device'}: ${issue}`);
        }}
        className="room-shell"
      >
        <div className="teacher-workspace">
          <TeacherVideoGrid snapshots={scores} sidebarOpen={sidebarOpen} />
          
          <aside className={`teacher-sidebar ${sidebarOpen ? 'open' : ''}`}>
            <TeacherStatusTable students={Object.values(scores).map(score => ({
               studentId: score.student_id,
               status: score.status,
               score: score.score,
               camera: score.camera_on ? 'On' : 'Off',
               lastUpdate: score.last_update
            }))} />
          </aside>
        </div>

        <div className="room-controls">
          <ControlBar variation="minimal" controls={{ microphone: true, camera: true, screenShare: true, chat: false }} />
          <div className="divider" />
          <button 
            className={`lk-button toggle-sidebar-btn ${sidebarOpen ? 'active' : ''}`} 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title="Toggle Status Sidebar"
          >
            {sidebarOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            <span>Status</span>
          </button>
        </div>
      </LiveKitRoom>

      {report && (
        <div className="report-overlay">
          <ReportView 
            report={report} 
            onBack={() => {
              setReport(null);
              navigate('/');
            }} 
          />
        </div>
      )}
    </main>
  );
}

