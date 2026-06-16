import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { LiveKitRoom, ControlBar, useTracks, VideoTrack } from '@livekit/components-react';
import { Track } from 'livekit-client';
import { Hand, PanelRightClose, PanelRightOpen } from 'lucide-react';
import '@livekit/components-styles';
import { endRoom } from '../lib/api';
import { getSession } from '../lib/sessionStore';
import ReportView from './ReportView';

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

// ─── Grid layout constants ────────────────────────────────────────────────────
// Clearances to leave space for the fixed top-bar and bottom toolbar
const GRID_PT = 60;  // top: room-info-chip
const GRID_PB = 84;  // bottom: room-controls bar
const GRID_PH = 10;  // horizontal padding each side
const GRID_GAP = 8;  // gap between tiles (px)
// Minimum tile dimensions (below this the tile becomes unreadable)
const MIN_TILE_W = 300;
const MIN_TILE_H = Math.round(MIN_TILE_W * 9 / 16); // 169

/**
 * Watches the grid container element and computes the optimal tile layout:
 * - equal-size tiles with 16:9 aspect ratio
 * - tiles grow to fill available space; shrink to make room for more
 * - once tiles hit MIN_TILE_W, pagination kicks in
 */
function useGridLayout(containerRef, totalStudents, page) {
  const [layout, setLayout] = useState(null);

  const compute = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;

    const { width, height } = el.getBoundingClientRect();
    const availW = Math.max(0, width  - GRID_PH * 2);
    const availH = Math.max(0, height - GRID_PT - GRID_PB);

    // Maximum cols / rows before tiles become too small
    const maxCols = Math.max(1, Math.floor((availW + GRID_GAP) / (MIN_TILE_W + GRID_GAP)));
    const maxRows = Math.max(1, Math.floor((availH + GRID_GAP) / (MIN_TILE_H + GRID_GAP)));
    const pageSize = maxCols * maxRows;

    const startIdx  = page * pageSize;
    const pageTileCount = Math.min(pageSize, Math.max(0, totalStudents - startIdx));
    const totalPages = pageSize > 0 ? Math.ceil(totalStudents / pageSize) : 1;

    if (pageTileCount === 0) {
      setLayout({ cols: 1, rows: 1, tileW: Math.floor(availW), tileH: Math.floor(availW * 9 / 16), pageSize, totalPages, pageTileCount: 0 });
      return;
    }

    // Iterate every valid column count; pick the arrangement with the largest tiles
    let bestCols = 1, bestRows = 1, bestTileW = 0, bestTileH = 0;

    for (let c = 1; c <= Math.min(maxCols, pageTileCount); c++) {
      const r = Math.ceil(pageTileCount / c);
      if (r > maxRows) continue;

      // Tile width if width is the binding constraint
      const tileWcandidate = (availW - (c - 1) * GRID_GAP) / c;
      const tileHcandidate = tileWcandidate * 9 / 16;
      const neededH = r * tileHcandidate + (r - 1) * GRID_GAP;

      let tileW, tileH;
      if (neededH <= availH) {
        // Width-constrained: use full tile width
        tileW = tileWcandidate;
        tileH = tileHcandidate;
      } else {
        // Height-constrained: scale down to fit
        tileH = (availH - (r - 1) * GRID_GAP) / r;
        tileW = tileH * 16 / 9;
      }

      if (tileW > bestTileW) {
        bestTileW = tileW;
        bestTileH = tileH;
        bestCols  = c;
        bestRows  = r;
      }
    }

    setLayout({
      cols:          bestCols,
      rows:          bestRows,
      tileW:         Math.floor(bestTileW),
      tileH:         Math.floor(bestTileH),
      pageSize,
      totalPages,
      pageTileCount,
    });
  }, [containerRef, totalStudents, page]);

  useEffect(() => {
    compute();
    const el = containerRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(compute);
    ro.observe(el);
    return () => ro.disconnect();
  }, [compute, containerRef]);

  return layout;
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
  const [swapPip, setSwapPip]         = useState(false);
  const [page, setPage]               = useState(0);
  const containerRef                  = useRef(null);

  const rawTracks = useTracks(
    [Track.Source.Camera, Track.Source.ScreenShare],
    { onlySubscribed: true },
  );

  const students = useMemo(() => {
    const groups = new Map();
    for (const trackRef of rawTracks) {
      const identity = trackRef.participant.identity || '';
      if (!identity.startsWith('student-')) continue;
      const studentId = getStudentIdentity(identity);
      if (!groups.has(studentId)) {
        groups.set(studentId, { studentId, participant: trackRef.participant, cameraTrack: null, screenTrack: null, handRaised: false });
      }
      const group = groups.get(studentId);
      if (trackRef.source === Track.Source.Camera)      group.cameraTrack = trackRef;
      if (trackRef.source === Track.Source.ScreenShare) group.screenTrack  = trackRef;
      try { group.handRaised = JSON.parse(trackRef.participant.metadata || '{}').hand_raised === true; } catch { group.handRaised = false; }
    }
    for (const studentId of Object.keys(snapshots)) {
      if (!groups.has(studentId)) groups.set(studentId, { studentId, participant: null, cameraTrack: null, screenTrack: null, handRaised: false });
    }
    return Array.from(groups.values()).map(g => {
      const score = snapshots[g.studentId];
      return {
        ...g,
        status:    score?.status  || 'Waiting...',
        score:     score ? `${Math.round(score.score)}/100` : 'No data',
        scoreNum:  score?.score   || 0,
        camera:    score ? (score.camera_on ? 'On' : 'Off') : (g.cameraTrack ? 'On' : 'Unknown'),
        lastUpdate: score?.last_update || null,
        isWarning: !score || score.is_warning,
      };
    }).sort((a, b) => {
      if (a.handRaised !== b.handRaised) return a.handRaised ? -1 : 1;
      return a.studentId.localeCompare(b.studentId);
    });
  }, [snapshots, rawTracks]);

  // Reset page when student count changes
  useEffect(() => { setPage(0); }, [students.length]);

  const layout = useGridLayout(containerRef, students.length, page);

  // ── Focused single-student view ───────────────────────────────────────────
  const focusedStudent = focusedStudentId ? students.find(s => s.studentId === focusedStudentId) : null;

  if (focusedStudent) {
    const student = focusedStudent;
    const hasScreen = !!student.screenTrack;
    let mainTrack = swapPip ? student.cameraTrack : (hasScreen ? student.screenTrack : student.cameraTrack);
    let pipTrack  = swapPip ? student.screenTrack  : (hasScreen ? student.cameraTrack : null);
    if (student.camera === 'Off' && mainTrack === student.cameraTrack) mainTrack = null;
    if (student.camera === 'Off' && pipTrack  === student.cameraTrack) pipTrack  = null;
    const returnToGrid = () => { setFocusedStudentId(null); setSwapPip(false); };

    return (
      <div className={`focused-container teacher-focused-container ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <section className="focus-layout teacher-focus-layout">
          {mainTrack ? (
            <article className="main-view" onDoubleClick={returnToGrid}>
              <VideoTrack trackRef={mainTrack} />
              <div className="view-overlay">
                <span>{student.studentId}{mainTrack.source === Track.Source.ScreenShare ? "'s screen" : ''}</span>
              </div>
            </article>
          ) : student.camera === 'Off' ? (
            <CameraOffTile label={`${student.studentId} camera off`} />
          ) : (
            <div className="empty-main screen-center"><p className="muted">No video available for {student.studentId}</p></div>
          )}
          {pipTrack && (
            <article className="pip-view pip-left" onClick={() => setSwapPip(!swapPip)} onDoubleClick={e => e.stopPropagation()}>
              <VideoTrack trackRef={pipTrack} />
            </article>
          )}
          <button className="floating-back-btn" onClick={returnToGrid} title="Back to grid">&#8592; Grid</button>
        </section>
      </div>
    );
  }

  // ── Grid view ─────────────────────────────────────────────────────────────
  const pageStudents = layout
    ? students.slice(page * layout.pageSize, (page + 1) * layout.pageSize)
    : students;

  return (
    <div
      ref={containerRef}
      className={`grid-container teacher-grid-container ${sidebarOpen ? 'sidebar-open' : ''}`}
    >
      {students.length === 0 ? (
        <div className="meeting-empty-state">
          <p className="muted">Waiting for students to join...</p>
        </div>
      ) : layout ? (
        <>
          {/* Tile grid with computed sizes */}
          <div
            className="teacher-grid meeting-grid"
            style={{
              gridTemplateColumns: `repeat(${layout.cols}, ${layout.tileW}px)`,
              gridTemplateRows:    `repeat(${layout.rows}, ${layout.tileH}px)`,
              gap:                 `${GRID_GAP}px`,
            }}
          >
            {pageStudents.map(student => {
              const displayTrack = student.screenTrack || student.cameraTrack;
              return (
                <article
                  key={student.studentId}
                  className={`meeting-tile ${student.handRaised ? 'hand-raised' : ''} ${student.isWarning ? 'tile-warning' : ''}`}
                  style={{ width: layout.tileW, height: layout.tileH }}
                  onClick={() => { setFocusedStudentId(student.studentId); setSwapPip(false); }}
                  title={`Click to focus ${student.studentId}`}
                >
                  {displayTrack ? <VideoTrack trackRef={displayTrack} /> : student.camera === 'Off' ? <CameraOffTile /> : <div className="video-placeholder" />}
                  <div className="tile-nameplate">
                    {student.handRaised && <Hand size={14} />}
                    <span>{student.studentId}</span>
                    <span className={`tile-status-dot ${getStatusClass(student.status)}`} />
                  </div>
                  <div className="tile-hover-hint">Click to focus</div>
                </article>
              );
            })}
          </div>

          {/* Pagination controls — only shown when there are multiple pages */}
          {layout.totalPages > 1 && (
            <div className="grid-pagination">
              <button
                className="page-btn"
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                aria-label="Previous page"
              >&#8249;</button>
              <span className="page-indicator">{page + 1} / {layout.totalPages}</span>
              <button
                className="page-btn"
                onClick={() => setPage(p => Math.min(layout.totalPages - 1, p + 1))}
                disabled={page >= layout.totalPages - 1}
                aria-label="Next page"
              >&#8250;</button>
            </div>
          )}
        </>
      ) : null}
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
          <ControlBar variation="minimal" controls={{ microphone: true, camera: true, screenShare: true, chat: false, leave: false }} />
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

