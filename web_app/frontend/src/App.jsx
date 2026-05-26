import { useMemo, useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate, useSearchParams, Navigate } from 'react-router-dom';
import TeacherDashboard from './pages/TeacherDashboard';
import StudentRoom from './pages/StudentRoom';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ClassHistory from './pages/ClassHistory';
import HistoryReportPage from './pages/HistoryReportPage';
import { createRoom, joinRoom } from './lib/api';
import { saveSession } from './lib/sessionStore';
import { AuthProvider, useAuth } from './lib/AuthContext';
import ProtectedRoute from './lib/ProtectedRoute';
import { CalendarClock, LogOut, Plus, Video } from 'lucide-react';
import './index.css';

function Home() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user, logout } = useAuth();
  const [loading, setLoading] = useState(false);
  const [roomName, setRoomName] = useState('Focus Monitoring Class');
  const [roomCode, setRoomCode] = useState(searchParams.get('join') || '');
  const [error, setError] = useState('');

  const inviteMode = useMemo(() => Boolean(searchParams.get('join')), [searchParams]);

  async function handleCreateRoom(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const session = await createRoom({ roomName: roomName.trim() });
      saveSession('teacher', session);
      navigate(`/teacher/${session.room_code}`, { state: { session } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleJoinRoom(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const session = await joinRoom({ roomCode: roomCode.trim().toUpperCase() });
      saveSession('student', { ...session, student_id: user.username });
      navigate(`/student/${session.room_code}/${encodeURIComponent(user.username)}`, {
        state: { session: { ...session, student_id: user.username } },
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="zoom-shell">
      <header className="zoom-nav">
        <div className="brand-lockup">
          <span className="brand-mark">AI</span>
          <div>
            <div className="brand-title">AI Focus Classroom</div>
            <div className="brand-subtitle">Logged in as {user?.username} ({user?.role})</div>
          </div>
        </div>
        <div className="nav-actions">
          {user?.role === 'teacher' && (
            <button onClick={() => navigate('/history')} className="secondary-button">
              <CalendarClock size={17} />
              Class History
            </button>
          )}
          <button onClick={logout} className="danger-button">
            <LogOut size={17} />
            Sign Out
          </button>
        </div>
      </header>

      <section className="home-layout">
        <div className="panel home-hero">
          <div className="hero-copy">
            <span className="eyebrow"><Video size={16} /> Live classroom focus monitoring</span>
            <h1>Bring class video and attention signals into one clean workspace.</h1>
            <p>
              Host monitored sessions, invite students, and review focus reports from a familiar meeting-first interface.
            </p>
            <div className="hero-actions">
              {user?.role === 'teacher' && (
                <button type="button" onClick={() => document.getElementById('roomNameInput')?.focus()}>
                  <Plus size={18} />
                  New classroom
                </button>
              )}
              {user?.role === 'student' && (
                <button type="button" onClick={() => document.getElementById('roomCodeInput')?.focus()}>
                  <Video size={18} />
                  Join classroom
                </button>
              )}
              {user?.role === 'teacher' && (
                <button type="button" className="secondary-button" onClick={() => navigate('/history')}>
                  <CalendarClock size={18} />
                  View history
                </button>
              )}
            </div>
          </div>

          <aside className="meeting-panel">
            <div className="meeting-preview">
              <div className="preview-video">
                <span className="preview-chip">AI Focus Classroom</span>
              </div>
            </div>
            {error && <p className="error-text">{error}</p>}
            {user?.role === 'teacher' ? (
              <form className="form-panel" onSubmit={handleCreateRoom}>
                <div>
                  <h2>Host a class</h2>
                  <p className="muted">Create a room and share the invite code with students.</p>
                </div>
                <input
                  id="roomNameInput"
                  value={roomName}
                  onChange={(event) => setRoomName(event.target.value)}
                  placeholder="Room name"
                  required
                />
                <button disabled={loading} type="submit">
                  <Plus size={18} />
                  {loading ? 'Creating...' : 'Create classroom'}
                </button>
              </form>
            ) : (
              <form className="form-panel" onSubmit={handleJoinRoom}>
                <div>
                  <h2>Join a class</h2>
                  <p className="muted">Enter the room code provided by your teacher.</p>
                </div>
                <input
                  id="roomCodeInput"
                  value={roomCode}
                  onChange={(event) => setRoomCode(event.target.value.toUpperCase())}
                  placeholder="Room code"
                  required
                />
                <button disabled={loading} type="submit">
                  <Video size={18} />
                  {loading ? 'Joining...' : inviteMode ? 'Join invited room' : 'Join classroom'}
                </button>
              </form>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}

function MainApp() {
  const { user, loading } = useAuth();
  
  if (loading) {
    return <main className="screen-center">Loading...</main>;
  }

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/" /> : <LoginPage />} />
      <Route path="/register" element={user ? <Navigate to="/" /> : <RegisterPage />} />
      
      <Route path="/" element={<ProtectedRoute><Home /></ProtectedRoute>} />
      
      <Route path="/teacher/:roomId" element={
        <ProtectedRoute requiredRole="teacher">
          <TeacherDashboard />
        </ProtectedRoute>
      } />
      
      <Route path="/student/:roomId/:studentId" element={
        <ProtectedRoute requiredRole="student">
          <StudentRoom />
        </ProtectedRoute>
      } />

      <Route path="/history" element={
        <ProtectedRoute requiredRole="teacher">
          <ClassHistory />
        </ProtectedRoute>
      } />

      <Route path="/history/:roomCode/report" element={
        <ProtectedRoute requiredRole="teacher">
          <HistoryReportPage />
        </ProtectedRoute>
      } />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <MainApp />
      </Router>
    </AuthProvider>
  );
}

