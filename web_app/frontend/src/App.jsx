import { useMemo, useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate, useSearchParams, Navigate } from 'react-router-dom';
import TeacherDashboard from './pages/TeacherDashboard';
import StudentRoom from './pages/StudentRoom';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ClassHistory from './pages/ClassHistory';
import { createRoom, joinRoom } from './lib/api';
import { saveSession } from './lib/sessionStore';
import { AuthProvider, useAuth } from './lib/AuthContext';
import ProtectedRoute from './lib/ProtectedRoute';
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
    <main className="home-layout">
      <header className="app-header panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <span className="bold">AI Focus Classroom</span>
          <span className="muted" style={{ marginLeft: '1rem', borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '1rem' }}>
            Logged in as {user?.username} ({user?.role})
          </span>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          {user?.role === 'teacher' && (
            <button onClick={() => navigate('/history')} className="secondary-button" style={{ background: 'rgba(255,255,255,0.1)' }}>
              Class History
            </button>
          )}
          <button onClick={logout} className="secondary-button" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}>
            Sign Out
          </button>
        </div>
      </header>

      <section className="panel">
        <h1>Welcome, {user?.username}!</h1>
        <p className="muted">Select an action below based on your role.</p>
        {error && <p className="error-text">{error}</p>}
      </section>

      <section className="forms-layout" style={{ marginTop: '1.5rem' }}>
        {user?.role === 'teacher' ? (
          <form className="panel form-panel" onSubmit={handleCreateRoom}>
            <h2>Host a Class</h2>
            <input
              value={roomName}
              onChange={(event) => setRoomName(event.target.value)}
              placeholder="Room name"
              required
            />
            <button disabled={loading} type="submit">Create classroom</button>
          </form>
        ) : (
          <form className="panel form-panel" onSubmit={handleJoinRoom}>
            <h2>Join a Class</h2>
            <input
              value={roomCode}
              onChange={(event) => setRoomCode(event.target.value.toUpperCase())}
              placeholder="Room code"
              required
            />
            <button disabled={loading} type="submit">
              {inviteMode ? 'Join invited room' : 'Join classroom'}
            </button>
          </form>
        )}
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

