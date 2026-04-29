import { useMemo, useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate, useSearchParams } from 'react-router-dom';
import TeacherDashboard from './pages/TeacherDashboard';
import StudentRoom from './pages/StudentRoom';
import { createRoom, joinRoom } from './lib/api';
import { saveSession } from './lib/sessionStore';
import './index.css';

function Home() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [teacherId, setTeacherId] = useState('');
  const [roomName, setRoomName] = useState('Focus Monitoring Class');
  const [roomCode, setRoomCode] = useState(searchParams.get('join') || '');
  const [studentId, setStudentId] = useState('');
  const [error, setError] = useState('');

  const inviteMode = useMemo(() => Boolean(searchParams.get('join')), [searchParams]);

  async function handleCreateRoom(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const session = await createRoom({ teacherId: teacherId.trim(), roomName: roomName.trim() });
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
      const session = await joinRoom({ roomCode: roomCode.trim().toUpperCase(), studentId: studentId.trim() });
      saveSession('student', { ...session, student_id: studentId.trim() });
      navigate(`/student/${session.room_code}/${encodeURIComponent(studentId.trim())}`, {
        state: { session: { ...session, student_id: studentId.trim() } },
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="home-layout">
      <section className="panel">
        <h1>AI Focus Classroom</h1>
        <p className="muted">LiveKit SFU + Worker ONNX scoring + anti-cheat verification</p>
        {error && <p className="error-text">{error}</p>}
      </section>

      <section className="forms-layout">
        <form className="panel form-panel" onSubmit={handleCreateRoom}>
          <h2>Teacher</h2>
          <input
            value={teacherId}
            onChange={(event) => setTeacherId(event.target.value)}
            placeholder="Teacher ID"
            required
          />
          <input
            value={roomName}
            onChange={(event) => setRoomName(event.target.value)}
            placeholder="Room name"
            required
          />
          <button disabled={loading} type="submit">Create classroom</button>
        </form>

        <form className="panel form-panel" onSubmit={handleJoinRoom}>
          <h2>Student</h2>
          <input
            value={roomCode}
            onChange={(event) => setRoomCode(event.target.value.toUpperCase())}
            placeholder="Room code"
            required
          />
          <input
            value={studentId}
            onChange={(event) => setStudentId(event.target.value)}
            placeholder="Student ID"
            required
          />
          <button disabled={loading} type="submit">
            {inviteMode ? 'Join invited room' : 'Join classroom'}
          </button>
        </form>
      </section>
    </main>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/teacher/:roomId" element={<TeacherDashboard />} />
        <Route path="/student/:roomId/:studentId" element={<StudentRoom />} />
      </Routes>
    </Router>
  );
}

