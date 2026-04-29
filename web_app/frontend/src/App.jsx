import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import { useState } from 'react';
import './index.css';
import TeacherDashboard from './pages/TeacherDashboard';
import StudentRoom from './pages/StudentRoom';

function Home() {
  const [role, setRole] = useState('student');
  const [roomId, setRoomId] = useState('');
  const [studentId, setStudentId] = useState('');
  const navigate = useNavigate();

  const handleJoin = (e) => {
    e.preventDefault();
    if (role === 'teacher') {
      const newRoom = Math.random().toString(36).substring(2, 8).toUpperCase();
      navigate(`/teacher/${newRoom}`);
    } else {
      if (!roomId || !studentId) return alert('Room ID and Student ID are required');
      navigate(`/student/${roomId}/${studentId}`);
    }
  };

  return (
    <div className="home-container">
      <div className="glass-card zoom-in">
        <h1 className="gradient-text">Focus AI Room</h1>
        <p className="subtitle">Real-time attention tracking powered by AI</p>

        <form onSubmit={handleJoin} className="join-form">
          <div className="role-selector">
            <button 
              type="button"
              className={role === 'student' ? 'active' : ''} 
              onClick={() => setRole('student')}
            >Student</button>
            <button 
              type="button"
              className={role === 'teacher' ? 'active' : ''} 
              onClick={() => setRole('teacher')}
            >Teacher</button>
          </div>

          {role === 'student' && (
            <>
              <input type="text" placeholder="Room ID" value={roomId} onChange={e => setRoomId(e.target.value)} required />
              <input type="text" placeholder="Your Name" value={studentId} onChange={e => setStudentId(e.target.value)} required />
            </>
          )}

          <button type="submit" className="primary-btn pulse-glow">
            {role === 'teacher' ? 'Create New Session' : 'Join Session'}
          </button>
        </form>
      </div>
    </div>
  );
}

function App() {
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

export default App;
