import { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

export default function TeacherDashboard() {
  const { roomId } = useParams();
  const navigate = useNavigate();
  const [scores, setScores] = useState({});
  const wsRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/teacher/${roomId}`);
    
    wsRef.current.onopen = () => {
      console.log("Teacher WS connected");
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'scores_update') {
        setScores(data.scores); // Expected structure: { studentId: { score: 95, label: 'Focused' } }
      }
    };

    wsRef.current.onclose = () => {
      console.log("Teacher WS disconnected");
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [roomId]);

  const endSession = () => {
    // Generate Report
    let totalScore = 0;
    const studentKeys = Object.keys(scores);
    studentKeys.forEach(k => totalScore += scores[k].score);
    const average = studentKeys.length ? (totalScore / studentKeys.length).toFixed(1) : 0;
    
    alert(`Session Report for Room: ${roomId}\nTotal Students: ${studentKeys.length}\nAverage Focus Score: ${average}/100`);
    navigate('/');
  };

  const getScoreColor = (sc) => {
    if (sc > 80) return '#00ff6a';
    if (sc > 50) return '#ffd000';
    return '#ff3737';
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div>
          <h2 className="gradient-text">Teacher Dashboard</h2>
          <span className="room-badge">Room Code: {roomId}</span>
          <span className="student-count">{Object.keys(scores).length} / 50 Students</span>
        </div>
        <button onClick={endSession} className="danger-btn">End Session & Report</button>
      </header>
      
      <div className="table-container slide-up">
        {Object.keys(scores).length === 0 ? (
          <div className="empty-state">Waiting for students to join...</div>
        ) : (
          <table className="student-table">
            <thead>
              <tr>
                <th>Student</th>
                <th>Status</th>
                <th>Focus Score</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(scores).map(([studentId, score]) => (
                <tr key={studentId}>
                  <td>{studentId}</td>
                  <td>
                    <span 
                      className="status-pill" 
                      style={{ 
                        color: getScoreColor(score.score), 
                        backgroundColor: getScoreColor(score.score) + '22',
                        border: `1px solid ${getScoreColor(score.score)}55`
                      }}
                    >
                      {score.label || (score.score > 80 ? 'Focused' : (score.score > 50 ? 'Distracted' : 'Unfocused'))}
                    </span>
                  </td>
                  <td>
                    <div className="score-cell">
                      <span className="score-text" style={{ color: getScoreColor(score.score) }}>
                        {Math.round(score.score)}/100
                      </span>
                      <div className="progress-bar-bg">
                        <div 
                          className="progress-bar-fill" 
                          style={{ width: `${score.score}%`, backgroundColor: getScoreColor(score.score) }} 
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
