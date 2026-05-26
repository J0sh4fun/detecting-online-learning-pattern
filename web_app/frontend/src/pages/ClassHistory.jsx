import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getClassHistory } from '../lib/api';

export default function ClassHistory() {
  const navigate = useNavigate();
  const [rooms, setRooms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await getClassHistory();
      setRooms(data.rooms || []);
    } catch (err) {
      setError('Failed to load class history');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <main className="screen-center">Loading history...</main>;

  return (
    <main className="history-layout">
      <header className="class-header panel">
        <div>
          <h1>Class History</h1>
          <p className="muted">Review completed rooms and open detailed reports.</p>
        </div>
        <button onClick={() => navigate('/')} className="secondary-button">
          Back to Dashboard
        </button>
      </header>

      {error && <p className="error-text panel">{error}</p>}

      <div className="history-content history-content-single">
        <div className="panel history-list">
          {rooms.length === 0 ? (
            <p className="muted">No classes finished yet.</p>
          ) : (
            <table className="status-table">
              <thead>
                <tr>
                  <th>Room</th>
                  <th>Status</th>
                  <th>Date</th>
                  <th>Students</th>
                  <th>Average</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {rooms.map((room) => (
                  <tr key={room.room_code}>
                    <td>
                      <div className="bold">{room.room_name}</div>
                      <div className="text-sm muted">{room.room_code}</div>
                    </td>
                    <td>
                      <span className={`status-badge ${room.status === 'ended' ? 'success' : 'warning'}`}>
                        {room.status}
                      </span>
                    </td>
                    <td>{new Date(room.created_at).toLocaleDateString()}</td>
                    <td>{room.student_count > 0 ? room.student_count : '-'}</td>
                    <td>{room.class_average ? `${room.class_average.toFixed(1)}%` : '-'}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button 
                        className="secondary-button" 
                        onClick={() => navigate(`/history/${encodeURIComponent(room.room_code)}/report`)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </main>
  );
}
