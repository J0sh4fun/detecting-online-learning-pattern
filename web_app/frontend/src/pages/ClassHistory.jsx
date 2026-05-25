import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getClassHistory, getClassReport } from '../lib/api';

export default function ClassHistory() {
  const navigate = useNavigate();
  const [rooms, setRooms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedReport, setSelectedReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);

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

  const viewReport = async (roomCode) => {
    setReportLoading(true);
    setSelectedReport(null);
    try {
      const data = await getClassReport(roomCode);
      setSelectedReport(data);
    } catch (err) {
      alert('Failed to load report for ' + roomCode);
    } finally {
      setReportLoading(false);
    }
  };

  if (loading) return <main className="screen-center">Loading history...</main>;

  return (
    <main className="home-layout" style={{ maxWidth: '1000px' }}>
      <header className="app-header panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2>Class History</h2>
        <button onClick={() => navigate('/')} className="secondary-button" style={{ background: 'rgba(255,255,255,0.1)' }}>
          Back to Dashboard
        </button>
      </header>

      {error && <p className="error-text panel">{error}</p>}

      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
        <div className="panel" style={{ flex: 1, maxHeight: 'calc(100vh - 150px)', overflowY: 'auto' }}>
          {rooms.length === 0 ? (
            <p className="muted">No classes finished yet.</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <th style={{ padding: '0.5rem' }}>Room</th>
                  <th style={{ padding: '0.5rem' }}>Status</th>
                  <th style={{ padding: '0.5rem' }}>Date</th>
                  <th style={{ padding: '0.5rem' }}>Students</th>
                  <th style={{ padding: '0.5rem' }}>Average</th>
                  <th style={{ padding: '0.5rem' }}></th>
                </tr>
              </thead>
              <tbody>
                {rooms.map((room) => (
                  <tr key={room.room_code} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '0.5rem' }}>
                      <div className="bold">{room.room_name}</div>
                      <div className="text-sm muted">{room.room_code}</div>
                    </td>
                    <td style={{ padding: '0.5rem' }}>
                      <span style={{ 
                        padding: '0.2rem 0.5rem', 
                        borderRadius: '4px', 
                        fontSize: '0.8rem',
                        background: room.status === 'ended' ? 'rgba(74, 222, 128, 0.2)' : 'rgba(250, 204, 21, 0.2)',
                        color: room.status === 'ended' ? '#4ade80' : '#facc15'
                      }}>
                        {room.status}
                      </span>
                    </td>
                    <td style={{ padding: '0.5rem' }}>{new Date(room.created_at).toLocaleDateString()}</td>
                    <td style={{ padding: '0.5rem' }}>{room.student_count > 0 ? room.student_count : '-'}</td>
                    <td style={{ padding: '0.5rem' }}>{room.class_average ? `${room.class_average.toFixed(1)}%` : '-'}</td>
                    <td style={{ padding: '0.5rem', textAlign: 'right' }}>
                      <button 
                        className="secondary-button" 
                        style={{ padding: '0.3rem 0.6rem', fontSize: '0.8rem' }}
                        onClick={() => viewReport(room.room_code)}
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

        {/* Report Preview Pane */}
        {selectedReport || reportLoading ? (
          <div className="panel animate-in" style={{ flex: 1, position: 'sticky', top: '1.5rem' }}>
            {reportLoading ? (
              <p>Loading report...</p>
            ) : (
              <div>
                <h3>{selectedReport.room_name} Report</h3>
                <p className="muted text-sm" style={{ marginBottom: '1rem' }}>
                  Code: {selectedReport.room_code} • Generated: {new Date(selectedReport.generated_at).toLocaleString()}
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                    <div className="text-sm muted">Class Average</div>
                    <div className="bold" style={{ fontSize: '1.5rem', color: selectedReport.class_average_score >= 80 ? '#4ade80' : '#facc15' }}>
                      {selectedReport.class_average_score}%
                    </div>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                    <div className="text-sm muted">Students</div>
                    <div className="bold" style={{ fontSize: '1.5rem' }}>
                      {selectedReport.students.length}
                    </div>
                  </div>
                </div>

                <h4>Student Scores</h4>
                <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                  {selectedReport.students.map((student) => (
                    <div key={student.student_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <span>{student.student_id}</span>
                      <span className="bold">{student.average_score}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </main>
  );
}
