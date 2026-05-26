import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getClassReport } from '../lib/api';
import ReportView from './ReportView';

export default function HistoryReportPage() {
  const { roomCode } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let disposed = false;

    async function loadReport() {
      setLoading(true);
      setError('');
      try {
        const data = await getClassReport(roomCode);
        if (!disposed) {
          setReport(data);
        }
      } catch (err) {
        if (!disposed) {
          setError(err.message || 'Failed to load report');
        }
      } finally {
        if (!disposed) {
          setLoading(false);
        }
      }
    }

    loadReport();
    return () => {
      disposed = true;
    };
  }, [roomCode]);

  if (loading) {
    return <main className="screen-center">Loading report...</main>;
  }

  if (error) {
    return (
      <main className="screen-center">
        <div className="panel text-center" style={{ maxWidth: '460px', padding: '2rem' }}>
          <h1>Report unavailable</h1>
          <p className="error-text">{error}</p>
          <button className="secondary-button" onClick={() => navigate('/history')}>
            Back to history
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="report-overlay">
      <ReportView report={report} onBack={() => navigate('/history')} backLabel="Back to history" />
    </main>
  );
}
