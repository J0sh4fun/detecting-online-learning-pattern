import { useMemo, useState, Fragment, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, Legend 
} from 'recharts';
import { 
  Users, Video, VideoOff, Award, Download, ChevronDown, ChevronUp, Clock,
  CheckCircle, AlertTriangle, UserMinus
} from 'lucide-react';
import { exportReportToPdf } from '../utils/pdfExport';

function getStudentRating(avgScore) {
  if (avgScore >= 70) return { label: 'Focused', class: 'success', icon: CheckCircle };
  if (avgScore >= 30) return { label: 'Distracted', class: 'warning', icon: AlertTriangle };
  return { label: 'Absence', class: 'error', icon: UserMinus };
}

function calculateLabelFrequencies(timeline) {
  const counts = {};
  timeline.forEach(point => {
    counts[point.status] = (counts[point.status] || 0) + 1;
  });
  const total = timeline.length || 1;
  return Object.entries(counts).map(([label, count]) => ({
    label,
    count,
    percentage: Math.round((count / total) * 100)
  })).sort((a, b) => b.count - a.count);
}

export default function ReportView({ report, onBack }) {
  const [expandedStudent, setExpandedStudent] = useState(null);
  const [isExporting, setIsExporting] = useState(false);

  async function handleExport() {
    setIsExporting(true);
    try {
      exportReportToPdf(report);
    } finally {
      setIsExporting(false);
    }
  }

  const stats = useMemo(() => {
    const totalStudents = report.students.length;
    let cameraOnCount = 0;
    report.students.forEach(s => {
      const lastPoint = s.timeline[s.timeline.length - 1];
      if (lastPoint?.camera_on) cameraOnCount++;
    });

    return {
      totalStudents,
      cameraOnCount,
      cameraOffCount: totalStudents - cameraOnCount,
      classAvg: report.class_average_score
    };
  }, [report]);

  // Process timeline data for charts
  const timelineData = useMemo(() => {
    // We need to align all student timelines into a single class timeline
    // For simplicity, we'll bucket by time or just use the largest timeline as base
    const allTimestamps = new Set();
    report.students.forEach(s => {
      s.timeline.forEach(p => allTimestamps.add(new Date(p.timestamp).getTime()));
    });

    const sortedTimestamps = Array.from(allTimestamps).sort();
    
    // Bucket into ~20-30 points for the chart to be readable
    const bucketCount = 20;
    const step = Math.max(1, Math.floor(sortedTimestamps.length / bucketCount));
    
    const chartPoints = [];
    for (let i = 0; i < sortedTimestamps.length; i += step) {
      const ts = sortedTimestamps[i];
      const timeStr = new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      
      let sumScore = 0;
      let presentCount = 0;
      let camOnCount = 0;

      report.students.forEach(s => {
        // Find the point closest to this timestamp in this student's timeline
        const point = s.timeline.find(p => Math.abs(new Date(p.timestamp).getTime() - ts) < 5000); // 5s window
        if (point) {
          sumScore += point.score;
          presentCount++;
          if (point.camera_on) camOnCount++;
        }
      });

      chartPoints.push({
        time: timeStr,
        timestamp: ts,
        avgScore: presentCount > 0 ? Math.round(sumScore / presentCount) : 0,
        cameraOn: camOnCount,
        totalStudents: presentCount
      });
    }

    return chartPoints;
  }, [report]);

  return (
    <div className="report-panel panel animate-in">
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Class Report: {report.room_name}</h1>
          <p className="muted">Generated at {new Date(report.generated_at).toLocaleString()}</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            onClick={handleExport} 
            disabled={isExporting}
            className="secondary-button" 
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.1)' }}
          >
            <Download size={18} /> {isExporting ? 'Generating...' : 'Export PDF'}
          </button>
          <button onClick={onBack}>Back to Home</button>
        </div>
      </header>

      <section className="report-summary-grid">
        <div className="summary-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <h3>Class Average</h3>
            <Award className="success-text" size={20} />
          </div>
          <div className="summary-value" style={{ color: stats.classAvg > 70 ? 'var(--success)' : stats.classAvg > 40 ? 'var(--warning)' : 'var(--error)' }}>
            {stats.classAvg}%
          </div>
        </div>
        <div className="summary-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <h3>Students Present</h3>
            <Users size={20} />
          </div>
          <div className="summary-value">{stats.totalStudents}</div>
        </div>
        <div className="summary-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <h3>Camera Usage</h3>
            <Video size={20} />
          </div>
          <div className="summary-value">
            <span style={{ color: 'var(--success)' }}>{stats.cameraOnCount}</span>
            <span className="muted" style={{ fontSize: '1.2rem', margin: '0 0.5rem' }}>/</span>
            <span style={{ color: 'var(--error)' }}>{stats.cameraOffCount}</span>
          </div>
          <p className="text-sm muted">ON / OFF</p>
        </div>
      </section>

      <section className="charts-row">
        <div className="chart-container">
          <h3 className="text-sm muted" style={{ marginBottom: '1rem', textTransform: 'uppercase' }}>FOCUS SCORE OVER TIME</h3>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="time" stroke="var(--muted)" fontSize={12} />
              <YAxis domain={[0, 100]} stroke="var(--muted)" fontSize={12} />
              <Tooltip 
                contentStyle={{ background: 'var(--card)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text)' }}
              />
              <Line type="monotone" dataKey="avgScore" stroke="var(--primary)" strokeWidth={3} dot={false} name="Avg Focus %" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-container">
          <h3 className="text-sm muted" style={{ marginBottom: '1rem', textTransform: 'uppercase' }}>ATTENDANCE & CAMERA TRENDS</h3>
          <ResponsiveContainer width="100%" height="90%">
            <AreaChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="time" stroke="var(--muted)" fontSize={12} />
              <YAxis stroke="var(--muted)" fontSize={12} />
              <Tooltip 
                contentStyle={{ background: 'var(--card)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
              />
              <Legend verticalAlign="top" height={36}/>
              <Area type="monotone" dataKey="totalStudents" stackId="1" stroke="#8884d8" fill="#8884d8" name="Total Present" />
              <Area type="monotone" dataKey="cameraOn" stackId="2" stroke="#82ca9d" fill="#82ca9d" name="Camera On" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="status-panel">
        <h3>Student Detailed Performance</h3>
        <table className="status-table">
          <thead>
            <tr>
              <th style={{ width: '40px' }}></th>
              <th>Student ID</th>
              <th>Average Score</th>
              <th>Rating</th>
              <th>Sessions</th>
            </tr>
          </thead>
          <tbody>
            {report.students.map((student) => {
              const rating = getStudentRating(student.average_score);
              const RatingIcon = rating.icon;
              const isExpanded = expandedStudent === student.student_id;
              const labelFreqs = calculateLabelFrequencies(student.timeline);

              return (
                <Fragment key={student.student_id}>
                  <tr 
                    className="student-report-row" 
                    onClick={() => setExpandedStudent(isExpanded ? null : student.student_id)}
                  >
                    <td>
                      {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </td>
                    <td className="bold">{student.student_id}</td>
                    <td>
                      <div className="table-score-container">
                        <div className="table-score-bar">
                          <div 
                            className="bar-fill" 
                            style={{ 
                              width: `${student.average_score}%`,
                              backgroundColor: student.average_score > 70 ? 'var(--success)' : student.average_score > 40 ? 'var(--warning)' : 'var(--error)'
                            }}
                          ></div>
                        </div>
                        <span className="score-label">{Math.round(student.average_score)}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`status-badge ${rating.class}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                        <RatingIcon size={14} />
                        {rating.label}
                      </span>
                    </td>
                    <td className="muted">{student.timeline.length} samples</td>
                  </tr>
                  {isExpanded && (
                    <tr>
                      <td colSpan="5" className="student-details-cell">
                        <div className="animate-in">
                          <h4 className="text-sm muted" style={{ marginBottom: '1rem', textTransform: 'uppercase' }}>Focus Label Frequency</h4>
                          <div className="label-frequency-grid">
                            {labelFreqs.map(freq => (
                              <div key={freq.label} className="label-stat">
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                                  <span className="label-name">{freq.label}</span>
                                  <span className="bold">{freq.percentage}%</span>
                                </div>
                                <div className="label-bar-container">
                                  <div 
                                    className="bar-fill" 
                                    style={{ 
                                      width: `${freq.percentage}%`, 
                                      backgroundColor: freq.label === 'Focused' ? 'var(--success)' : 'var(--primary)',
                                      height: '100%'
                                    }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}
