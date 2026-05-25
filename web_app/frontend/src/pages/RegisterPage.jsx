import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../lib/AuthContext';

export default function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('student');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (password.length < 6) {
      setError('Password must be at least 6 characters long');
      return;
    }

    setLoading(true);
    try {
      await register(username, password, role);
      navigate('/');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="screen-center animate-in">
      <div className="panel auth-panel">
        <h1 className="text-center" style={{ marginBottom: '0.5rem' }}>Create Account</h1>
        <p className="text-center muted" style={{ marginBottom: '2rem' }}>Join AI Focus Classroom today</p>
        
        {error && <p className="error-text text-center">{error}</p>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="role-picker">
            <button 
              type="button" 
              className={`role-option ${role === 'student' ? 'active' : ''}`}
              onClick={() => setRole('student')}
            >
              <div className="role-title">Student</div>
              <div className="text-sm muted">Join classes</div>
            </button>
            <button 
              type="button" 
              className={`role-option ${role === 'teacher' ? 'active' : ''}`}
              onClick={() => setRole('teacher')}
            >
              <div className="role-title">Teacher</div>
              <div className="text-sm muted">Host classes</div>
            </button>
          </div>

          <div className="form-group">
            <label>Username</label>
            <input 
              type="text" 
              value={username} 
              onChange={(e) => setUsername(e.target.value)} 
              required 
              placeholder="Choose a username"
              minLength={2}
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input 
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required 
              placeholder="At least 6 characters"
              minLength={6}
            />
          </div>
          
          <button type="submit" disabled={loading} style={{ width: '100%', marginTop: '1rem' }}>
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>
        
        <div className="text-center mt-4 text-sm">
          <span className="muted">Already have an account? </span>
          <Link to="/login" style={{ color: 'var(--primary)' }}>Sign in</Link>
        </div>
      </div>
    </main>
  );
}
