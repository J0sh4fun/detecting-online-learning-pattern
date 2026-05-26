import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../lib/AuthContext';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const data = await login(username, password);
      if (data.role === 'teacher') {
        navigate('/');
      } else {
        navigate('/');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-shell animate-in">
      <section className="auth-visual">
        <div className="brand-lockup">
          <span className="brand-mark">AI</span>
          <div>
            <div className="brand-title">AI Focus Classroom</div>
            <div className="brand-subtitle" style={{ color: 'rgba(255,255,255,0.76)' }}>Video learning workspace</div>
          </div>
        </div>
        <h1>Connect live classes with focus insight.</h1>
      </section>

      <section className="auth-side">
        <div className="panel auth-panel">
          <h1 className="text-center" style={{ marginBottom: '0.5rem' }}>Welcome back</h1>
          <p className="text-center muted" style={{ marginBottom: '2rem' }}>Sign in to continue to your classroom.</p>
        
          {error && <p className="error-text text-center">{error}</p>}
        
          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label>Username</label>
              <input 
                type="text" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                required 
                placeholder="Enter your username"
              />
            </div>
            <div className="form-group">
              <label>Password</label>
              <input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                required 
                placeholder="Enter your password"
              />
            </div>
          
            <button type="submit" disabled={loading} style={{ width: '100%', marginTop: '1rem' }}>
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>
        
          <div className="text-center mt-4 text-sm">
            <span className="muted">Don't have an account? </span>
            <Link to="/register">Create one</Link>
          </div>
        </div>
      </section>
    </main>
  );
}
