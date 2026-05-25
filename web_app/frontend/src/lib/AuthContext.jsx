import { createContext, useContext, useEffect, useState } from 'react';
import { getMe, login as apiLogin, register as apiRegister } from './api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('focus_auth_token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      localStorage.setItem('focus_auth_token', token);
      getMe()
        .then((userData) => {
          setUser(userData);
        })
        .catch(() => {
          setToken(null);
          setUser(null);
          localStorage.removeItem('focus_auth_token');
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      localStorage.removeItem('focus_auth_token');
      setUser(null);
      setLoading(false);
    }
  }, [token]);

  const login = async (username, password) => {
    const data = await apiLogin({ username, password });
    setToken(data.token);
    setUser({ id: data.id, username: data.username, role: data.role });
    return data;
  };

  const register = async (username, password, role) => {
    const data = await apiRegister({ username, password, role });
    setToken(data.token);
    setUser({ id: data.id, username: data.username, role: data.role });
    return data;
  };

  const logout = () => {
    setToken(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
