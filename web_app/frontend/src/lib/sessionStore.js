const KEY = 'focus-ai-room-session';

export function saveSession(role, payload) {
  const current = loadSessions();
  current[role] = payload;
  window.sessionStorage.setItem(KEY, JSON.stringify(current));
}

export function getSession(role) {
  const all = loadSessions();
  return all[role] || null;
}

function loadSessions() {
  const raw = window.sessionStorage.getItem(KEY);
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

