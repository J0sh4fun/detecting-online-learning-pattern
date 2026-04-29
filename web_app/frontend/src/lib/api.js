const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export function createRoom({ teacherId, roomName }) {
  return request('/api/rooms', {
    method: 'POST',
    body: JSON.stringify({ teacher_id: teacherId, room_name: roomName }),
  });
}

export function joinRoom({ roomCode, studentId }) {
  return request('/api/rooms/join', {
    method: 'POST',
    body: JSON.stringify({ room_code: roomCode, student_id: studentId }),
  });
}

export function endRoom({ roomCode, token }) {
  const search = new URLSearchParams({ token });
  return request(`/api/rooms/${encodeURIComponent(roomCode)}/end?${search.toString()}`, {
    method: 'POST',
  });
}

export function verifyFrame({ token, roomCode, studentId, clientScore, frameBase64 }) {
  return request('/api/verify/frame', {
    method: 'POST',
    body: JSON.stringify({
      token,
      room_code: roomCode,
      student_id: studentId,
      client_score: clientScore,
      frame_base64: frameBase64,
    }),
  });
}

export const AppConfig = {
  apiBaseUrl: API_BASE_URL,
};

