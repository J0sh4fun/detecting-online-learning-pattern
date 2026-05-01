const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const FALLBACK_API_BASE_URL = API_BASE_URL.includes('localhost')
  ? API_BASE_URL.replace('localhost', '127.0.0.1')
  : (API_BASE_URL.includes('127.0.0.1') ? API_BASE_URL.replace('127.0.0.1', 'localhost') : null);

function isNetworkError(error) {
  return error instanceof TypeError;
}

async function doRequest(baseUrl, path, options = {}) {
  return fetch(`${baseUrl}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
}

async function request(path, options = {}) {
  let response;
  try {
    response = await doRequest(API_BASE_URL, path, options);
  } catch (error) {
    if (FALLBACK_API_BASE_URL && isNetworkError(error)) {
      try {
        response = await doRequest(FALLBACK_API_BASE_URL, path, options);
      } catch (fallbackError) {
        throw new Error(
          `Cannot connect to backend API (${API_BASE_URL} or ${FALLBACK_API_BASE_URL}). `
          + 'Start FastAPI server: uvicorn main:app --reload --port 8000',
          { cause: fallbackError },
        );
      }
    } else {
      throw new Error(
        `Cannot connect to backend API (${API_BASE_URL}). `
        + 'Start FastAPI server: uvicorn main:app --reload --port 8000',
        { cause: error },
      );
    }
  }

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

export function scoreFrame({ token, roomCode, studentId, frameBase64 }) {
  return request('/api/score/frame', {
    method: 'POST',
    body: JSON.stringify({
      token,
      room_code: roomCode,
      student_id: studentId,
      frame_base64: frameBase64,
    }),
  });
}

export const AppConfig = {
  apiBaseUrl: API_BASE_URL,
};

