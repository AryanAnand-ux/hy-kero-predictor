const resolveApiBase = () => {
  const configured = (import.meta.env.VITE_API_URL || '').trim();
  if (configured) return configured.replace(/\/$/, '');

  return '/api';
};

const API = resolveApiBase();

const resolveApiKey = () => {
  const configured = (import.meta.env.VITE_API_KEY || '').trim();
  if (configured) return configured;
  return 'hykero-secret-key';
};

const API_KEY = resolveApiKey();

const getJsonHeaders = () => {
  const headers = { 'Content-Type': 'application/json' };
  if (API_KEY) headers['X-API-Key'] = API_KEY;
  return headers;
};

const getAuthHeaders = () => {
  const headers = {};
  if (API_KEY) headers['X-API-Key'] = API_KEY;
  return headers;
};

export const fetchHistory = (shift = '', startDate = '', endDate = '', limit = 200) => {
  let url = `${API}/history?limit=${limit}`;
  if (shift) url += `&shift=${shift}`;
  if (startDate) url += `&start_date=${startDate}`;
  if (endDate) url += `&end_date=${endDate}`;
  return fetch(url, { headers: getJsonHeaders() }).then(r => {
    if (!r.ok) throw new Error('Failed to fetch history');
    return r.json();
  });
};

export const fetchHistoryStats = () =>
  fetch(`${API}/history/stats`, { headers: getJsonHeaders() }).then(r => {
    if (!r.ok) throw new Error('Failed to fetch stats');
    return r.json();
  });

export const fetchModelMetrics = () =>
  fetch(`${API}/model-metrics`, { headers: getJsonHeaders() }).then(r => {
    if (!r.ok) throw new Error('Failed to fetch model metrics');
    return r.json();
  });

export const fetchFeatureImportance = (topN = 15) =>
  fetch(`${API}/feature-importance?top_n=${topN}`, { headers: getJsonHeaders() }).then(r => {
    if (!r.ok) throw new Error('Failed to fetch feature importance');
    return r.json();
  });

export const predictFromSensors = (sensors, lag1 = null, lag2 = null, lag3 = null) =>
  fetch(`${API}/predict`, {
    method: 'POST',
    headers: getJsonHeaders(),
    body: JSON.stringify({
      sensors,
      lag_flash_gc: lag1,
      lag2_flash_gc: lag2,
      lag3_flash_gc: lag3
    }),
  }).then(r => {
    if (!r.ok) throw new Error('Prediction failed');
    return r.json();
  });

export const predictFromWindow = (timestamp, lag1 = null, lag2 = null, lag3 = null) =>
  fetch(`${API}/predict/window`, {
    method: 'POST',
    headers: getJsonHeaders(),
    body: JSON.stringify({
      timestamp,
      lag_flash_gc: lag1,
      lag2_flash_gc: lag2,
      lag3_flash_gc: lag3
    }),
  }).then(r => {
    if (!r.ok) {
      return r.json().then(err => { throw new Error(err.detail || 'Prediction from window failed'); });
    }
    return r.json();
  });

export const uploadPredictBatch = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return fetch(`${API}/upload/predict-batch`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  }).then(r => {
    if (!r.ok) {
      return r.json().then(err => {
        throw new Error(err.detail || 'Batch prediction upload failed');
      });
    }
    return r.json();
  });
};

export const fetchHealth = () =>
  fetch(`${API}/health`).then(r => {
    if (!r.ok) return r.json().then(err => { throw new Error(err.error || 'Health check failed'); });
    return r.json();
  });

// ── Chat API ─────────────────────────────────────────────────────────────────

/** Get or create a stable guest session ID (persisted in localStorage). */
export const getSessionId = () => {
  let id = localStorage.getItem('hykero_session_id');
  if (!id) {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      id = crypto.randomUUID();
    } else {
      // Fallback high-entropy random string
      const array = new Uint32Array(4);
      if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        crypto.getRandomValues(array);
      } else {
        for (let i = 0; i < 4; i++) array[i] = Math.floor(Math.random() * 0x100000000);
      }
      id = Array.from(array, num => num.toString(36)).join('-');
    }
    localStorage.setItem('hykero_session_id', id);
  }
  return id;
};

export const fetchConversations = (sessionId) =>
  fetch(`${API}/chat/conversations?session_id=${encodeURIComponent(sessionId)}`, {
    headers: getJsonHeaders()
  }).then(r => {
    if (!r.ok) throw new Error('Failed to fetch conversations');
    return r.json();
  });

export const createConversation = (sessionId, title = 'New Conversation') =>
  fetch(`${API}/chat/conversations`, {
    method: 'POST',
    headers: getJsonHeaders(),
    body: JSON.stringify({ session_id: sessionId, title }),
  }).then(r => {
    if (!r.ok) throw new Error('Failed to create conversation');
    return r.json();
  });

export const deleteConversation = (conversationId) =>
  fetch(`${API}/chat/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: getJsonHeaders()
  }).then(r => {
    if (!r.ok) throw new Error('Failed to delete conversation');
    return r.json();
  });

export const fetchChatMessages = (conversationId) =>
  fetch(`${API}/chat/conversations/${conversationId}/messages`, {
    headers: getJsonHeaders()
  }).then(r => {
    if (!r.ok) throw new Error('Failed to fetch messages');
    return r.json();
  });

/**
 * Send a message and consume the SSE stream.
 * @param {number} conversationId
 * @param {string} message
 * @param {(chunk: string) => void} onChunk  Called with each text chunk
 * @param {() => void} onDone               Called when streaming is complete
 * @param {(err: string) => void} onError    Called on error
 * @returns {AbortController} controller to abort the request
 */
export const sendChatMessage = (conversationId, message, onChunk, onDone, onError) => {
  const controller = new AbortController();

  fetch(`${API}/chat/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: getJsonHeaders(),
    body: JSON.stringify({ message, stream: true }),
    signal: controller.signal,
  })
    .then(response => {
      if (!response.ok) throw new Error('Stream request failed');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) { onDone(); return; }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.chunk) onChunk(payload.chunk);
              if (payload.done) { onDone(); return; }
              if (payload.error) { onError(payload.error); return; }
            } catch { /* skip malformed lines */ }
          }
          return pump();
        });
      }
      return pump();
    })
    .catch(err => {
      if (err.name !== 'AbortError') onError(err.message);
    });

  return controller;
};
