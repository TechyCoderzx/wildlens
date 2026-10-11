const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function readResponse(response, fallback) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || `${fallback} (${response.status}).`)
  return data
}

async function readVoiceError(response, fallback) {
  const data = await response.json().catch(() => ({}))
  const failure = data?.error
  const message = typeof failure?.message === 'string' ? failure.message : `${fallback} (${response.status}).`
  const error = new Error(message)
  error.code = typeof failure?.code === 'string' ? failure.code : 'voice_unknown'
  error.stage = typeof failure?.stage === 'string' ? failure.stage : 'unknown'
  throw error
}

export async function getApiStatus() {
  const response = await fetch(`${API_BASE}/api/v1/vision/status`)
  return readResponse(response, 'Could not check local model status')
}

export async function analyzeImage(file) {
  const form = new FormData()
  form.append('image', file)
  const startedAt = performance.now()
  try {
    const response = await fetch(`${API_BASE}/api/v1/vision/analyze`, { method: 'POST', body: form })
    const result = await readResponse(response, 'Analysis failed')
    console.info(`wildlens_timing frontend_request_duration_ms=${(performance.now() - startedAt).toFixed(1)}`)
    return result
  } catch (error) {
    console.info(`wildlens_timing frontend_request_duration_ms=${(performance.now() - startedAt).toFixed(1)}`)
    throw error
  }
}

export async function followUp(previous, observation, image) {
  const form = new FormData()
  form.append('previous', JSON.stringify(previous))
  if (observation) form.append('observation', observation)
  if (image) form.append('image', image)
  const response = await fetch(`${API_BASE}/api/v1/vision/follow-up`, { method: 'POST', body: form })
  return readResponse(response, 'Follow-up failed')
}

export async function createVoiceSession(identification) {
  const response = await fetch(`${API_BASE}/api/v1/voice/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ identification: identification || null }),
  })
  if (!response.ok) await readVoiceError(response, 'Could not prepare the voice assistant')
  const session = await response.json().catch(() => null)
  if (!session?.conversation_token || typeof session.agent_prompt !== 'string') {
    const error = new Error('The voice service returned incomplete setup details.')
    error.code = 'voice_unknown'
    error.stage = 'agent_token'
    throw error
  }
  return session
}

export async function createSpeech(text) {
  const response = await fetch(`${API_BASE}/api/v1/voice/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!response.ok) {
    await readVoiceError(response, 'Speech playback failed')
  }
  return response.blob()
}
