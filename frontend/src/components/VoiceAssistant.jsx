import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ConversationProvider, useConversation } from '@elevenlabs/react'
import { createVoiceSession } from '../services/api.js'

const TRANSCRIPT_LIMIT = 40

function sanitizedSdkMessage(value) {
  const raw = typeof value === 'string' ? value : value?.message
  if (typeof raw !== 'string') return ''
  return raw
    .replace(/https?:\/\/\S+/gi, '[service URL]')
    .replace(/\bBearer\s+[^\s,;]+/gi, 'Bearer [redacted]')
    .replace(/(conversation[_ -]?token|signed[_ -]?url|xi-api-key)(\s*[=:]\s*)[^\s,;]+/gi, '$1$2[redacted]')
    .replace(/\b[A-Za-z0-9_-]{24,}\b/g, '[redacted]')
    .slice(0, 240)
}

function readableVoiceError(failure, fallback = 'The voice connection encountered a problem. Try again or continue with text.') {
  const message = typeof failure === 'string' ? failure : failure?.message
  const detail = `${failure?.name || ''} ${message || ''}`.toLowerCase()
  if (/notallowederror|permission denied|microphone access.*denied/.test(detail)) {
    return 'Microphone access was denied. Allow microphone access for this site, then try again.'
  }
  if (/notfounderror|no microphone|audio input.*not found/.test(detail)) {
    return 'No microphone is available. Connect or enable a microphone, then try again.'
  }
  if (/failed to fetch|networkerror|load failed|websocket.*failed|network request/.test(detail)) {
    return 'Could not reach the voice service. Check your internet connection and that the WildLens backend is available, then try again.'
  }
  return message || fallback
}

function VoiceAssistantDialog({ identification, onActiveChange, onClose }) {
  const [prompt, setPrompt] = useState('')
  const [pendingSession, setPendingSession] = useState(null)
  const [preparing, setPreparing] = useState(false)
  const [transcript, setTranscript] = useState([])
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const transcriptEndRef = useRef(null)
  const dialogRef = useRef(null)
  const closeRef = useRef(null)
  const sessionRequestRef = useRef(0)

  const rememberMessage = useCallback(event => {
    const text = typeof event?.message === 'string' ? event.message : typeof event?.text === 'string' ? event.text : ''
    if (!text.trim()) return
    const role = event?.source === 'user' || event?.role === 'user' ? 'You' : 'WildLens'
    const eventId = event?.event_id
    setTranscript(current => {
      const existing = eventId == null ? -1 : current.findIndex(line => line.eventId === eventId)
      const nextLine = { role, text: text.trim(), eventId }
      if (existing >= 0) return current.map((line, index) => index === existing ? nextLine : line)
      if (current.at(-1)?.role === role && current.at(-1)?.text === nextLine.text) return current
      return [...current, nextLine].slice(-TRANSCRIPT_LIMIT)
    })
  }, [])
  const rememberError = useCallback((value, context) => {
    const message = sanitizedSdkMessage(value)
    const contextReason = sanitizedSdkMessage(context?.reason)
    const detail = message || contextReason
    setError(detail ? `Voice connection error: ${detail}` : readableVoiceError(value))
  }, [])
  const rememberDisconnect = useCallback(details => {
    if (details?.reason === 'user') return
    if (details?.reason === 'agent') {
      const reason = sanitizedSdkMessage(details?.closeReason || details?.context?.reason)
      setError(reason ? `The WildLens agent ended the conversation: ${reason}` : 'The WildLens agent ended the conversation.')
      return
    }
    const reason = sanitizedSdkMessage(details?.message || details?.closeReason || details?.context?.reason)
    setError(reason ? `The voice session disconnected: ${reason}` : 'The voice session disconnected unexpectedly. Check microphone permission and network access, then try again.')
  }, [])
  const overrides = useMemo(() => prompt ? { agent: { prompt: { prompt } } } : undefined, [prompt])
  const conversation = useConversation({ overrides, onMessage: rememberMessage, onError: rememberError, onDisconnect: rememberDisconnect })
  const { startSession, endSession, sendUserMessage, sendUserActivity, setMuted } = conversation
  const status = conversation.status || 'disconnected'
  const isSpeaking = Boolean(conversation.isSpeaking)
  const isMuted = Boolean(conversation.isMuted)
  const isConnected = status === 'connected'

  useEffect(() => {
    onActiveChange?.(isConnected || status === 'connecting')
  }, [isConnected, onActiveChange, status])
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ block: 'nearest' })
  }, [transcript])
  useEffect(() => {
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    closeRef.current?.focus()
    function onKeyDown(event) {
      if (event.key === 'Escape') {
        event.preventDefault()
        void finishConversation()
      }
      if (event.key !== 'Tab') return
      const focusable = dialogRef.current?.querySelectorAll('button:not(:disabled), input:not(:disabled)')
      if (!focusable?.length) return
      if (event.shiftKey && document.activeElement === focusable[0]) {
        event.preventDefault()
        focusable[focusable.length - 1].focus()
      } else if (!event.shiftKey && document.activeElement === focusable[focusable.length - 1]) {
        event.preventDefault()
        focusable[0].focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [])

  useEffect(() => {
    if (!pendingSession || !prompt) return undefined
    const token = pendingSession
    setPendingSession(null)
    try { startSession({ conversationToken: token, connectionType: 'webrtc' }) }
    catch (failure) {
      const detail = sanitizedSdkMessage(failure)
      setError(detail ? `Voice connection error: ${detail}` : 'Could not start the voice conversation. Check microphone permission and try again.')
    }
    return undefined
  }, [pendingSession, prompt, startSession])

  async function beginConversation() {
    setError('')
    setTranscript([])
    setPreparing(true)
    const requestId = ++sessionRequestRef.current
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser does not support microphone access. Open WildLens in a modern browser or use the regular observation field.')
      setPreparing(false)
      return
    }
    try {
      const session = await createVoiceSession(identification)
      if (requestId !== sessionRequestRef.current) return
      if (!session?.conversation_token || !session?.agent_prompt) {
        throw new Error('The voice service returned incomplete setup details. Check the ElevenLabs Agent configuration and try again.')
      }
      setPrompt(session.agent_prompt)
      setPendingSession(session.conversation_token)
    } catch (failure) {
      if (requestId === sessionRequestRef.current) setError(readableVoiceError(failure, 'Could not prepare the voice assistant. You can try again.'))
    } finally {
      if (requestId === sessionRequestRef.current) setPreparing(false)
    }
  }

  async function finishConversation() {
    sessionRequestRef.current += 1
    try { await endSession() } catch { /* session may already have disconnected */ }
    setPendingSession(null)
    onClose?.()
  }

  function sendText(event) {
    event.preventDefault()
    const value = message.trim()
    if (!value || !isConnected) return
    try {
      sendUserMessage(value)
      setTranscript(current => [...current, { role: 'You', text: value }].slice(-TRANSCRIPT_LIMIT))
      setMessage('')
    } catch (failure) {
      setError(failure?.message || 'Your message could not be sent.')
    }
  }

  useEffect(() => () => {
    sessionRequestRef.current += 1
    try { endSession() } catch { /* no active session */ }
  }, [])

  return <div className="voice-overlay" onMouseDown={event => { if (event.target === event.currentTarget) void finishConversation() }}>
      <section className="voice-dialog" role="dialog" aria-modal="true" aria-labelledby="voice-title" ref={dialogRef}>
        <header className="voice-header">
          <div><p className="eyebrow">FIELD GUIDE, IN CONVERSATION</p><h2 id="voice-title">Talk to WildLens</h2><p>Ask about visible clues, uncertainty, or what to observe next.</p></div>
          <button ref={closeRef} className="voice-close" type="button" onClick={() => void finishConversation()} aria-label="Close voice assistant">×</button>
        </header>
        <div className={`voice-orb${isSpeaking ? ' is-speaking' : ''}${status === 'connecting' ? ' is-thinking' : ''}`} aria-hidden="true"><span /><span /><span /></div>
        <p className="voice-status" role="status">{preparing || status === 'connecting' ? 'Preparing a secure voice session…' : isSpeaking ? 'WildLens is speaking · you can interrupt by speaking' : isConnected ? (isMuted ? 'Microphone muted' : 'Listening for your question') : 'Conversation ended'}</p>
        {!isConnected && status !== 'connecting' && !preparing && <button className="voice-start-action" type="button" onClick={beginConversation}>{error ? 'Try again' : 'Start conversation'}</button>}
        {error && <p className="voice-error" role="alert">{error}</p>}
        <div className="voice-transcript" aria-label="Conversation transcript" aria-live="polite">
          {!transcript.length ? <p className="voice-empty">Your conversation will appear here. A transcript is also available while you speak.</p> : transcript.map((line, index) => <p className={`transcript-line${line.role === 'You' ? ' transcript-user' : ''}`} key={`${index}-${line.role}`}><strong>{line.role}</strong><span>{line.text}</span></p>)}
          <div ref={transcriptEndRef} />
        </div>
        <form className="voice-text-form" onSubmit={sendText}>
          <label className="sr-only" htmlFor="voice-text">Type a message instead</label>
          <input id="voice-text" value={message} maxLength={500} onChange={event => { setMessage(event.target.value); if (isConnected) sendUserActivity() }} placeholder="Type a question instead…" disabled={!isConnected} aria-describedby="voice-text-help" />
          <button type="submit" disabled={!isConnected || !message.trim()}>Send</button>
        </form>
        <span className="sr-only" id="voice-text-help">Text messages are available while the voice session is connected.</span>
        <div className="voice-controls">
          {isConnected && <>
            <button className="voice-secondary" type="button" onClick={() => setMuted(!isMuted)}>{isMuted ? 'Unmute microphone' : 'Mute microphone'}</button>
            <button className="voice-secondary" type="button" onClick={() => sendUserActivity()} aria-label="Pause the assistant briefly so you can speak">Pause &amp; speak</button>
            <button className="voice-end" type="button" onClick={() => void finishConversation()}>End conversation</button>
          </>}
          {status === 'connecting' && <button className="voice-end" type="button" onClick={() => void finishConversation()}>Cancel</button>}
          {!isConnected && status !== 'connecting' && <button className="voice-end" type="button" onClick={() => void finishConversation()}>Close</button>}
        </div>
        <p className="voice-privacy">Microphone audio is handled by ElevenLabs for this live conversation. WildLens does not store a recording.</p>
      </section>
    </div>
}

export default function VoiceAssistant(props) {
  return <ConversationProvider><VoiceAssistantDialog {...props} /></ConversationProvider>
}
