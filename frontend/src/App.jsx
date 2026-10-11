import { Component, lazy, Suspense, useEffect, useRef, useState } from 'react'
import { analyzeImage, createSpeech, followUp } from './services/api.js'
import { cameraErrorMessage, captureCameraPhoto, requestCameraStream, stopCameraStream } from './camera.js'
const VoiceAssistant = lazy(() => import('./components/VoiceAssistant.jsx'))

const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const ACCEPTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
const CARE_NOTE = 'AI suggestions are hypotheses. Verify with a trusted field guide and observe wildlife from a respectful distance.'
const DISCOVERY_CATEGORIES = [
  { title: 'Plants & fungi', copy: 'Leaves, lichens, moss, flowers, and the quiet forms growing underfoot.', image: '/images/moss-fungi.webp', fallback: '/images/moss-fungi.jpg', alt: 'A small mushroom rising from moss and dew', className: 'discovery-plants' },
  { title: 'Insects & small life', copy: 'Wings, markings, tiny visitors, and the habitats they move through.', image: '/images/insect-fern.webp', fallback: '/images/insect-fern.jpg', alt: 'A delicate insect perched on a fern frond', className: 'discovery-insects' },
  { title: 'Birds & wildlife', copy: 'Feathers, tracks, patterns, and other signs of nearby life.', image: '/images/bird-wildlife.webp', fallback: '/images/bird-wildlife.jpg', alt: 'A small bird perched on a mossy branch', className: 'discovery-wildlife' },
]
const FIELD_METHOD = [
  { number: '01', title: 'Evidence first', copy: 'See which visible clues support an identification, and what the image cannot tell us.' },
  { number: '02', title: 'Compare lookalikes', copy: 'Consider other plausible candidates and the details that could distinguish them.' },
  { number: '03', title: 'Observe next', copy: 'Collect another clue from a safe distance, then refine the field note with your observation.' },
]

class VoiceAssistantBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error) {
    console.error('WildLens voice assistant failed to render.', error)
  }

  render() {
    if (this.state.failed) {
      return <div className="voice-overlay"><section className="voice-dialog" role="dialog" aria-modal="true" aria-labelledby="voice-load-error-title">
        <header className="voice-header"><div><p className="eyebrow">FIELD GUIDE, IN CONVERSATION</p><h2 id="voice-load-error-title">Voice assistant unavailable</h2></div>
          <button className="voice-close" type="button" onClick={this.props.onClose} aria-label="Close voice assistant">×</button>
        </header>
        <p className="voice-error" role="alert">WildLens could not open the voice interface. Close this message and try again. Your photo identification and field notes are still available.</p>
        <div className="voice-controls"><button className="voice-end" type="button" onClick={this.props.onClose}>Return to WildLens</button></div>
      </section></div>
    }
    return this.props.children
  }
}

function validateImage(file) {
  if (!file) return 'Choose a photograph to continue.'
  const extension = `.${file.name.split('.').pop()?.toLowerCase()}`
  if (!ACCEPTED_TYPES.includes(file.type) && !ACCEPTED_EXTENSIONS.includes(extension)) {
    return 'Choose a JPEG, PNG, or WebP image.'
  }
  if (file.size > MAX_IMAGE_BYTES) return 'This image is larger than 10 MB. Choose a smaller file.'
  if (file.size === 0) return 'This image is empty. Choose another file.'
  return ''
}

function EvidenceList({ items }) {
  if (!items?.length) return <p className="empty-state">No specific visual clues were strong enough to report for this image.</p>
  return <ul className="evidence-list">{items.map((item, index) => <li className="evidence-item" key={`${item.feature}-${index}`}>
    <div className="evidence-topline"><span className="evidence-number">{String(index + 1).padStart(2, '0')}</span><span className={`strength strength-${item.support}`}>{item.support} support</span></div>
    <h3>{item.feature}</h3>
    <div className="evidence-explanation"><span>Why it matters</span><p>{item.inference}</p></div>
    {item.limitation && <p className="evidence-limit"><strong>Limitation</strong> {item.limitation}</p>}
    <span className="source-label">{item.source === 'user_reported' ? 'User reported' : item.source === 'follow_up_image' ? 'Follow-up photo' : 'Original photo'}</span>
  </li>)}</ul>
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [observation, setObservation] = useState('')
  const [followFile, setFollowFile] = useState(null)
  const [followPreview, setFollowPreview] = useState('')
  const [followBusy, setFollowBusy] = useState(false)
  const [followError, setFollowError] = useState('')
  const [changeNote, setChangeNote] = useState('')
  const [dragging, setDragging] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [cameraOpen, setCameraOpen] = useState(false)
  const [cameraStream, setCameraStream] = useState(null)
  const [cameraFacingMode, setCameraFacingMode] = useState('environment')
  const [cameraLoading, setCameraLoading] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [cameraNotice, setCameraNotice] = useState('')
  const [cameraPhoto, setCameraPhoto] = useState(null)
  const [cameraPhotoPreview, setCameraPhotoPreview] = useState('')
  const [voiceActive, setVoiceActive] = useState(false)
  const [voiceOpen, setVoiceOpen] = useState(false)
  const [speechUrl, setSpeechUrl] = useState('')
  const [speechPlaying, setSpeechPlaying] = useState(false)
  const [speechBusy, setSpeechBusy] = useState(false)
  const [speechError, setSpeechError] = useState('')
  const mainInputRef = useRef(null)
  const followInputRef = useRef(null)
  const cameraTriggerRef = useRef(null)
  const cameraCloseRef = useRef(null)
  const cameraDialogRef = useRef(null)
  const videoRef = useRef(null)
  const cameraStreamRef = useRef(null)
  const cameraRequestIdRef = useRef(0)
  const cameraOperationLock = useRef(false)
  const cameraSwitchLock = useRef(false)
  const captureLock = useRef(false)
  const closeCameraRef = useRef(null)
  const mainLock = useRef(false)
  const followLock = useRef(false)
  const speechAudioRef = useRef(null)
  const speechRequestRef = useRef(0)

  useEffect(() => {
    if (!speechUrl) return undefined
    return () => URL.revokeObjectURL(speechUrl)
  }, [speechUrl])

  useEffect(() => {
    if (!voiceActive) return
    speechRequestRef.current += 1
    speechAudioRef.current?.pause()
    setSpeechPlaying(false)
  }, [voiceActive])

  useEffect(() => () => {
    speechRequestRef.current += 1
    speechAudioRef.current?.pause()
    speechAudioRef.current = null
  }, [])

  useEffect(() => {
    speechRequestRef.current += 1
    speechAudioRef.current?.pause()
    speechAudioRef.current = null
    setSpeechPlaying(false)
    setSpeechUrl('')
    setSpeechError('')
  }, [result?.common_name, result?.summary])

  async function playSummary() {
    if (!result?.summary || voiceActive || speechBusy) return
    setSpeechError('')
    const requestId = ++speechRequestRef.current
    setSpeechBusy(true)
    try {
      let url = speechUrl
      if (!url) {
        const blob = await createSpeech(`${result.common_name}. ${result.summary}`)
        if (requestId !== speechRequestRef.current) return
        url = URL.createObjectURL(blob)
        setSpeechUrl(url)
      }
      if (speechAudioRef.current?.src === url) {
        await speechAudioRef.current.play()
      } else {
        speechAudioRef.current?.pause()
        const audio = new Audio(url)
        speechAudioRef.current = audio
        audio.onplay = () => setSpeechPlaying(true)
        audio.onpause = () => setSpeechPlaying(false)
        audio.onended = () => {
          setSpeechPlaying(false)
          URL.revokeObjectURL(url)
          if (speechAudioRef.current === audio) speechAudioRef.current = null
          setSpeechUrl(current => current === url ? '' : current)
        }
        await audio.play()
      }
    } catch (failure) {
      setSpeechError(failure.message || 'Could not play the spoken summary.')
      setSpeechPlaying(false)
    } finally {
      setSpeechBusy(false)
    }
  }

  async function replaySummary() {
    if (!speechUrl || voiceActive || speechBusy) return
    if (speechAudioRef.current) speechAudioRef.current.currentTime = 0
    await playSummary()
  }

  function stopSummary() {
    speechAudioRef.current?.pause()
    if (speechAudioRef.current) speechAudioRef.current.currentTime = 0
    setSpeechPlaying(false)
  }

  useEffect(() => {
    if (!preview) return undefined
    return () => URL.revokeObjectURL(preview)
  }, [preview])

  useEffect(() => {
    if (!followPreview) return undefined
    return () => URL.revokeObjectURL(followPreview)
  }, [followPreview])

  useEffect(() => {
    if (!cameraOpen || !cameraStream || !videoRef.current) return undefined
    const video = videoRef.current
    let active = true
    video.srcObject = cameraStream
    video.play().catch(() => {
      if (active) setCameraError('The camera preview could not start. Close other apps using the camera, or choose a photo instead.')
    })
    return () => {
      active = false
      if (video.srcObject === cameraStream) video.srcObject = null
    }
  }, [cameraOpen, cameraStream])

  useEffect(() => {
    if (!cameraPhoto) {
      setCameraPhotoPreview('')
      return undefined
    }
    const objectUrl = URL.createObjectURL(cameraPhoto)
    setCameraPhotoPreview(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [cameraPhoto])

  useEffect(() => {
    if (!cameraOpen) return undefined
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    cameraCloseRef.current?.focus()

    function onDialogKeyDown(event) {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeCameraRef.current?.()
      }
      if (event.key !== 'Tab') return
      const focusable = cameraDialogRef.current?.querySelectorAll('button:not(:disabled), video[tabindex]')
      if (!focusable?.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onDialogKeyDown)
    return () => {
      document.removeEventListener('keydown', onDialogKeyDown)
      document.body.style.overflow = previousOverflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [cameraOpen])

  useEffect(() => () => {
    cameraRequestIdRef.current += 1
    cameraOperationLock.current = false
    stopCameraStream(cameraStreamRef.current)
    cameraStreamRef.current = null
  }, [])

  closeCameraRef.current = closeCamera

  function stopActiveCamera() {
    stopCameraStream(cameraStreamRef.current)
    cameraStreamRef.current = null
    setCameraStream(null)
  }

  async function startCamera(facingMode, recoverWith = null) {
    if (cameraOperationLock.current) return
    cameraOperationLock.current = true
    const requestId = ++cameraRequestIdRef.current
    stopActiveCamera()
    setCameraLoading(true)
    setCameraError('')
    setCameraNotice('')

    const mediaDevices = typeof navigator !== 'undefined' ? navigator.mediaDevices : null
    try {
      const stream = await requestCameraStream(mediaDevices, facingMode)
      if (requestId !== cameraRequestIdRef.current) {
        stopCameraStream(stream)
        return
      }
      cameraStreamRef.current = stream
      setCameraStream(stream)
      setCameraFacingMode(facingMode)
      mediaDevices?.enumerateDevices?.().then(devices => {
        if (requestId !== cameraRequestIdRef.current) return
        const cameras = devices.filter(device => device.kind === 'videoinput')
        if (cameras.length === 1) setCameraNotice('Only one camera is available on this device.')
      }).catch(() => {})
    } catch (error) {
      if (requestId !== cameraRequestIdRef.current) return
      if (recoverWith) {
        try {
          const fallbackStream = await requestCameraStream(mediaDevices, recoverWith)
          if (requestId !== cameraRequestIdRef.current) {
            stopCameraStream(fallbackStream)
            return
          }
          cameraStreamRef.current = fallbackStream
          setCameraStream(fallbackStream)
          setCameraFacingMode(recoverWith)
          setCameraNotice(`Could not switch cameras. The ${recoverWith === 'user' ? 'front' : 'rear'} camera remains active.`)
        } catch {
          if (requestId === cameraRequestIdRef.current) setCameraError(cameraErrorMessage(error))
        }
      } else {
        setCameraError(cameraErrorMessage(error))
      }
    } finally {
      if (requestId === cameraRequestIdRef.current) {
        cameraOperationLock.current = false
        setCameraLoading(false)
      }
    }
  }

  function openCamera() {
    if (busy || cameraOperationLock.current) return
    setCameraPhoto(null)
    setCameraOpen(true)
    setCameraError('')
    setCameraNotice('')
    void startCamera('environment')
  }

  function closeCamera() {
    cameraRequestIdRef.current += 1
    cameraOperationLock.current = false
    stopActiveCamera()
    setCameraOpen(false)
    setCameraLoading(false)
    setCameraPhoto(null)
    setCameraError('')
    setCameraNotice('')
  }

  async function switchCamera() {
    if (cameraLoading || cameraSwitchLock.current || !cameraStreamRef.current) return
    cameraSwitchLock.current = true
    const requestId = cameraRequestIdRef.current
    const previousMode = cameraFacingMode
    const nextMode = previousMode === 'environment' ? 'user' : 'environment'
    try {
      const mediaDevices = navigator.mediaDevices
      const devices = await mediaDevices?.enumerateDevices?.()
      if (requestId !== cameraRequestIdRef.current) return
      const cameras = devices?.filter(device => device.kind === 'videoinput')
      if (cameras?.length === 1) {
        setCameraNotice('Only one camera is available on this device.')
        return
      }
      await startCamera(nextMode, previousMode)
    } catch {
      setCameraNotice('Could not check for another camera. The current preview is still available.')
    } finally {
      cameraSwitchLock.current = false
    }
  }

  async function capturePhoto() {
    if (captureLock.current || busy || !videoRef.current) return
    captureLock.current = true
    const requestId = cameraRequestIdRef.current
    setCameraError('')
    try {
      const image = await captureCameraPhoto(videoRef.current)
      if (requestId !== cameraRequestIdRef.current) return
      const validationError = validateImage(image)
      if (validationError) {
        setCameraError(validationError)
        return
      }
      setCameraPhoto(image)
      stopActiveCamera()
    } catch (error) {
      setCameraError(error.message || 'The photo could not be captured. Please try again.')
    } finally {
      captureLock.current = false
    }
  }

  function returnToUpload() {
    closeCamera()
    requestAnimationFrame(() => document.getElementById('identify')?.scrollIntoView({ behavior: 'smooth' }))
  }

  function useCameraPhoto() {
    if (!cameraPhoto || busy || mainLock.current) return
    const validationError = validateImage(cameraPhoto)
    if (validationError) {
      setCameraError(validationError)
      return
    }
    const image = cameraPhoto
    closeCamera()
    chooseFile(image)
    void identifyImage(image)
  }

  function chooseFile(nextFile) {
    setUploadError('')
    if (nextFile) {
      const problem = validateImage(nextFile)
      if (problem) {
        setUploadError(problem)
        if (mainInputRef.current) mainInputRef.current.value = ''
        return
      }
    }
    setResult(null)
    setError('')
    setChangeNote('')
    setFile(nextFile || null)
    setPreview(nextFile ? URL.createObjectURL(nextFile) : '')
    if (mainInputRef.current) mainInputRef.current.value = ''
    setObservation('')
    setFollowError('')
    setFollowFile(null)
    setFollowPreview('')
    if (followInputRef.current) followInputRef.current.value = ''
  }

  async function identifyImage(image) {
    if (!image || mainLock.current || busy) return
    mainLock.current = true
    setBusy(true)
    setError('')
    setResult(null)
    try {
      setResult(await analyzeImage(image))
    } catch (err) {
      setError(err.message || 'Analysis failed. Check your connection and retry.')
    } finally {
      mainLock.current = false
      setBusy(false)
    }
  }

  function chooseFollowFile(nextFile) {
    setFollowError('')
    if (nextFile) {
      const problem = validateImage(nextFile)
      if (problem) {
        setFollowError(problem)
        if (followInputRef.current) followInputRef.current.value = ''
        return
      }
    }
    setFollowFile(nextFile || null)
    setFollowPreview(nextFile ? URL.createObjectURL(nextFile) : '')
    if (followInputRef.current) followInputRef.current.value = ''
  }

  function onDrop(event) {
    event.preventDefault()
    setDragging(false)
    if (!busy) chooseFile(event.dataTransfer.files?.[0])
  }

  async function submit(event) {
    event.preventDefault()
    if (!file || mainLock.current || busy) return
    await identifyImage(file)
  }

  async function submitFollowup(event) {
    event.preventDefault()
    if (!result || followLock.current || followBusy || (!observation.trim() && !followFile)) return
    followLock.current = true
    setFollowBusy(true)
    setFollowError('')
    try {
      const update = await followUp(result, observation.trim(), followFile)
      setResult(update.identification)
      setChangeNote(update.what_changed || '')
      setObservation('')
      chooseFollowFile(null)
      if (followInputRef.current) followInputRef.current.value = ''
    } catch (err) {
      setFollowError(err.message || 'Follow-up failed. Your current assessment is still available.')
    } finally {
      followLock.current = false
      setFollowBusy(false)
    }
  }

  function useOutcome(outcome) {
    setObservation(current => current ? `${current}\n${outcome}` : outcome)
  }

  function closeMenu() {
    setMenuOpen(false)
  }

  return <main className="site-shell" id="top">
    <header className="site-header">
      <a className="brand" href="#top" aria-label="WildLens home" onClick={closeMenu}>
        <span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 40 40"><path d="M20 31V12m0 10c-8 0-11-5-11-11 7 0 11 4 11 11Zm0-5c0-7 4-11 11-11 0 7-4 11-11 11Z" /></svg></span>
        <span>WildLens<span className="brand-period">.</span></span>
      </a>
      <button className="menu-toggle" type="button" aria-expanded={menuOpen} aria-controls="primary-nav" onClick={() => setMenuOpen(open => !open)}>
        <span className="sr-only">{menuOpen ? 'Close navigation menu' : 'Open navigation menu'}</span><span className="menu-lines" aria-hidden="true" />
      </button>
      <nav className={`primary-nav${menuOpen ? ' is-open' : ''}`} id="primary-nav" aria-label="Main navigation">
        <a href="#discover" onClick={closeMenu}>Explore</a>
        <a href="#identify" onClick={closeMenu}>Identify</a>
        <a href="#about" onClick={closeMenu}>About</a>
        <a className="nav-cta" href="#identify" onClick={closeMenu}>Start exploring <span aria-hidden="true">↗</span></a>
      </nav>
    </header>

    <section className="hero" aria-labelledby="hero-title">
      <div className="hero-copy">
        <p className="eyebrow"><span className="eyebrow-rule" /> A FIELD GUIDE FOR THE CURIOUS</p>
        <h1 id="hero-title">The wild is<br /><em>closer than<br />you think.</em></h1>
        <p className="hero-intro">Your pocket field guide to the natural world. Start with a photograph, follow the evidence, and learn what to observe next.</p>
        <div className="hero-actions">
        <button className="voice-launch" type="button" onClick={() => setVoiceOpen(true)} aria-haspopup="dialog" aria-expanded={voiceOpen} disabled={voiceOpen}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"/><path d="M19 11.5a7 7 0 0 1-14 0M12 18.5V22m-4 0h8"/></svg>
          <span>Talk to WildLens</span>
        </button>
        <a className="hero-action" href="#identify">Start with a photograph <span aria-hidden="true">↗</span></a>
        </div>
        {voiceOpen && <VoiceAssistantBoundary onClose={() => setVoiceOpen(false)}><Suspense fallback={<div className="voice-overlay"><p className="voice-status" role="status">Opening WildLens voice assistant…</p></div>}><VoiceAssistant identification={result} onActiveChange={setVoiceActive} onClose={() => setVoiceOpen(false)} /></Suspense></VoiceAssistantBoundary>}
        <p className="hero-note"><span className="note-star" aria-hidden="true">✳</span> Curiosity, grounded in what you can see.</p>
      </div>
      <figure className="hero-visual">
        <button ref={cameraTriggerRef} className="camera-access" type="button" onClick={openCamera} disabled={busy}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7.5h3l1.4-2h7.2l1.4 2h3A1.5 1.5 0 0 1 21.5 9v9a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 18V9A1.5 1.5 0 0 1 4 7.5Z"/><circle cx="12" cy="13" r="3.5"/></svg>
          <span>Identify with camera</span><span className="camera-access-arrow" aria-hidden="true">↗</span>
        </button>
        <picture><source srcSet="/images/forest-floor-hero.webp" type="image/webp" /><img src="/images/forest-floor-hero.jpg" alt="Moss, tiny fungi, and fallen leaves in a miniature forest-floor ecosystem" width="1600" height="1050" fetchPriority="high" decoding="async" /></picture>
        <figcaption><span>THE FOREST FLOOR, UP CLOSE</span><span>A world within a world.</span></figcaption>
        <span className="image-index">FIELD STUDY <i>·</i> 001</span>
      </figure>
      <div className="hero-footnote"><span>FIELD NOTES FOR THE LIVING WORLD</span><span>Look, learn, leave no trace.</span></div>
    </section>

    <section className="discovery-section" id="discover" aria-labelledby="discover-heading">
      <div className="discovery-heading">
        <div><p className="eyebrow"><span className="eyebrow-rule" /> START WHERE CURIOSITY LEADS</p><h2 id="discover-heading">There is more than one way<br /><em>into the wild.</em></h2></div>
        <p>From the forest floor to the open sky, follow a detail that catches your eye. Choose a subject to guide your first observation.</p>
      </div>
      <div className="discovery-grid">{DISCOVERY_CATEGORIES.map((category, index) => <article className={`discovery-card ${category.className}`} key={category.title}>
        <a className="discovery-image" href="#identify" aria-label={`Explore ${category.title} with a photograph`}>
          <picture><source srcSet={category.image} type="image/webp" /><img src={category.fallback} alt={category.alt} width={index === 2 ? 1600 : 900} height={index === 2 ? 1067 : 1100} loading="lazy" decoding="async" /></picture>
          <span className="discovery-number">0{index + 1}</span>
        </a>
        <div className="discovery-copy"><p className="eyebrow">A PLACE TO BEGIN</p><h3>{category.title}</h3><p>{category.copy}</p><a href="#identify">Explore with a photo <span aria-hidden="true">↗</span></a></div>
      </article>)}</div>
      <p className="discovery-footnote">A starting point, not a filter. WildLens considers the natural subject in the photograph you provide.</p>
    </section>

    <section className="method-section" aria-labelledby="method-heading">
      <div className="method-intro"><p className="eyebrow"><span className="eyebrow-rule" /> A CAREFUL WAY OF LOOKING</p><h2 id="method-heading">Notice.<br /><em>Compare. Return.</em></h2><p>WildLens turns a photograph into a small field investigation, keeping the evidence and its uncertainties in view.</p></div>
      <div className="method-steps">{FIELD_METHOD.map(step => <article className="method-step" key={step.number}>
        <span className="method-number">{step.number}</span><h3>{step.title}</h3><p>{step.copy}</p>
      </article>)}</div>
    </section>

    <section className="identify-section" id="identify" aria-labelledby="identify-title">
      <div className="section-intro">
        <div><p className="eyebrow"><span className="eyebrow-rule" /> YOUR NEXT DISCOVERY</p><h2 id="identify-title">What did you find?</h2></div>
        <p>Every good identification begins with a closer look. Add a photograph to start your field note.</p>
      </div>
      <div className="identify-layout">
        <form className="upload-panel" onSubmit={submit}>
          <div className="upload-heading"><span className="field-number">FIELD NOTE <strong>001</strong></span><span className="step-label">01 — IDENTIFY</span></div>
          <div className={`dropzone${dragging ? ' is-dragging' : ''}${preview ? ' has-preview' : ''}`} onDragEnter={event => { event.preventDefault(); setDragging(true) }} onDragOver={event => event.preventDefault()} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false) }} onDrop={onDrop}>
            <input ref={mainInputRef} id="first-photo" className="file-input" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" capture="environment" aria-describedby="upload-help upload-error" onChange={event => chooseFile(event.target.files?.[0])} />
            {preview ? <>
              <img className="upload-preview" src={preview} alt="Selected nature photograph preview" />
              <div className="preview-caption"><span>{file?.name}</span><button className="inline-action" type="button" onClick={() => mainInputRef.current?.click()}>Replace photograph</button></div>
            </> : <div className="upload-prompt">
              <span className="upload-symbol" aria-hidden="true"><svg viewBox="0 0 32 32"><path d="M16 22V7m0 0-5 5m5-5 5 5M6 20v6h20v-6" /></svg></span>
              <strong>Drop a photograph here</strong><span className="upload-or">or</span>
              <label className="browse-button" htmlFor="first-photo">Browse files</label>
              <span className="upload-formats" id="upload-help">JPEG, PNG or WebP <i>·</i> up to 10 MB</span>
            </div>}
          </div>
          {uploadError && <p className="error-message" id="upload-error" role="alert">{uploadError}</p>}
          <div className="upload-actions">
            {preview ? <button className="secondary-button" type="button" onClick={() => chooseFile(null)} disabled={busy}>Remove photo</button> : <span className="action-hint">Your image stays here until you choose to identify it.</span>}
            <button className="primary-button" type="submit" disabled={!file || !!uploadError || busy}>{busy ? <><span className="spinner" aria-hidden="true" /> Reading the details…</> : error ? 'Try identification again' : <>Identify this <span aria-hidden="true">↗</span></>}</button>
          </div>
          {busy && <p className="loading-message" role="status">WildLens is examining the photo and checking the result. Your field note will appear when validation is complete.</p>}
          {error && <p className="error-message" role="alert">{error}</p>}
        </form>
        <aside className="ethics-note" id="about">
          <p className="eyebrow">A NOTE ON CARE</p>
          <span className="ethics-flower" aria-hidden="true">✳</span>
          <h3>Observe first.<br /><em>Leave no trace.</em></h3>
          <p>Keep a respectful distance from wildlife. Never touch an unknown plant or insect, and leave habitats as you found them.</p>
          <div className="ethics-divider" />
          <p className="privacy-note">Your photo is sent to the WildLens service for analysis. It is not stored by this interface.</p>
        </aside>
      </div>
    </section>

    {result && <section className="results" id="field-guide" aria-live="polite" aria-labelledby="result-heading">
      <div className="results-masthead"><span className="field-number">FIELD NOTE <strong>001</strong></span><span className="results-kicker">A CLOSER READING</span></div>
      <article className="identification-card">
        <div className="identification-copy">
          <p className="eyebrow"><span className="eyebrow-rule" /> {result.status === 'identified' ? 'A LIKELY IDENTIFICATION' : result.status === 'unknown' ? 'IDENTIFICATION OUT OF REACH' : 'AN OPEN QUESTION'}</p>
          <h2 id="result-heading">{result.common_name}</h2>
          {result.scientific_name && <p className="scientific-name">{result.scientific_name}</p>}
          <div className="result-labels"><span className={`status-label status-${result.status}`}>{result.status}</span><span className={`confidence-label confidence-${result.confidence}`}>{result.confidence} confidence</span></div>
          {result.summary && <><p className="identification-summary">{result.summary}</p><div className="summary-audio-controls"><button className="listen-summary" type="button" onClick={speechPlaying ? () => speechAudioRef.current?.pause() : playSummary} disabled={speechBusy || voiceActive}>{speechBusy ? 'Preparing audio…' : speechPlaying ? 'Pause summary' : speechUrl ? 'Resume summary' : 'Listen to summary'}</button>{speechUrl && <><button className="stop-summary" type="button" onClick={replaySummary} disabled={voiceActive || speechBusy}>Replay</button><button className="stop-summary" type="button" onClick={stopSummary}>Stop</button></>}{speechError && <span className="speech-error" role="alert">{speechError}</span>}</div></>}
          <div className={`evidence-quality quality-${result.evidence_assessment?.quality || 'insufficient'}`}>
            <span className="quality-symbol" aria-hidden="true">{result.evidence_assessment?.quality === 'strong' ? '●' : result.evidence_assessment?.quality === 'partial' ? '◐' : '○'}</span>
            <div><strong>{({ strong: 'Strong supporting evidence', partial: 'Partial evidence', insufficient: 'Insufficient evidence' })[result.evidence_assessment?.quality] || 'Evidence still developing'}</strong>
              {result.evidence_assessment?.rationale && <p>{result.evidence_assessment.rationale}</p>}</div>
          </div>
          <p className="hypothesis-note">{result.disclaimer || CARE_NOTE}</p>
        </div>
        {preview && <figure className="identified-photo"><img src={preview} alt="Photograph submitted for this identification" /><figcaption><span>THE ORIGINAL OBSERVATION</span><span>{file?.name || 'Submitted photograph'}</span></figcaption></figure>}
      </article>

      <section className="result-section" aria-labelledby="evidence-heading">
        <div className="result-section-heading"><div><p className="eyebrow">OBSERVATION, THEN INFERENCE</p><h2 id="evidence-heading">Evidence in your photo</h2></div><span className="section-index">01</span></div>
        {result.evidence_details?.length ? <EvidenceList items={result.evidence_details} /> : result.visible_evidence?.length ? <ul className="visible-clues">{result.visible_evidence.map((item, index) => <li key={index}><span className="clue-dot" />{item}<span className="source-label">Original photo</span></li>)}</ul> : <p className="empty-state">The image did not provide specific visible clues to list. The identification remains a hypothesis.</p>}
        {!!result.evidence_assessment?.supporting_points?.length && <p className="supporting-points"><strong>Clues that matter</strong> {result.evidence_assessment.supporting_points.join(' · ')}</p>}
        {!!result.evidence_assessment?.limitations?.length && <p className="limitations"><strong>Still unknown</strong> {result.evidence_assessment.limitations.join(' · ')}</p>}
      </section>

      <section className="result-section lookalike-section" aria-labelledby="lookalike-heading">
        <div className="result-section-heading"><div><p className="eyebrow">COMPARE, DON’T ASSUME</p><h2 id="lookalike-heading">Lookalike Detective</h2></div><span className="section-index">02</span></div>
        {!result.candidates?.length ? <p className="empty-state">There is not enough evidence to compare credible alternatives. Another observation may help.</p> : <>
          <p className="section-deck">A useful possibility is one that can be tested against what is visible.</p>
          <div className="candidate-list">{result.candidates.slice(0, 3).map((candidate, index) => <article className="candidate" key={`${candidate.common_name}-${index}`}>
            <div className="candidate-number">0{index + 1}</div><div className="candidate-content">
              <p className="eyebrow">{index === 0 ? 'LEADING POSSIBILITY' : `ALTERNATIVE ${index}`}</p><h3>{candidate.common_name}</h3>
              {candidate.scientific_name && <p className="scientific-name">{candidate.scientific_name}</p>}
              <p className="candidate-reason">{candidate.match_explanation}</p>
              {!!candidate.supporting_features?.length && <p className="candidate-detail"><strong>In its favour</strong> {candidate.supporting_features.join(' · ')}</p>}
              {!!candidate.distinguishing_features?.length && <ul className="distinguishing-list">{candidate.distinguishing_features.map((feature, featureIndex) => <li key={`${feature.feature}-${featureIndex}`}><strong>{feature.feature}</strong><span>{feature.observation || 'Not visible'} — {feature.interpretation}</span></li>)}</ul>}
              {!!candidate.uncertainties?.length && <p className="candidate-detail candidate-uncertainty"><strong>Uncertain</strong> {candidate.uncertainties.join(' · ')}</p>}
            </div>
          </article>)}</div>
          {result.comparison_summary && <p className="comparison-note"><span className="comparison-mark" aria-hidden="true">↳</span>{result.comparison_summary}</p>}
        </>}
      </section>

      <section className="observe-section" aria-labelledby="observe-heading">
        <div className="observe-copy">
          <p className="eyebrow">A SMALL NEXT STEP <span className="section-index">03</span></p>
          <h2 id="observe-heading">Observe next</h2>
          <p className="observe-intro">Let the next clue come from the world, at its own pace.</p>
          {result.observation_tasks?.length > 0 ? <div className="task-list">{result.observation_tasks.slice(0, 3).map((task, index) => <article className="observation-task" key={`${task.title}-${index}`}>
            <div className="task-heading"><span>{String(index + 1).padStart(2, '0')}</span><h3>{task.title}</h3></div>
            <p><strong>Look for</strong> {task.what_to_look_for}</p><p><strong>Why it matters</strong> {task.why_it_matters}</p><p><strong>Keep it safe</strong> {task.how_safely}</p>
            {!!task.possible_outcomes?.length && <div className="outcome-actions"><span>If you noticed:</span>{task.possible_outcomes.map((outcome, outcomeIndex) => <button className="outcome-button" type="button" key={outcomeIndex} onClick={() => useOutcome(outcome)}>{outcome} <span aria-hidden="true">+</span></button>)}</div>}
          </article>)}</div> : result.observe_next ? <p className="observe-recommendation">{result.observe_next}</p> : <p className="empty-state">No specific observation was suggested for this field note.</p>}
          {!!result.interesting_facts?.length && <div className="interesting-facts"><h3>From the field guide</h3><ul>{result.interesting_facts.map((fact, index) => <li key={index}>{fact}</li>)}</ul></div>}
          <p className="care-reminder">{CARE_NOTE}</p>
        </div>
        <div className="followup-panel">
          <p className="field-number">ADD A CLUE <strong>↗</strong></p><h3>What did you notice?</h3>
          <p className="followup-intro">A detail from your own observation can help refine the field note. It will be kept distinct from photo evidence.</p>
          {changeNote && <p className="change-note" role="status"><strong>Field note updated</strong>{changeNote}</p>}
          <form onSubmit={submitFollowup}>
            <label className="field-label" htmlFor="observation">Your observation</label>
            <textarea id="observation" rows="4" maxLength="800" value={observation} onChange={event => setObservation(event.target.value)} placeholder="For example: I can see alternating dark and light bands on the tail." aria-describedby="observation-count" />
            <div className="textarea-meta"><span>Only include what you actually observed.</span><span id="observation-count">{observation.length}/800</span></div>
            <label className="field-label follow-photo-label" htmlFor="follow-photo">Add a second photograph <span>(optional)</span></label>
            <input ref={followInputRef} className="follow-file-input" id="follow-photo" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" aria-describedby="follow-error" onChange={event => chooseFollowFile(event.target.files?.[0])} />
            {followPreview && <div className="follow-preview"><img src={followPreview} alt="Follow-up photograph preview" /><div><span>{followFile?.name}</span><button className="inline-action" type="button" onClick={() => { chooseFollowFile(null); if (followInputRef.current) followInputRef.current.value = '' }}>Remove image</button></div></div>}
            <label className="follow-browse" htmlFor="follow-photo">{followFile ? 'Replace photograph' : 'Choose a photograph'} <span aria-hidden="true">＋</span></label>
            {followError && <p className="error-message" id="follow-error" role="alert">{followError}</p>}
            <button className="primary-button follow-submit" type="submit" disabled={followBusy || (!observation.trim() && !followFile)}>{followBusy ? <><span className="spinner" aria-hidden="true" /> Re-reading the clues…</> : <>Update field note <span aria-hidden="true">↗</span></>}</button>
            {followBusy && <p className="loading-message" role="status">Considering the new clue with the earlier evidence. This may take a little while.</p>}
          </form>
        </div>
      </section>
    </section>}

    <footer className="site-footer"><a className="brand footer-brand" href="#top"><span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 40 40"><path d="M20 31V12m0 10c-8 0-11-5-11-11 7 0 11 4 11 11Zm0-5c0-7 4-11 11-11 0 7-4 11-11 11Z" /></svg></span><span>WildLens<span className="brand-period">.</span></span></a><p>Made for the curious, outdoors.</p><a href="#top" className="back-to-top">Back to top ↑</a></footer>
    {cameraOpen && <div className="camera-overlay" onMouseDown={event => { if (event.target === event.currentTarget) closeCamera() }}>
      <section className="camera-dialog" role="dialog" aria-modal="true" aria-labelledby="camera-title" aria-describedby="camera-description" ref={cameraDialogRef} tabIndex="-1">
        <header className="camera-dialog-header">
          <div><p className="camera-kicker">WILDLENS FIELD CAMERA</p><h2 id="camera-title">Take a closer look</h2><p id="camera-description">Frame a natural detail, then capture it for identification.</p></div>
          <button ref={cameraCloseRef} className="camera-close" type="button" onClick={closeCamera} aria-label="Close camera">×</button>
        </header>

        <div className="camera-preview-wrap">
          {cameraPhoto
            ? <img className="camera-photo-preview" src={cameraPhotoPreview} alt="Captured photo preview" />
            : <video ref={videoRef} className="camera-video" autoPlay muted playsInline aria-label="Live camera preview" />}
          {cameraLoading && !cameraPhoto && <p className="camera-loading" role="status">Connecting to camera…</p>}
          {!cameraStream && !cameraPhoto && !cameraLoading && !cameraError && <p className="camera-loading" role="status">Camera preview is unavailable.</p>}
        </div>

        {cameraError && <p className="camera-error" role="alert">{cameraError}</p>}
        {cameraNotice && <p className="camera-notice" role="status">{cameraNotice}</p>}

        {cameraPhoto ? <div className="camera-review-actions">
          <button className="camera-secondary-action" type="button" onClick={() => { setCameraPhoto(null); void startCamera(cameraFacingMode) }} disabled={cameraLoading}>Retake</button>
          <button className="camera-primary-action" type="button" onClick={useCameraPhoto} disabled={busy}>Use photo</button>
        </div> : <div className="camera-controls">
          <button className="camera-switch-action" type="button" onClick={switchCamera} disabled={!cameraStream || cameraLoading}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 9h1.6l1-1.5h3.2l1 1.5h1.2A1.2 1.2 0 0 1 16.2 10v3.2a1.2 1.2 0 0 1-1.2 1.2H8.2A1.2 1.2 0 0 1 7 13.2Z"/><circle cx="11.5" cy="11.6" r="1.6"/><path d="M5 6a8.5 8.5 0 0 1 13-1l1 1m0-3v3h-3M19 18a8.5 8.5 0 0 1-13 1l-1-1m0 3v-3h3"/></svg>
            <span>Switch camera</span>
          </button>
          <button className="camera-capture-action" type="button" onClick={capturePhoto} disabled={!cameraStream || cameraLoading || busy} aria-label="Capture photo"><span /></button>
          <span className="camera-control-spacer" aria-hidden="true" />
        </div>}

        <button className="camera-upload-fallback" type="button" onClick={returnToUpload}>Use photo upload instead</button>
      </section>
    </div>}
  </main>
}

export default App
