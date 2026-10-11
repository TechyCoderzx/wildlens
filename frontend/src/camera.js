export async function requestCameraStream(mediaDevices, facingMode = 'environment') {
  if (!mediaDevices?.getUserMedia) {
    throw new Error('Live camera access is not available here. Check that this page uses HTTPS or localhost, or choose a photo instead.')
  }

  return mediaDevices.getUserMedia({
    video: { facingMode: { ideal: facingMode } },
    audio: false,
  })
}

export function stopCameraStream(stream) {
  stream?.getTracks?.().forEach(track => {
    try { track.stop() } catch { /* Continue releasing the remaining tracks. */ }
  })
}

export function cameraErrorMessage(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError' || error?.name === 'SecurityError') {
    return 'Camera access was denied. Allow camera access in your browser settings, or choose a photo instead.'
  }
  if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') {
    return 'No camera was found on this device. Choose a photo instead.'
  }
  if (error?.name === 'NotReadableError' || error?.name === 'AbortError') {
    return 'The camera could not start. Close other apps using it, or choose a photo instead.'
  }
  if (error?.name === 'OverconstrainedError' || error?.name === 'ConstraintNotSatisfiedError') {
    return 'That camera is unavailable. Try switching cameras or choose a photo instead.'
  }
  return 'The camera could not start. Check browser permissions and secure access, or choose a photo instead.'
}

export function captureCameraPhoto(video, { createCanvas = () => document.createElement('canvas'), FileType = File } = {}) {
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return Promise.reject(new Error('The live preview is not ready yet. Please wait and try again.'))
  }

  const canvas = createCanvas()
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const context = canvas.getContext('2d')
  if (!context) return Promise.reject(new Error('The photo could not be captured. Please try again.'))

  try {
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
  } catch {
    return Promise.reject(new Error('The photo could not be captured. Please try again.'))
  }

  return new Promise((resolve, reject) => {
    if (!canvas.toBlob) {
      reject(new Error('Photo capture is not supported in this browser. Choose a photo instead.'))
      return
    }
    canvas.toBlob(blob => {
      if (!blob) {
        reject(new Error('The photo could not be captured. Please try again.'))
        return
      }
      resolve(new FileType([blob], `wildlens-camera-${Date.now()}.jpg`, {
        type: blob.type || 'image/jpeg',
        lastModified: Date.now(),
      }))
    }, 'image/jpeg', 0.9)
  })
}
