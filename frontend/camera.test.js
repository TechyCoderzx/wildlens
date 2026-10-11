import test from 'node:test'
import assert from 'node:assert/strict'
import {
  cameraErrorMessage,
  captureCameraPhoto,
  requestCameraStream,
  stopCameraStream,
} from './src/camera.js'

test('requests a silent camera stream with the preferred facing mode', async () => {
  let constraints
  const stream = { id: 'stream' }
  const mediaDevices = { getUserMedia: async value => { constraints = value; return stream } }

  assert.equal(await requestCameraStream(mediaDevices, 'environment'), stream)
  assert.deepEqual(constraints, { video: { facingMode: { ideal: 'environment' } }, audio: false })
})

test('requests the front facing mode when switching cameras', async () => {
  let constraints
  const mediaDevices = { getUserMedia: async value => { constraints = value; return {} } }

  await requestCameraStream(mediaDevices, 'user')
  assert.deepEqual(constraints.video.facingMode, { ideal: 'user' })
})

test('reports missing camera API support', async () => {
  await assert.rejects(requestCameraStream(null), /HTTPS or localhost/i)
})

test('stops every track when a stream is released', () => {
  let stopped = 0
  stopCameraStream({ getTracks: () => [{ stop: () => stopped++ }, { stop: () => stopped++ }] })
  assert.equal(stopped, 2)
})

test('captures the video frame into a JPEG File', async () => {
  const drawCalls = []
  const canvas = {
    getContext: () => ({ drawImage: (...args) => drawCalls.push(args) }),
    toBlob: callback => callback(new Blob(['captured frame'], { type: 'image/jpeg' })),
  }
  const video = { readyState: 2, videoWidth: 1280, videoHeight: 720 }
  const photo = await captureCameraPhoto(video, { createCanvas: () => canvas })

  assert.equal(photo.type, 'image/jpeg')
  assert.equal(photo.name.startsWith('wildlens-camera-'), true)
  assert.equal(canvas.width, 1280)
  assert.equal(canvas.height, 720)
  assert.equal(drawCalls.length, 1)
})

test('rejects captures before the live preview is ready', async () => {
  await assert.rejects(captureCameraPhoto({ readyState: 0, videoWidth: 0, videoHeight: 0 }), /preview is not ready/i)
})

test('returns useful permission and availability messages', () => {
  assert.match(cameraErrorMessage({ name: 'NotAllowedError' }), /access was denied/i)
  assert.match(cameraErrorMessage({ name: 'NotFoundError' }), /no camera was found/i)
  assert.match(cameraErrorMessage({ name: 'NotReadableError' }), /could not start/i)
})
