# WildLens voice assistant setup

WildLens uses ElevenLabs Agents for live speech and ElevenLabs Text to Speech for reading an identification summary. The API key stays in the FastAPI backend. The frontend receives only a short lived conversation token.

## Backend settings

Add these values to `backend/.env` (or the backend process environment):

```dotenv
ELEVENLABS_API_KEY=your_server_side_api_key
ELEVENLABS_AGENT_ID=your_agent_id
ELEVENLABS_VOICE_ID=your_tts_voice_id
WILDLENS_CORS_ORIGINS=https://your-wildlens-frontend.example
```

The Agent ID is required for Talk to WildLens. The Voice ID is only needed for the separate Listen to summary control. Keep the API key out of frontend `.env` files and source control. Restart the backend after setting values. If a value is missing, the related route returns a configuration error instead of faking a session or audio.

## ElevenLabs dashboard

1. Create and configure an ElevenLabs Agent with a voice and conversational model, then copy its Agent ID into `ELEVENLABS_AGENT_ID`.
2. Keep the Agent private/authenticated so WildLens can request its WebRTC conversation token from the backend.
3. Enable Agent prompt overrides in the Agent security settings. WildLens sends a per-session prompt containing the validated identification, concise visual evidence, alternatives, and observation suggestions. The prompt instructs the Agent to preserve uncertainty and distinguish user reports from image evidence.
4. Enable voice interruptions / barge-in in the Agent turn-taking settings so a user can speak while WildLens is talking. The in-app “Pause & speak” control also signals user activity.
5. Enable the user transcription and Agent response client events in the Agent's advanced event settings if they are not enabled by default; the UI transcript depends on those events.
6. Choose a voice and copy its Voice ID into `ELEVENLABS_VOICE_ID` for summary playback.

The account must permit Agent conversations, WebRTC conversation tokens, and Text to Speech for the selected voice. Availability, credits, and usage limits are controlled by the ElevenLabs account. `WILDLENS_CORS_ORIGINS` is a comma-separated list; local Vite origins remain allowed for development.

## Browser access

The microphone permission is requested only after the user opens Talk to WildLens. Browsers require a secure context for microphone access: use HTTPS in deployment or localhost for local development. A phone cannot use the development computer's `localhost`; deploy to HTTPS or use a secure tunnel and point `VITE_API_BASE_URL` at the reachable backend.

No microphone recording is saved by the WildLens frontend. Conversation transcription and processing are handled by ElevenLabs according to the Agent/account configuration.
