# WildLens architecture

## Request flow
1. React lets the user select or capture a photo.
2. The browser sends the photo as multipart form data to FastAPI.
3. FastAPI validates file type and size.
4. The vision service encodes the image and calls Ollama's local `/api/chat` endpoint.
5. Pydantic validates the model response into the API response schema.
6. React displays likely identification, evidence, uncertainty, facts, and a field observation prompt.

## Trust boundaries
- Model outputs are hypotheses, not verified taxonomy.
- The model must be allowed to return `unknown` or a low-confidence result.
- Facts are model-generated until an independently curated reference source is added.
- Raw images should not be sent to observability services by default.
- Never put provider credentials in frontend code.

## Planned increments
1. Verify Ollama version, disk space, model tag, image support, latency, and quality.
2. Add IndexedDB journal records and a service worker for the app shell.
3. Add Sentry tracing around the request and inference stages; test with a real failure and document the fix.
4. Add optional ElevenLabs narration through a backend-only integration.
5. Add tests for image validation, malformed model output, timeout, unavailable model, and uncertain identification.
