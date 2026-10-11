WildLens 🌿
Discover the wild. Understand what you see. Then look closer.
WildLens is an AI-assisted nature field guide built to help people explore the living world around them. Upload or capture a photograph of an organism and WildLens returns a structured identification hypothesis, visible clues, possible lookalikes, and ideas for what to observe next. You can also ask follow-up questions or talk to the WildLens field guide using voice.
The idea is simple: use AI to make a moment outdoors more interesting, not to keep someone staring at a screen for longer than necessary.
Important: WildLens is an educational exploration tool. Its results are model-generated hypotheses, not verified scientific identifications. Do not use it to decide whether a plant, mushroom, animal, or other organism is safe to touch, eat, or handle.

Live project
- Website: https://wildlens-frontend.onrender.com
- Backend health: https://wildlens-k0ey.onrender.com/health
- Interactive API documentation: https://wildlens-k0ey.onrender.com/docs
- Source code: https://github.com/TechyCoderzX/wildlens
The frontend and backend are deployed as separate Render services. The backend may take longer to respond after inactivity on a free hosting instance, and model response times depend on the upstream API.
What WildLens does
- Photo-based nature identification: Analyze uploaded photographs of birds, animals, plants, insects, fungi, and microorganisms when the image contains useful visual evidence.
- Evidence-first results: Show visible features that support an identification rather than presenting a name with no explanation.
- Lookalike comparisons: Include plausible alternatives when the evidence supports them, with features that may help distinguish one candidate from another.
- Uncertainty-aware responses: The model can return an uncertain or unknown result instead of being forced to claim an exact species.
- Observe Next: Suggest a small observation task that encourages the user to notice details in the real world.
- Follow-up questions: Add an observation or another photograph and ask for a closer explanation of the previous result.
- Camera support: Capture a photograph in the browser where device and browser permissions allow it.
- Talk to WildLens: Have a spoken conversation with the field-guide agent powered by ElevenLabs Agents.
- Listen to a summary: Play an audio version of a result using ElevenLabs Text to Speech.
Not every organism can be identified reliably from every photo. Similar-looking species, tiny or obscured subjects, unusual lighting, and low-resolution images can all reduce the quality of a result.
Technology stack
Layer	Technology	Role
Frontend	React, Vite, JavaScript, CSS	User interface, photo selection/capture, result display, follow-up and voice controls
Backend	Python, FastAPI, Uvicorn	API routes, image validation and preparation, provider requests, response validation
Image processing	Pillow	Read, orient, resize, and prepare image bytes for model input
Main vision model	Gemma 4 through the Google Gemini API	Interpret the photo and generate a structured nature-identification response
Response validation	Pydantic	Enforce the expected response structure and types before results reach the UI
Live voice	ElevenLabs Agents	Conversational voice session for asking questions about the identification
Spoken summary	ElevenLabs Text to Speech	Generate audio for the summary separately from the live agent conversation
Hosting	Render	Serve the static frontend and run the FastAPI backend


WildLens uses a hosted model API for photo analysis; image analysis is not fully offline. The browser sends the selected image to the WildLens backend, and the backend sends the prepared image to Google’s Gemini API for Gemma inference. The ElevenLabs features also use their hosted services.
How the Gemma 4 API is used
Gemma 4 is the core image-understanding component of WildLens. It is not used to train a new model in this project; WildLens sends each photo to the configured model for inference.
1. The image request reaches FastAPI
The React frontend sends an image to the POST /api/v1/vision/analyze route as multipart form data using the image field. The backend checks and reads the uploaded image before starting model inference.
2. The image is prepared for inference
The vision service uses Pillow to read image dimensions and orientation. Images with a long edge greater than 1280 pixels are reduced while preserving their aspect ratio. Images may be converted to JPEG at quality 82 to keep the input manageable; an already suitable JPEG can pass through without being re-encoded. If preparation fails, the service logs the error type rather than logging the image contents.
This step reduces unnecessary image payload size. It does not guarantee a particular model response time, because inference latency also depends on the provider, network and model workload.
3. The backend calls Google's Gemini API with Gemma 4
The current default model identifier is:
gemma-4-26b-a4b-it
The model name can be overridden with GEMMA_MODEL. The backend calls the Google Generative Language API generateContent endpoint for that model. The request contains:
- A system instruction describing the identification boundaries and expected response format.
- The image as inline image data, with its MIME type and base64-encoded content.
- A concise user prompt asking for a complete identification result.
- A generation configuration with low temperature (0.1), JSON MIME output (application/json), and a maximum output of 2048 tokens.
The API key is supplied by the backend in the request header. It is never meant to be placed in the React bundle or sent to the browser.
The request uses the JSON REST API directly with httpx, rather than requiring a local model server. That makes deployment straightforward, but it does mean image analysis requires network access and valid API credentials/quota.
4. The prompt is designed around visible evidence
The prompt asks Gemma to identify organisms from visible evidence only and to keep uncertainty intact. It covers animals (including birds), plants, fungi and microorganisms (including bacteria). It also defines what to return when an organism cannot be identified or the photograph is outside the application's nature-identification scope.
The structured response includes fields such as:
- status, common_name, scientific_name, and confidence
- summary and visible_evidence
- evidence_details and evidence_assessment
- candidates and optional comparison information
- interesting_facts, observation_tasks, and observe_next
- disclaimer
The model is instructed to avoid extra JSON keys, use the expected nested structures, keep the output concise, and ensure the first candidate—when present—matches the primary common name. The prompt also limits the number of alternatives and shortens supporting details to reduce the chance of an oversized response.
5. The response is parsed and validated
The backend extracts a JSON object, then validates it with the existing Pydantic IdentificationResult schema. A result is accepted only if the required fields, data types, and allowed values are valid. This is important because syntactically plausible text is not necessarily a valid API response.
If Gemma returns malformed JSON, an incompatible structure, or a truncated response, WildLens rejects the response instead of inventing missing information. The backend logs safe diagnostic information such as processing timings, token counts when supplied by the provider, and validation paths/types. It does not need to log the API key or complete model output to diagnose these failures.
Why Gemma 4?
WildLens needs a model that can interpret an image and return a structured explanation, not just a single class label. Gemma 4 is used as the vision and reasoning component that proposes an identification, relates it to visible clues, and generates follow-up observations. The structured result lets the frontend present the answer as a field-guide entry rather than dumping a block of model text onto the page.
This integration is inference-based: no custom training dataset or fine-tuning step is required by the current application. Model quality still varies by subject and photograph, and the application deliberately treats the result as a hypothesis.
How ElevenLabs is used
ElevenLabs provides two distinct voice features. They have separate flows and configuration, even though both are available from the WildLens interface.
1. Talk to WildLens — live conversational voice
The live conversation uses ElevenLabs Agents. The API key is held by the FastAPI backend, not exposed to the frontend.
When a user opens the live voice panel, the frontend requests a session from POST /api/v1/voice/session. The backend:
1. Checks that the required ElevenLabs API key and Agent ID are configured.
2. Requests a short-lived conversation token from ElevenLabs.
3. Builds a session prompt from the current validated identification, visible evidence, alternatives and observation suggestions.
4. Returns the temporary token and agent prompt to the frontend.
The browser uses the short-lived token to establish the agent conversation using the configured browser voice flow (WebRTC). The token is not the long-lived API key. The agent's prompt tells it to be a concise field guide, preserve uncertainty, and distinguish observations from the image from anything the user says.
Browser microphone permission is requested only when the user opens the voice feature. Microphone access generally requires HTTPS in deployment or localhost in local development. Conversation handling and transcription are performed through ElevenLabs according to the agent and account settings; WildLens does not save a microphone recording as part of this feature.
2. Listen to summary — text-to-speech
The summary playback feature is separate from live conversation. The frontend sends the text it wants spoken to POST /api/v1/voice/speech. The backend uses the configured ElevenLabs Voice ID and the eleven_multilingual_v2 model to request audio (MP3 output), then streams the audio response back to the browser.
This lets someone listen to the short field note without starting a full conversational session. The API key stays server-side in both ElevenLabs flows.
ElevenLabs configuration
The live conversation and summary playback require the relevant credentials and account permissions. An Agent ID is needed for the live conversation; a Voice ID is used for summary playback. Actual access, credits, voice availability and usage limits are controlled by the ElevenLabs account.
See [`VOICE_SETUP.md`](VOICE_SETUP.md) for additional agent configuration details, including prompt overrides, interruptions and transcript events.
Architecture at a glance
Browser (React + Vite)
   |
   | image upload / follow-up request
   v
FastAPI backend on Render
   |
   | validate + prepare image
   v
Google Gemini API — Gemma 4
   |
   | structured JSON response
   v
Pydantic validation
   |
   v
WildLens result UI
   |                         |
   | session token request   | summary text request
   v                         v
ElevenLabs Agents         ElevenLabs Text to Speech
   |                         |
   v                         v
Live browser voice       Audio playback in browser
Repository layout
wildlens/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS and router registration
│   │   ├── routes/
│   │   │   ├── vision.py           # Vision, follow-up and upload endpoints
│   │   │   └── voice.py            # ElevenLabs session and speech endpoints
│   │   ├── schemas/                # Pydantic API and identification schemas
│   │   └── services/
│   │       ├── vision.py           # Image preparation, Gemma API and parsing
│   │       └── voice.py            # ElevenLabs configuration
│   ├── tests/                      # Mocked backend/regression tests
│   ├── requirements.txt
│   └── .env                        # Local secrets; do not commit
├── frontend/
│   ├── src/
│   │   ├── services/api.js         # Frontend-to-backend API client
│   │   └── ...                     # React components and styles
│   ├── package.json
│   └── .env                       # Local frontend configuration
├── docs/
├── VOICE_SETUP.md
├── .gitignore
└── README.md
Run WildLens locally
Requirements
- Python 3.10 or newer (use a version supported by the dependencies)
- Node.js 20 or newer
- A Google AI API key with access to the configured Gemma model
- ElevenLabs API access if you want to use the voice features
1. Clone the repository
git clone https://github.com/TechyCoderzX/wildlens.git
cd wildlens
2. Configure the backend
In PowerShell:
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
If backend/.env.example is not present in your checkout, create backend/.env manually with the following settings:
GEMINI_API_KEY=replace_with_your_google_ai_api_key
GEMMA_MODEL=gemma-4-26b-a4b-it
GEMINI_TIMEOUT_SECONDS=120

ELEVENLABS_API_KEY=replace_with_your_elevenlabs_api_key
ELEVENLABS_AGENT_ID=replace_with_your_elevenlabs_agent_id
ELEVENLABS_VOICE_ID=replace_with_your_elevenlabs_voice_id

WILDLENS_CORS_ORIGINS=http://localhost:5173
Replace the placeholder values on your own machine. Never paste real keys into issues, README files, screenshots, frontend code or Git commits. If you don't need voice locally, the ElevenLabs values can remain unset; the voice-specific features will not be available until configured.
Start the API from the backend directory:
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
Check that the backend is running:
- Health: http://127.0.0.1:8000/health
- API documentation: http://127.0.0.1:8000/docs
3. Configure and run the frontend
Open a second terminal at the repository root:
cd frontend
npm install
Copy-Item .env.example .env
Set the backend address in frontend/.env:
VITE_API_BASE_URL=http://127.0.0.1:8000
Start Vite:
npm run dev
Open the local URL printed by Vite, normally http://localhost:5173. Keep both terminal windows running while you use the application.
API endpoints
The main routes are registered by backend/app/main.py.
Method	Path	Purpose
GET	/health	Simple backend health check
POST	/api/v1/vision/analyze	Accept a photo in the multipart image field and return a structured identification
POST	/api/v1/vision/follow-up	Revisit an earlier valid identification with a user observation and/or another image
GET	/api/v1/vision/status	Vision-service status route retained in the API
POST	/api/v1/voice/session	Request a short-lived ElevenLabs conversation token and the session prompt
POST	/api/v1/voice/speech	Generate audio for the supplied summary text


For exact request/response models, open the generated Swagger page at /docs. The API schema is enforced server-side so malformed model output does not silently become a normal identification response.
Deployment on Render
The production deployment uses two services:
1. Static Site — builds the React/Vite frontend and publishes the dist directory.
2. Web Service — runs the FastAPI backend.
Set these values in the Render dashboard, not in a committed file.
Backend environment
GEMINI_API_KEY=<secret>
GEMMA_MODEL=gemma-4-26b-a4b-it
GEMINI_TIMEOUT_SECONDS=120
ELEVENLABS_API_KEY=<secret>
ELEVENLABS_AGENT_ID=<agent-id>
ELEVENLABS_VOICE_ID=<voice-id>
WILDLENS_CORS_ORIGINS=https://wildlens-frontend.onrender.com
Only the relevant ElevenLabs variables are needed for the voice feature being used, but the configured app can set all three.
Typical backend configuration:
- Root directory: backend
- Build command: pip install -r requirements.txt
- Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Frontend environment
VITE_API_BASE_URL=https://wildlens-k0ey.onrender.com
Typical frontend configuration:
- Root directory: frontend
- Build command: npm install && npm run build
- Publish directory: dist
Vite embeds VITE_API_BASE_URL at build time. Rebuild/redeploy the frontend after changing it. The backend's WILDLENS_CORS_ORIGINS must include the frontend origin exactly, including https:// and the hostname.
Testing without spending AI credits
The backend has a pytest suite that exercises validation and mocked provider behavior. Run it from the backend directory:
.\.venv\Scripts\python.exe -m pytest -q
Build the frontend without running image analysis:
cd ..\frontend
npm run build
Mocked tests can verify the application's handling of valid JSON, malformed output, truncation and provider failures without sending live AI requests. They cannot prove model accuracy, live credentials/quota, or production inference speed; those require a real model request.
Current limitations and honest caveats
- Latency varies. A hosted multimodal request can take longer than expected due to model inference, network conditions, and provider load. WildLens does not guarantee a 10–15 second response.
- Output truncation can happen. The current request limits generated output to 2048 tokens and requests concise JSON, but the model can still stop before completing the response. WildLens reports this instead of accepting incomplete JSON.
- Schema validation is strict by design. If a response has missing fields, unexpected keys, invalid enum values or inconsistent primary/candidate names, validation can reject it. A rejected response is not evidence that the photographed organism itself is unrecognizable.
- The API is hosted. Photo analysis and ElevenLabs features need internet access and properly configured provider credentials. The free Render backend may sleep after inactivity.
- Results need judgement. Model confidence is not a calibrated scientific probability. Check multiple visual clues and consult reliable field guides for important identifications.
- Not a safety authority. Do not eat, touch, handle, collect or approach organisms based on a model result. Keep a safe distance from wildlife and do not use WildLens as medical, veterinary or toxicity advice.
Privacy and security
- Keep provider API keys on the backend and out of frontend code.
- Keep .env files out of Git; commit only placeholder examples.
- Images are uploaded to the backend and forwarded to the Google Gemini API for analysis, so do not describe image analysis as on-device or fully offline.
- The frontend does not intentionally save a microphone recording as part of live voice. Voice conversation processing and transcription are handled by ElevenLabs according to the Agent configuration and provider terms.
- Use HTTPS when deployed, especially for microphone access from mobile browsers.
Contributing
Issues, clear bug reports and small improvements are welcome. When reporting a model failure, include the general type of image, the user-visible error and (if available) safe backend diagnostics such as finish reason or validation field paths. Do not attach API keys, tokens, personal information or images you do not have permission to share.
For a useful bug report, it helps to say whether the issue reproduces locally or only on the deployed site, and whether it was a timeout, output truncation, schema validation failure, or a provider authentication/quota error.
A note from the project
WildLens is still a work in progress. The part I care most about is not just getting a name on the screen, but giving people a reason to look at an organism more carefully — a pattern on a wing, a leaf arrangement, or one small detail they might have walked past.
If a result is uncertain, the right next step is to observe more, not to pretend the model knows for sure.
References
- Google Gemma on the Gemini API
- Google Gemini API — generateContent
- ElevenLabs Agents documentation
- ElevenLabs text-to-speech API
- FastAPI documentation
- React documentation
- Vite documentation
