# Manual smoke test

1. Start Ollama and confirm `ollama list` includes the configured model.
2. Start the API; verify `GET /health` returns `{"status":"ok","service":"wildlens-api"}`.
3. Open `/docs`; verify the vision status endpoint reports Ollama availability.
4. Start Vite and open the page.
5. Select a non-sensitive JPEG/PNG/WebP photo under 10 MB and run identification.
6. Test a text file renamed as `.jpg`, an unsupported file type, an empty request, and an unavailable model.
7. Record inference duration and the exact model tag. Do not claim accuracy or offline mobile support without testing.

For automated backend tests, from `backend/` run:
```powershell
pip install -r requirements.txt
pytest
```
