import asyncio
from io import BytesIO
import json

import httpx
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.routes import vision as route
from app.schemas.vision import IdentificationResult
from app.services import vision as gemma


client = TestClient(app)


def result_data(**overrides):
    data = {
        "status": "identified",
        "common_name": "House Sparrow",
        "scientific_name": "Passer domesticus",
        "confidence": "moderate",
        "summary": "Visible plumage is consistent with this species.",
        "visible_evidence": ["Brown streaking is visible on the back"],
        "evidence_details": [{
            "feature": "Brown back streaking",
            "inference": "Consistent with a house sparrow",
            "support": "moderate",
            "limitation": "The tail is partly obscured",
            "source": "original_image",
        }],
        "evidence_assessment": {
            "quality": "partial",
            "rationale": "One useful marking is visible, while the tail is obscured.",
            "supporting_points": ["Brown streaking"],
            "limitations": ["Tail pattern is not visible"],
        },
        "candidates": [{
            "common_name": "House Sparrow",
            "scientific_name": "Passer domesticus",
            "match_explanation": "The visible back markings are consistent.",
            "supporting_features": ["Brown streaking"],
            "distinguishing_features": [{
                "feature": "Tail pattern",
                "observation": "Not visible in this photo",
                "interpretation": "Cannot compare this marking yet",
            }],
            "uncertainties": ["Tail pattern is hidden"],
        }],
        "comparison_summary": "The visible markings fit the leading candidate, but the tail is hidden.",
        "interesting_facts": ["House sparrows are social birds."],
        "observation_tasks": [{
            "title": "Check the tail markings",
            "what_to_look_for": "Look for the tail from a side angle.",
            "why_it_matters": "Tail pattern can help compare similar sparrows.",
            "how_safely": "Use zoom from a distance; do not approach the bird.",
            "possible_outcomes": ["A visible pattern may narrow the comparison"],
        }],
        "observe_next": "Look for the tail pattern from a respectful distance.",
        "change_summary": "The tail remains unclear.",
        "disclaimer": "AI hypothesis; verify with a trusted field guide.",
    }
    data.update(overrides)
    return data


def test_analyze_route_returns_extended_identification(monkeypatch):
    async def fake_analyze(content, content_type, extra_instructions=None):
        assert content == b"photo"
        assert content_type == "image/jpeg"
        return IdentificationResult.model_validate(result_data())

    monkeypatch.setattr(route, "analyze_image", fake_analyze)
    response = client.post("/api/v1/vision/analyze", files={"image": ("bird.jpg", b"photo", "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["evidence_assessment"]["quality"] == "partial"
    assert body["candidates"][0]["common_name"] == "House Sparrow"


def test_missing_api_key_fails_without_request(monkeypatch):
    monkeypatch.setattr(gemma.settings, "gemini_api_key", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))


@pytest.mark.parametrize(
    ("content_type", "content", "expected"),
    [("text/plain", b"x", 415), ("image/jpeg", b"", 400)],
)
def test_analyze_route_rejects_bad_upload(monkeypatch, content_type, content, expected):
    async def fail_if_called(*args, **kwargs):
        pytest.fail("model must not be called for invalid upload")

    monkeypatch.setattr(route, "analyze_image", fail_if_called)
    response = client.post("/api/v1/vision/analyze", files={"image": ("upload", content, content_type)})
    assert response.status_code == expected


def test_analyze_route_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(route, "MAX_IMAGE_BYTES", 8)
    response = client.post("/api/v1/vision/analyze", files={"image": ("upload.jpg", b"x" * 9, "image/jpeg")})
    assert response.status_code == 413


def test_oversized_model_image_is_resized_aspect_preserved_and_encoded_as_jpeg():
    original = Image.new("RGB", (3000, 1500), color=(30, 100, 50))
    encoded = BytesIO()
    original.save(encoded, format="PNG")

    prepared, content_type = gemma.prepare_model_image(encoded.getvalue(), "image/png")

    with Image.open(BytesIO(prepared)) as resized:
        assert resized.format == "JPEG"
        assert resized.size == (1280, 640)
    assert content_type == "image/jpeg"


def test_small_jpeg_model_image_is_not_upscaled_or_reencoded():
    original = Image.new("RGB", (640, 360), color=(30, 100, 50))
    encoded = BytesIO()
    original.save(encoded, format="JPEG", quality=90)
    image_bytes = encoded.getvalue()

    prepared, content_type = gemma.prepare_model_image(image_bytes, "image/jpeg")

    assert prepared == image_bytes
    assert content_type == "image/jpeg"


def install_gemini_mock(monkeypatch, handler):
    original_client = httpx.AsyncClient

    class MockClient(original_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(gemma.httpx, "AsyncClient", MockClient)
    monkeypatch.setattr(gemma.settings, "gemini_api_key", "test-key")


def gemini_response(text, status=200):
    return httpx.Response(
        status,
        json={"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/mock"),
    )


def test_valid_bird_model_response_is_parsed_and_schema_validated(monkeypatch):
    def handler(request):
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        prompt = body["systemInstruction"]["parts"][0]["text"] + body["contents"][0]["parts"][0]["text"]
        assert body["generationConfig"]["maxOutputTokens"] == 2048
        assert "at most 3 short visible_evidence items" in prompt
        assert "at most 2 short interesting_facts" in prompt
        assert "at most 1 short observation_task" in prompt
        assert "No repeated descriptions, Markdown, preamble, or extra keys" in prompt
        assert "animals (including birds), plants, fungi" in prompt
        assert "microorganisms (including bacteria)" in prompt
        assert "exactly" in prompt and "add no keys" in prompt
        return gemini_response(json.dumps(result_data()))

    install_gemini_mock(monkeypatch, handler)
    result = asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))
    assert gemma.REQUIRED_RESULT_FIELDS.issubset(result.model_fields_set)
    assert result.common_name == "House Sparrow"
    assert len(result.candidates) == 1
    assert result.comparison_summary
    assert result.observe_next
    assert result.change_summary


def test_valid_bacteria_model_response_is_accepted(monkeypatch):
    bacteria = result_data(
        status="uncertain", common_name="Bacteria", scientific_name=None,
        confidence="low", summary="Small clustered cells are visible, but species-level details are not.",
        visible_evidence=["Small clustered cells"],
        evidence_details=[{
            "feature": "Small clustered cells", "inference": "Consistent with a bacterial colony",
            "support": "weak", "limitation": "The image does not show species-level features",
            "source": "original_image",
        }],
        evidence_assessment={"quality": "partial", "rationale": "Cell shape is visible, but diagnostic detail is limited.",
                             "supporting_points": ["Clustered cell-like forms"], "limitations": ["Scale is unknown"]},
        candidates=[{
            "common_name": "Bacteria", "scientific_name": None,
            "match_explanation": "The visible forms are consistent with bacteria, without enough detail for a narrower identification.",
            "supporting_features": ["Clustered cell-like forms"], "distinguishing_features": [],
            "uncertainties": ["Species cannot be determined from this image"],
        }],
        observation_tasks=[],
    )
    install_gemini_mock(monkeypatch, lambda request: gemini_response(json.dumps(bacteria)))

    result = asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))

    assert result.common_name == "Bacteria"
    assert result.status == "uncertain"


def test_valid_mammal_model_response_is_accepted(monkeypatch):
    mammal = result_data(
        common_name="Red Fox", scientific_name="Vulpes vulpes",
        summary="Coat and muzzle fit a red fox.",
        visible_evidence=["Reddish coat", "Pointed muzzle"],
        evidence_details=[
            {"feature": "Reddish coat", "inference": "Consistent with a red fox", "support": "moderate",
             "limitation": "Lighting can alter coat color", "source": "original_image"},
            {"feature": "Pointed muzzle", "inference": "Consistent with a fox", "support": "weak",
             "limitation": "The face is partly obscured", "source": "original_image"},
        ],
        evidence_assessment={"quality": "partial", "rationale": "Two clues fit; lighting limits certainty.",
                             "supporting_points": ["Reddish coat"], "limitations": ["Face partly hidden"]},
        candidates=[{
            "common_name": "Red Fox", "scientific_name": "Vulpes vulpes",
            "match_explanation": "Coat and muzzle fit.",
            "supporting_features": ["Reddish coat", "Pointed muzzle"],
            "distinguishing_features": [], "uncertainties": ["Lighting affects apparent coat color"],
        }],
        comparison_summary=None,
        interesting_facts=["Foxes are canids."],
        observation_tasks=[{
            "title": "Check ear shape", "what_to_look_for": "View both ears clearly.",
            "why_it_matters": "Ear shape helps compare foxes.", "how_safely": "Observe from a distance.",
            "possible_outcomes": ["A clear view may help distinguish lookalikes."],
        }],
        observe_next=None,
        change_summary=None,
        disclaimer="AI hypothesis; verify with a trusted field guide.",
    )

    def handler(request):
        body = json.loads(request.content)
        prompt = body["systemInstruction"]["parts"][0]["text"] + body["contents"][0]["parts"][0]["text"]
        assert body["generationConfig"]["maxOutputTokens"] == 2048
        assert "at most 3 short visible_evidence items" in prompt
        assert "at most 2 short interesting_facts" in prompt
        assert "at most 1 short observation_task" in prompt
        assert "No repeated descriptions, Markdown, preamble, or extra keys" in prompt
        return gemini_response(json.dumps(mammal))

    install_gemini_mock(monkeypatch, handler)

    result = asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))

    assert result.common_name == "Red Fox"
    assert result.scientific_name == "Vulpes vulpes"


def test_token_usage_logging_is_allowlisted_and_does_not_log_raw_metadata(monkeypatch, caplog):
    def handler(request):
        data = result_data()
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": json.dumps(data)}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 1200,
                    "candidatesTokenCount": 540,
                    "thoughtsTokenCount": 0,
                    "totalTokenCount": 1740,
                    "untrustedField": "DO_NOT_LOG",
                },
                "sensitive": "DO_NOT_LOG_EITHER",
            },
            request=request,
        )

    install_gemini_mock(monkeypatch, handler)
    with caplog.at_level("INFO", logger="app.services.vision"):
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))
    assert "prompt_tokens=1200" in caplog.text
    assert "candidate_tokens=540" in caplog.text
    assert "total_tokens=1740" in caplog.text
    assert "DO_NOT_LOG" not in caplog.text


@pytest.mark.parametrize("missing", [False, True], ids=["explicit-null", "omitted"])
def test_optional_model_text_fields_accept_null_or_missing(monkeypatch, missing):
    optional_fields = ("comparison_summary", "observe_next", "change_summary")
    response_data = result_data()
    for field in optional_fields:
        if missing:
            response_data.pop(field)
        else:
            response_data[field] = None

    install_gemini_mock(monkeypatch, lambda request: gemini_response(json.dumps(response_data)))
    result = asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))

    for field in optional_fields:
        assert getattr(result, field) is None


def test_unknown_and_ambiguous_single_candidate_are_valid(monkeypatch):
    unknown = result_data(
        status="unknown", common_name="Unknown organism", scientific_name=None,
        confidence="low", visible_evidence=[], evidence_details=[], candidates=[],
        observation_tasks=[],
    )
    parsed = IdentificationResult.model_validate(unknown)
    assert parsed.status == "unknown"
    ambiguous = IdentificationResult.model_validate(result_data(status="uncertain", confidence="low"))
    assert ambiguous.confidence == "low"
    assert len(ambiguous.candidates) == 1


def test_first_candidate_matching_leading_identification_is_valid():
    parsed = IdentificationResult.model_validate(result_data())
    assert parsed.candidates[0].common_name == parsed.common_name


def test_case_only_candidate_name_difference_is_supported_by_existing_validator():
    data = result_data()
    data["candidates"][0]["common_name"] = "HOUSE SPARROW"
    parsed = IdentificationResult.model_validate(data)
    assert parsed.candidates[0].common_name == "HOUSE SPARROW"


def test_mismatching_first_candidate_fails_with_concise_actionable_error(monkeypatch):
    data = result_data()
    data["candidates"][0]["common_name"] = "Zebra Finch"
    install_gemini_mock(monkeypatch, lambda request: gemini_response(json.dumps(data)))

    with pytest.raises(ValueError, match="inconsistent candidate names") as failure:
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))

    assert "Refine the photo or submit the identification again" in str(failure.value)
    assert "input_value" not in str(failure.value)


def test_malformed_model_json_returns_without_expensive_retry(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return gemini_response("not JSON")

    install_gemini_mock(monkeypatch, handler)
    with pytest.raises(ValueError, match="No valid result JSON"):
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))
    assert calls == 1


def test_missing_required_fields_fail_validation(monkeypatch):
    install_gemini_mock(monkeypatch, lambda request: gemini_response('{"status":"identified","common_name":"Bird"}'))
    with pytest.raises(ValueError, match="missing required fields"):
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))


def test_invalid_enum_is_not_accepted_as_a_complete_structured_result(monkeypatch):
    invalid = result_data(confidence="certain")
    install_gemini_mock(monkeypatch, lambda request: gemini_response(json.dumps(invalid)))
    with pytest.raises(ValueError, match="failed schema validation"):
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))


def test_schema_validation_logs_only_field_paths_and_error_types(monkeypatch, caplog):
    invalid = result_data(confidence="certain", summary="MODEL_OUTPUT_SECRET")
    install_gemini_mock(monkeypatch, lambda request: gemini_response(json.dumps(invalid)))
    with caplog.at_level("WARNING", logger="app.services.vision"):
        with pytest.raises(ValueError, match="failed schema validation"):
            asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))

    assert "confidence" in caplog.text
    assert "literal_error" in caplog.text
    assert "MODEL_OUTPUT_SECRET" not in caplog.text
    assert "test-key" not in caplog.text


def test_api_rate_limit_fails_without_retry(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return gemini_response("{}", status=429)

    install_gemini_mock(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="quota or rate limit"):
        asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))
    assert calls == 1


def test_max_tokens_reports_truncation_and_safe_usage_without_retry(monkeypatch, caplog):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"candidates": [{
                "content": {"parts": [{"text": '{"status":"identified","common_name":"MODEL_OUTPUT_SECRET'}]},
                "finishReason": "MAX_TOKENS",
            }], "usageMetadata": {
                "promptTokenCount": 321, "candidatesTokenCount": 2048, "thoughtsTokenCount": 17,
            }},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/mock"),
        )

    install_gemini_mock(monkeypatch, handler)
    with caplog.at_level("INFO", logger="app.services.vision"):
        with pytest.raises(gemma.ModelOutputTruncatedError, match="2048-token limit"):
            asyncio.run(gemma.analyze_image(b"photo", "image/jpeg"))
    assert calls == 1
    assert "finish_reason=MAX_TOKENS" in caplog.text
    assert "prompt_tokens=321" in caplog.text
    assert "candidate_tokens=2048" in caplog.text
    assert "thought_tokens=17" in caplog.text
    assert "MODEL_OUTPUT_SECRET" not in caplog.text
    assert "test-key" not in caplog.text


def test_portrait_scope_is_explicitly_unknown_without_person_identification(monkeypatch):
    out_of_scope = result_data(
        status="unknown", common_name="Out of scope", scientific_name=None,
        confidence="low", summary="This portrait is outside WildLens's nature-identification scope.",
        visible_evidence=[], evidence_details=[],
        evidence_assessment={"quality": "insufficient", "rationale": "No nature subject is present."},
        candidates=[], comparison_summary="", interesting_facts=[], observation_tasks=[],
    )

    def handler(request):
        body = json.loads(request.content)
        system_prompt = body["systemInstruction"]["parts"][0]["text"]
        assert "Do not identify people" in system_prompt
        assert "Out of scope" in system_prompt
        return gemini_response(json.dumps(out_of_scope))

    install_gemini_mock(monkeypatch, handler)
    result = asyncio.run(gemma.analyze_image(b"portrait", "image/jpeg"))
    assert result.status == "unknown"
    assert result.common_name == "Out of scope"
    assert result.candidates == []


def test_follow_up_with_user_observation(monkeypatch):
    async def fake_follow_up(content, content_type, previous, observation):
        assert content is None
        assert observation == "The tail appears barred."
        updated = IdentificationResult.model_validate(result_data(change_summary="The report adds a tail observation."))
        updated.evidence_details[0].source = "user_reported"
        return updated

    monkeypatch.setattr(route, "analyze_follow_up", fake_follow_up)
    response = client.post("/api/v1/vision/follow-up", data={
        "previous": json.dumps(result_data()),
        "observation": "The tail appears barred.",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["identification"]["evidence_details"][0]["source"] == "user_reported"
    assert "tail observation" in body["what_changed"]


def test_follow_up_keeps_missing_change_summary_absent(monkeypatch):
    async def fake_follow_up(content, content_type, previous, observation):
        return IdentificationResult.model_validate(result_data(change_summary=None))

    monkeypatch.setattr(route, "analyze_follow_up", fake_follow_up)
    response = client.post("/api/v1/vision/follow-up", data={
        "previous": json.dumps(result_data()),
        "observation": "The tail appears barred.",
    })
    assert response.status_code == 200
    assert response.json()["what_changed"] is None


def test_follow_up_with_new_image(monkeypatch):
    async def fake_follow_up(content, content_type, previous, observation):
        assert (content, content_type, observation) == (b"new photo", "image/png", None)
        return IdentificationResult.model_validate(result_data())

    monkeypatch.setattr(route, "analyze_follow_up", fake_follow_up)
    response = client.post("/api/v1/vision/follow-up", data={"previous": json.dumps(result_data())}, files={
        "image": ("second.png", b"new photo", "image/png"),
    })
    assert response.status_code == 200


def test_follow_up_rejects_empty_invalid_and_oversized_inputs(monkeypatch):
    empty = client.post("/api/v1/vision/follow-up", data={"previous": json.dumps(result_data())})
    invalid = client.post("/api/v1/vision/follow-up", data={"previous": "{}", "observation": "Maybe a bird"})
    oversized = client.post("/api/v1/vision/follow-up", data={
        "previous": json.dumps(result_data()), "observation": "x" * 801,
    })
    monkeypatch.setattr(route, "MAX_IMAGE_BYTES", 8)
    too_large_image = client.post("/api/v1/vision/follow-up", data={
        "previous": json.dumps(result_data()),
    }, files={"image": ("large.png", b"x" * 9, "image/png")})
    assert empty.status_code == 422
    assert invalid.status_code == 422
    assert oversized.status_code == 422
    assert too_large_image.status_code == 413


def test_follow_up_user_text_is_not_marked_as_image_evidence(monkeypatch):
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["prompt"] = body["contents"][0]["parts"][0]["text"]
        body = json.loads(request.content)
        assert body["generationConfig"]["maxOutputTokens"] == 2048
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "SHORT complete JSON object" in captured["prompt"]
        assert "at most 2 candidates" in captured["prompt"]
        assert "at most 1 observation_task" in captured["prompt"]
        return gemini_response(json.dumps(result_data(change_summary="New detail added.")))

    install_gemini_mock(monkeypatch, handler)
    previous = IdentificationResult.model_validate(result_data())
    result = asyncio.run(gemma.analyze_follow_up(None, None, previous, "The tail looks striped."))
    assert "user_reported_observation" in captured["prompt"]
    assert "user_reported" in captured["prompt"]
    assert result.change_summary == "New detail added."


def test_successful_follow_up_with_photo_and_observation(monkeypatch):
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["body"] = body
        return gemini_response(json.dumps(result_data(change_summary="The new photo clarifies the tail.")))

    install_gemini_mock(monkeypatch, handler)
    previous = IdentificationResult.model_validate(result_data())
    result = asyncio.run(gemma.analyze_follow_up(
        b"second photo", "image/jpeg", previous, "The tail looks barred."
    ))
    parts = captured["body"]["contents"][0]["parts"]
    assert parts[1]["inlineData"]["data"]
    assert "The tail looks barred." in parts[0]["text"]
    assert result.change_summary == "The new photo clarifies the tail."
