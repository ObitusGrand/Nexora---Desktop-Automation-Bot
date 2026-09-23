from desktop_bot.app import AppConfig, direct_type_text, load_config


def test_direct_type_text_extracts_explicit_focused_control_commands():
    assert direct_type_text("type PEEPS BOBO") == "PEEPS BOBO"
    assert direct_type_text("type: hello world") == "hello world"
    assert direct_type_text("enter: message") == "message"
    assert direct_type_text("Open Notepad") is None
    assert direct_type_text("type:") is None


def test_load_config_uses_environment_values():
    config = load_config({
        "DESKTOP_BOT_VLM": "openai-compatible",
        "DESKTOP_BOT_MODEL": "vision-model",
        "DESKTOP_BOT_ENDPOINT": "https://example.test/v1",
        "DESKTOP_BOT_API_KEY": "secret",
        "DESKTOP_BOT_MEMORY_PATH": "memory",
    })

    assert config == AppConfig(
        provider="openai-compatible",
        model="vision-model",
        endpoint="https://example.test/v1",
        api_key="secret",
        memory_path="memory",
    )


def test_load_config_has_local_defaults():
    config = load_config({})

    assert config.provider == "ollama"
    assert config.model == "llava"
    assert config.endpoint == "http://localhost:11434"


def test_ollama_error_explains_when_service_is_unavailable():
    from unittest.mock import patch

    import httpx

    from desktop_bot.vlm import OllamaVLM, VLMError

    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    with patch("desktop_bot.vlm.httpx.post", side_effect=httpx.ConnectError("offline", request=request)):
        try:
            OllamaVLM("llava").decide(b"image", "test")
        except VLMError as exc:
            assert "Ollama is not running" in str(exc)
        else:
            raise AssertionError("expected a clear Ollama availability error")


def test_ollama_timeout_explains_when_service_is_unavailable():
    from unittest.mock import patch

    import httpx

    from desktop_bot.vlm import OllamaVLM, VLMError

    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    with patch("desktop_bot.vlm.httpx.post", side_effect=httpx.ReadTimeout("offline", request=request)):
        try:
            OllamaVLM("llava").decide(b"image", "test")
        except VLMError as exc:
            assert "Ollama is not running" in str(exc)
        else:
            raise AssertionError("expected a clear Ollama availability error")


def test_ollama_verification_retries_plain_json_after_http_500():
    from unittest.mock import patch

    import httpx

    from desktop_bot.vlm import OllamaVLM

    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    failed = httpx.Response(500, request=request, text="structured output failed")
    succeeded = httpx.Response(
        200,
        request=request,
        json={"message": {"content": '{"verified":true,"reason":"changed"}'}},
    )
    with patch("desktop_bot.vlm.httpx.post", side_effect=[failed, succeeded]) as post:
        result = OllamaVLM("llava").verify(b"before", b"after", "the screen changes")

    assert result.verified is True
    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"]["format"] == "json"