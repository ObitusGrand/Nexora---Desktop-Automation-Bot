from desktop_bot.app import AppConfig, load_config


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