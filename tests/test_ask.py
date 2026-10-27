"""質問スクリプトのテスト（LLMもAPIキーも使わない）。"""

import sys
from types import SimpleNamespace

from app import ask


def test_chat_keeps_follow_up_questions_in_one_session(monkeypatch, capsys):
    sent_messages = []

    class FakeChat:
        def send_message(self, message, **kwargs):
            sent_messages.append(message)
            return SimpleNamespace(text=f"回答: {message}")

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: FakeChat())
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    inputs = iter(["追加の質問", "", "もう一つ", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    ask.main()

    assert sent_messages == ["最初の質問", "追加の質問", "もう一つ"]
    assert "回答: 最初の質問" in capsys.readouterr().out


def test_once_keeps_single_question_behavior(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["app.ask", "--once", "一度だけの質問"])
    monkeypatch.setattr(ask, "ask", lambda question: f"回答: {question}")

    ask.main()

    assert "回答: 一度だけの質問" in capsys.readouterr().out


def test_ask_prints_usage_metadata(monkeypatch, capsys):
    fake_response = SimpleNamespace(
        text="回答テキスト",
        usage_metadata={"prompt_token_count": 123, "candidates_token_count": 456},
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: fake_response)
    )

    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    result = ask.ask("質問")

    assert result == "回答テキスト"
    assert "入力: 123, 出力: 456" in capsys.readouterr().out


def test_ask_passes_default_temperature_to_generate_content(monkeypatch, capsys):
    captured = {}
    fake_response = SimpleNamespace(text="温度ありの回答", usage_metadata=None)

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return fake_response

    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    result = ask.ask("質問")

    assert result == "温度ありの回答"
    assert captured["config"] == {"temperature": ask.DEFAULT_TEMPERATURE}


def test_main_uses_default_temperature_for_chat(monkeypatch, capsys):
    sent_configs = []

    class FakeChat:
        def send_message(self, message, **kwargs):
            sent_configs.append(kwargs)
            return SimpleNamespace(text=f"回答: {message}")

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: FakeChat())
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    ask.main()

    assert sent_configs == [{"config": {"temperature": ask.DEFAULT_TEMPERATURE}}]
    assert f"[temperature] {ask.DEFAULT_TEMPERATURE}" in capsys.readouterr().out


def test_chat_prints_fallback_when_response_has_no_text(monkeypatch, capsys):
    class FakeChat:
        def send_message(self, message, **kwargs):
            return SimpleNamespace(text=None)

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: FakeChat())
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(sys, "argv", ["app.ask", "質問"])
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    ask.main()

    assert "テキスト形式の回答がありませんでした。" in capsys.readouterr().out
