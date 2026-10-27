"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

起動中は会話履歴を保ったまま続けて質問できる。ツールは持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage_metadata(response: object) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_token_count", usage.get("input_tokens", 0))
        output_tokens = usage.get(
            "candidates_token_count", usage.get("output_tokens", 0)
        )
    else:
        input_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", 0)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", 0)

    print(f"[トークン数] 入力: {input_tokens}, 出力: {output_tokens}")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage_metadata(response)
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--once", action="store_true", help="一度だけ質問して終了する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        if args.once:
            print(ask(args.question))
            return

        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        question = args.question

        while True:
            response = chat.send_message(question)
            print(response.text)
            print_usage_metadata(response)

            while True:
                try:
                    next_question = input(
                        "\n続けて質問してください（終了: exit / quit）> "
                    ).strip()
                except EOFError:
                    return

                if next_question.lower() in {"exit", "quit", "/exit", "/quit"}:
                    return
                if not next_question:
                    continue
                question = next_question
                print(f"\n[質問] {question}\n")
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
