import os
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://tooken.club"
)

SYSTEM_PROMPT = """Ты — медицинский ассистент в приложении Insulyx для людей с диабетом 1 типа.
У тебя есть данные пользователя (последние замеры глюкозы, статистика TIR, eHbA1c).
Анализируй их, давай мягкие, не диагностические советы, задавай уточняющие вопросы.
Никогда не заменяй врача — при тревожных значениях советуй обратиться к специалисту.
Целевой eHbA1c считается оптимальным при значении ниже 7%."""


@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    user_message = data.get("message", "")
    context_summary = data.get("context", "")
    history = data.get("history", [])

    messages = history + [
        {"role": "user", "content": f"Контекст пользователя:\n{context_summary}\n\nВопрос: {user_message}"}
    ]

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    reply_text = ""
    for block in response.content:
        if block.type == "text":
            reply_text = block.text
            break

    return jsonify({"reply": reply_text})


@app.route("/", methods=["GET"])
def health_check():
    return "Insulyx backend is running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
