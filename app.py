import os
import re
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://tooken.club"
)

SYSTEM_PROMPT_TEMPLATE = """АБСОЛЮТНОЕ ПРАВИЛО #1: весь твой ответ, целиком, каждое слово, должен быть
написан на языке "{language}" — вне зависимости от того, на каком языке пользователь задал вопрос
и на каком языке написан контекст ниже. Даже если вопрос и данные на русском, а язык ответа "английский",
отвечай ПОЛНОСТЬЮ на английском. Это правило важнее содержания ответа.

Ты — медицинский ассистент в приложении Insulyx для людей с диабетом 1 типа.
В начале сообщения пользователя тебе передаётся его имя и подробная сводка данных: замеры глюкозы,
статистика TIR (время в целевом диапазоне) за сутки и за неделю, суммарные дозы короткого и длинного
инсулина за сутки и за неделю, последние отдельные записи.
Обращайся к пользователю по имени, если оно известно.
Анализируй ГЛЮКОЗУ И ИНСУЛИН ВМЕСТЕ, а не по отдельности: если сахар высокий, а доза короткого инсулина
низкая — отметь это; если доз инсулина много, а TIR всё равно низкий — предположи возможные причины
и предложи, что обсудить с врачом.
Ты не можешь рисовать графики — если просят "график", опиши динамику словами.
Никогда не заменяй врача — при тревожных значениях советуй обратиться к специалисту.
Целевой eHbA1c считается оптимальным при значении ниже 7%.
Отвечай СРАЗУ по существу, без долгих внутренних рассуждений.

АБСОЛЮТНОЕ ПРАВИЛО #2: пиши ОБЫЧНЫМ текстом. НИКОГДА не используй символы разметки:
никаких **, никаких ##, никаких дефисов-маркеров списков в начале строки. Если нужно перечислить
несколько пунктов — пиши их обычными предложениями через запятую или с новой строки без каких-либо
специальных символов впереди.

НАПОМИНАНИЕ: язык твоего ответа — строго "{language}". Проверь это перед тем, как закончить ответ."""


def strip_markdown(text):
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    return text


@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    user_message = data.get("message", "")
    context_summary = data.get("context", "")
    history = data.get("history", [])
    language_code = data.get("language", "ru")

    is_english = language_code == "en"
    language_name = "английский (English)" if is_english else "русский"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language_name)

    if is_english:
        wrapped_message = f"[IMPORTANT: answer strictly in English]\n\nUser context:\n{context_summary}\n\nQuestion: {user_message}"
    else:
        wrapped_message = f"Контекст пользователя:\n{context_summary}\n\nВопрос: {user_message}"

    messages = history + [
        {"role": "user", "content": wrapped_message}
    ]

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    )

    reply_text = ""
    for block in response.content:
        if block.type == "text":
            reply_text = block.text
            break

    reply_text = strip_markdown(reply_text).strip()

    if not reply_text:
        reply_text = ("Не удалось сформировать ответ на этот вопрос — попробуйте переформулировать "
                       "его короче или задать по частям." if language_code == "ru" else
                       "Could not generate a response to this question — try rephrasing it more concisely "
                       "or asking it in smaller parts.")

    return jsonify({"reply": reply_text})


@app.route("/", methods=["GET"])
def health_check():
    return "Insulyx backend is running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
