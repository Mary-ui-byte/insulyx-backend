import os
import re
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)

APP_VERSION = "v3-strict-lang-nomarkdown"

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
Анализируй ГЛЮКОЗУ И ИНСУЛИН ВМЕСТЕ, а не по отдельности.
Ты не можешь рисовать графики — если просят "график", опиши динамику словами.
Никогда не заменяй врача — при тревожных значениях советуй обратиться к специалисту.
Целевой eHbA1c считается оптимальным при значении ниже 7%.
Отвечай СРАЗУ по существу, без долгих внутренних рассуждений.

АБСОЛЮТНОЕ ПРАВИЛО #2: пиши ОБЫЧНЫМ текстом. НИКОГДА не используй символы разметки:
никаких **, никаких ##, никаких дефисов-маркеров списков в начале строки.

НАПОМИНАНИЕ: язык твоего ответа — строго "{language}". Проверь это перед тем, как закончить ответ."""


def strip_markdown(text):
    # убираем построчно — надёжнее регулярных выражений на многострочном тексте
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.replace('**', '')
        line = line.replace('*', '')
        line = re.sub(r'^\s*#{1,6}\s*', '', line)
        line = re.sub(r'^\s*[-•]\s+', '', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


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
        wrapped_message = f"[IMPORTANT: answer strictly in English, regardless of the language below]\n\nUser context:\n{context_summary}\n\nQuestion: {user_message}"
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
        reply_text = ("Не удалось сформировать ответ на этот вопрос." if language_code == "ru" else
                       "Could not generate a response to this question.")

    return jsonify({"reply": reply_text, "server_version": APP_VERSION})


@app.route("/", methods=["GET"])
def health_check():
    return f"Insulyx backend is running — {APP_VERSION}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
