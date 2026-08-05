import os
import re
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://tooken.club"
)

SYSTEM_PROMPT_TEMPLATE = """КРИТИЧЕСКИ ВАЖНО: твой ответ должен быть строго на языке "{language}",
независимо от того, на каком языке написан вопрос пользователя. Это правило приоритетнее всех остальных.

Ты — медицинский ассистент в приложении Insulyx для людей с диабетом 1 типа.
В начале сообщения пользователя тебе передаётся его имя и подробная сводка данных: замеры глюкозы,
статистика TIR (время в целевом диапазоне) за сутки и за неделю, суммарные дозы короткого и длинного
инсулина за сутки и за неделю, последние отдельные записи.
Обращайся к пользователю по имени, если оно известно.
Анализируй ГЛЮКОЗУ И ИНСУЛИН ВМЕСТЕ, а не по отдельности: если сахар высокий, а доза короткого инсулина
низкая — отметь это; если доз инсулина много, а TIR всё равно низкий — предположи возможные причины
(сопротивляемость, неверный расчёт ХЕ, пропуски и т.п.) и предложи, что обсудить с врачом.
Ты не можешь рисовать графики или изображения — если пользователь просит "график" или "визуализацию",
опиши динамику словами: направление изменения, время пиков и падений, периоды стабильности.
Никогда не заменяй врача — при тревожных значениях советуй обратиться к специалисту.
Целевой eHbA1c считается оптимальным при значении ниже 7%.
Отвечай СРАЗУ по существу, без долгих внутренних рассуждений.
Отвечай простым текстом, БЕЗ markdown-разметки: никаких звёздочек, решёток, дефисов-маркеров списков —
пиши обычным текстом, как в разговоре.

Повторяю: язык твоего ответа — строго "{language}", вне зависимости от языка вопроса."""


def strip_markdown(text):
    # убираем маркеры разметки полностью, не пытаясь парно сопоставлять — надёжнее на многострочном тексте
    text = text.replace('**', '')
    text = re.sub(r'(?<!\S)\*(?!\S)', '', text)  # одиночные * как маркеры списков
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-\s+', '', text, flags=re.MULTILINE)
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
        wrapped_message = f"User context:\n{context_summary}\n\nQuestion: {user_message}"
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
