import os
import re
import anthropic
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://tooken.club"
)

# Модель загружается один раз при старте сервера
gbm_model = joblib.load("glucose_gbm_model.pkl")
GBM_FEATURES = ["glucose_now", "carbs_xe", "insulin_short_u", "insulin_long_u", "minutes_since_dose", "hour_of_day"]

SYSTEM_PROMPT_TEMPLATE = """АБСОЛЮТНОЕ ПРАВИЛО #1: весь твой ответ, целиком, каждое слово, должен быть
написан на языке "{language}" — вне зависимости от того, на каком языке пользователь задал вопрос
и на каком языке написан контекст ниже.

Ты — медицинский ассистент в приложении Insulyx для людей с диабетом 1 типа.
В начале сообщения пользователя тебе передаётся его имя и подробная сводка данных: замеры глюкозы,
статистика TIR за сутки и за неделю, суммарные дозы короткого и длинного инсулина, последние записи,
а также, если доступен, прогноз уровня глюкозы через 30 минут от модели градиентного бустинга.
Обращайся к пользователю по имени, если оно известно.
Анализируй ГЛЮКОЗУ И ИНСУЛИН ВМЕСТЕ, а не по отдельности. Если есть прогноз модели — учитывай его
в рекомендациях (например, предупреждай о надвигающейся гипо- или гипергликемии).
Ты не можешь рисовать графики — если просят "график", опиши динамику словами.
Никогда не заменяй врача — при тревожных значениях советуй обратиться к специалисту.
Отвечай СРАЗУ по существу, без долгих внутренних рассуждений.

АБСОЛЮТНОЕ ПРАВИЛО #2: пиши ОБЫЧНЫМ текстом. НИКОГДА не используй символы разметки:
никаких **, никаких ##, никаких дефисов-маркеров списков в начале строки.

НАПОМИНАНИЕ: язык твоего ответа — строго "{language}"."""


def strip_markdown(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.replace('**', '')
        line = line.replace('*', '')
        line = re.sub(r'^\s*#{1,6}\s*', '', line)
        line = re.sub(r'^\s*[-•]\s+', '', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            float(data.get("glucose_now", 7.0)),
            float(data.get("carbs_xe", 0)),
            float(data.get("insulin_short_u", 0)),
            float(data.get("insulin_long_u", 0)),
            float(data.get("minutes_since_dose", 120)),
            float(data.get("hour_of_day", 12)),
        ]])
        prediction = float(gbm_model.predict(features)[0])
        prediction = max(2.0, min(25.0, prediction))
        return jsonify({"predicted_glucose_30min": round(prediction, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


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

    messages = history + [{"role": "user", "content": wrapped_message}]

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

    return jsonify({"reply": reply_text})


@app.route("/", methods=["GET"])
def health_check():
    return "Insulyx backend is running — with GBM prediction model"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
