import os
import time
from typing import Dict

import streamlit as st
from transformers import pipeline


# ---------- Page setup ----------
st.set_page_config(
    page_title="AI Mental Health Support Bot",
    page_icon="💙",
    layout="centered",
)


# ---------- Model loading (cached) ----------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource(show_spinner=False)
def load_emotion_pipeline():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
    )


def detect_crisis(text: str) -> bool:
    t = text.lower()
    crisis_terms = [
        "suicide",
        "kill myself",
        "end my life",
        "can't go on",
        "cant go on",
        "self harm",
        "self-harm",
        "hurt myself",
        "harm myself",
        "ending it",
        "no reason to live",
    ]
    return any(term in t for term in crisis_terms)


def build_empathetic_response(label: str, score: float, user_text: str, crisis: bool) -> str:
    if crisis:
        return (
            "I'm really sorry you're feeling this way. Your life matters and you deserve support. "
            "If you're in immediate danger, please call your local emergency number (e.g., 112/911/999).\n\n"
            "You can reach a crisis line right now:\n"
            "- International: https://findahelpline.com/\n"
            "- India: AASRA +91-9820466726 | https://aasra.info/\n"
            "- US: 988 Suicide & Crisis Lifeline (call/text 988) | https://988lifeline.org/\n\n"
            "I'm here to listen. Would you like to tell me a little more about what's on your mind?"
        )

    if label == "NEGATIVE":
        if score >= 0.7:
            return (
                "That sounds really tough, and it's completely okay to feel this way. "
                "You don't have to handle everything at once. Maybe try a small grounding step: "
                "take 3 slow breaths (in 4s, hold 4s, out 6s). If you'd like, we can break the situation "
                "into smaller parts together. What part feels heaviest right now?"
            )
        else:
            return (
                "I hear some heaviness in what you shared. You're not alone. "
                "What would feel most supportive for you right now—venting, problem-solving, or a simple check-in?"
            )

    if label == "POSITIVE":
        return (
            "I love the hopeful energy here. That's a strength you can build on. "
            "What helped you get to this point today? Maybe we can note a small win to carry forward."
        )

    # NEUTRAL or anything else
    return (
        "Thanks for sharing. I'm here with you. "
        "If you'd like, I can offer a gentle prompt: what's one small action that could make the next hour "
        "a bit easier?"
    )


def classify_emotion(user_text: str, emo_clf):
    r = emo_clf(user_text)[0]
    emo_label = str(r.get("label", "neutral")).lower()
    emo_score = float(r.get("score", 0.5))
    return emo_label, emo_score


def build_emotion_specific_response(emotion: str, user_text: str) -> str:
    templates = {
        "sadness": [
            "It sounds really heavy. It's okay to feel sad. A tiny step like writing down one worry or taking a 2‑minute stretch can help. What feels smallest to try?",
            "I’m hearing a lot of weight in this. You deserve gentleness right now. Could a short break with some music or a warm drink help even 1%?",
        ],
        "anger": [
            "That anger makes sense if things feel unfair. Want to try a 10‑second pause—inhale 4, hold 4, exhale 6—then we can sort what’s in your control?",
            "Your feelings are valid. We can channel this energy. Would listing the top 1–2 triggers help us plan a next step?",
        ],
        "fear": [
            "When worry spikes, your body is trying to protect you. Let’s ground: name 5 things you see, 4 you feel, 3 you hear. I’m with you.",
            "Anxiety can feel loud. Let’s shrink the moment: what’s the next tiny action (30 seconds or less) you could take?",
        ],
        "disgust": [
            "Feeling turned off or disappointed can be protective. If you zoom out, is there a boundary you’d like to set to feel safer?",
            "It’s okay to step back from what doesn’t feel right. What would a kinder environment look like for you today?",
        ],
        "surprise": [
            "That’s a lot to take in at once. Want to unpack it together, one small piece at a time?",
            "Unexpected moments can shake us. What’s one thing you know for sure right now?",
        ],
        "neutral": [
            "Thanks for sharing. I’m here with you. What’s one small action that could make the next hour a bit easier?",
            "I’m listening. If you’d like, we can choose between venting, problem‑solving, or a simple check‑in.",
        ],
        "joy": [
            "I love the hopeful energy here. What helped you get to this point today? Let’s note a small win to carry forward.",
            "That spark matters. What would help you keep this momentum for the next hour?",
        ],
    }
    key = emotion if emotion in templates else "neutral"
    choices = templates[key]
    idx = abs(hash(user_text)) % len(choices)
    return choices[idx]


def analyze_and_respond(user_text: str, clf, emo_clf) -> Dict:
    result = clf(user_text)[0]
    label = result.get("label", "NEUTRAL").upper()
    score = float(result.get("score", 0.5))
    crisis = detect_crisis(user_text)

    emo_label, emo_score = classify_emotion(user_text, emo_clf)

    if crisis:
        reply = build_empathetic_response(label, score, user_text, crisis)
    else:
        if label == "NEGATIVE" or emo_label in {"sadness", "anger", "fear", "disgust"}:
            reply = build_emotion_specific_response(emo_label, user_text)
        elif label == "POSITIVE" or emo_label == "joy":
            reply = build_emotion_specific_response("joy", user_text)
        else:
            reply = build_emotion_specific_response("neutral", user_text)

    return {
        "label": label,
        "score": score,
        "emotion": emo_label,
        "emotion_score": emo_score,
        "crisis": crisis,
        "reply": reply,
    }


# ---------- UI ----------
with st.sidebar:
    st.markdown("**About**")
    st.write(
        "This is a supportive chatbot that uses sentiment analysis to tailor responses. "
        "It is not medical advice. For emergencies, call your local emergency number."
    )
    st.markdown("---")
    st.markdown("**Model**: distilbert-base-uncased-finetuned-sst-2-english")
    st.markdown("**Emotion Model**: j-hartmann/emotion-english-distilroberta-base")

st.title("AI Mental Health Support Bot 💙")
st.caption(
    "Supportive, sentiment-aware responses. Not a substitute for professional help."
)

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm here to listen. What's on your mind today?"}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Type your message…")

if user_input:
    # Echo user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Respond
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                clf = load_sentiment_pipeline()
                emo_clf = load_emotion_pipeline()
                result = analyze_and_respond(user_input, clf, emo_clf)
                label = result["label"]
                score = result["score"]
                crisis = result["crisis"]
                reply = result["reply"]
                emo = result.get("emotion", "neutral")
                emo_score = float(result.get("emotion_score", 0.0))

                meta = f"Sentiment: {label} ({score:.2f}) | Emotion: {emo} ({emo_score:.2f})"
                if crisis:
                    meta += " | possible crisis language detected"

                st.write(reply)
                st.caption(meta)

                if not crisis and (label == "NEGATIVE" or emo in {"sadness", "anger", "fear", "disgust"}):
                    tips = [
                        "Mini reset: inhale 4, hold 4, exhale 6.",
                        "Micro‑action: sip water and roll your shoulders.",
                        "Grounding: name 5 things you can see right now.",
                        "30‑second pause: look out a window or step away from the screen.",
                    ]
                    idx = abs(hash(user_input)) % len(tips)
                    st.toast("You matter. I'm here with you. 💙", icon="✨")
                    st.toast(f"Micro‑boost: {tips[idx]}", icon="🌟")

                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error("Sorry, something went wrong while analyzing your message.")
                st.caption(str(e))
