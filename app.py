import os
import json
import io
import re
import subprocess
import html
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, atan2, sqrt
from urllib.parse import quote_plus

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageStat

import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from streamlit_geolocation import streamlit_geolocation

# =========================================================
# OPTIONAL VOICE IMPORTS
# =========================================================

try:
    import speech_recognition as sr
    from streamlit_mic_recorder import mic_recorder
    import imageio_ffmpeg

    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False


# =========================================================
# PAGE CONFIG (must be the first Streamlit call)
# =========================================================

st.set_page_config(
    page_title="AI Crop Health Early Warning",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title { font-size: 34px; font-weight: 700; margin-bottom: 5px; }
    .subtitle { color: #666; font-size: 16px; margin-bottom: 20px; }
    .risk-high { background: #ffdddd; border-left: 6px solid #e00000;
                 padding: 15px; border-radius: 8px; color: #222; }
    .risk-medium { background: #fff2cc; border-left: 6px solid #e0a000;
                   padding: 15px; border-radius: 8px; color: #222; }
    .risk-low { background: #ddffdd; border-left: 6px solid #159615;
                padding: 15px; border-radius: 8px; color: #222; }
    .chat-farmer { background: #e8f0fe; padding: 12px; border-radius: 12px;
                   margin: 8px 0; color: #222; }
    .chat-ai { background: #eaf7ea; padding: 12px; border-radius: 12px;
               margin: 8px 0; color: #222; }
    .info-box { padding: 15px; border-radius: 10px; background: #f5f5f5;
                margin-top: 10px; color: #222; }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

SESSION_DEFAULTS = {
    "last_analysis": None,
    "analysis_history": [],
    "conversation_history": [],
    "assistant_language": "Telugu",
    "pending_speech": None,      # {"text": ..., "language": ...}
    "last_question": "",
    "last_answer": "",
    "last_answer_language": "Telugu",
    "sms_status": "",
    "officer_status": "",
    "db_loaded": False,
}

for _key, _value in SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        # copy lists so that defaults are never shared between sessions
        st.session_state[_key] = list(_value) if isinstance(_value, list) else _value


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "crop_disease_model.pth")
CLASS_NAMES_PATH = os.path.join(PROJECT_DIR, "models", "class_names.json")


# =========================================================
# CLASS NAMES
# =========================================================

DEFAULT_CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight"
]

CLASS_NAMES = DEFAULT_CLASS_NAMES
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            loaded_names = json.load(f)
        if isinstance(loaded_names, list) and loaded_names:
            CLASS_NAMES = loaded_names
    except Exception:
        CLASS_NAMES = DEFAULT_CLASS_NAMES


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        # older PyTorch versions do not support weights_only
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


MODEL_ERROR = None
try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    model = None
    MODEL_ERROR = str(e)


# =========================================================
# IMAGE TRANSFORM + PREDICTION
# =========================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image):
    """Returns (best_class_name, best_confidence_percent, top3_list)."""
    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    k = min(3, probabilities.shape[0])
    top_probs, top_indexes = torch.topk(probabilities, k)

    top3 = []
    for prob, index in zip(top_probs.tolist(), top_indexes.tolist()):
        name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else "Unknown"
        crop_name, disease_name = extract_crop_and_disease(name)
        top3.append({"label": f"{crop_name} - {disease_name}", "confidence": prob * 100})

    best_index = top_indexes[0].item()
    class_name = CLASS_NAMES[best_index] if best_index < len(CLASS_NAMES) else "Unknown"
    return class_name, top_probs[0].item() * 100, top3


def extract_crop_and_disease(class_name):
    parts = class_name.split("___")

    if len(parts) != 2:
        return "Unknown", class_name.replace("_", " ")

    crop = parts[0].replace("_(maize)", "").replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip()
    return crop, disease


# =========================================================
# WEATHER
# =========================================================

def get_weather(latitude, longitude):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
            },
            timeout=8,
        )
        response.raise_for_status()
        current = response.json().get("current", {})

        precipitation = float(current.get("precipitation") or 0)
        rain = float(current.get("rain") or 0)

        return {
            "temperature": float(current.get("temperature_2m", 25)),
            "humidity": float(current.get("relative_humidity_2m", 70)),
            "rainfall": max(precipitation, rain),
            "source": "Open-Meteo",
        }
    except Exception:
        return {
            "temperature": 25.0,
            "humidity": 70.0,
            "rainfall": 0.0,
            "source": "Fallback demo data",
        }


# =========================================================
# RISK + ADVISORY
# =========================================================

def calculate_risk(temperature, humidity, rainfall, pest_level, disease):
    score = 0

    if humidity >= 85:
        score += 30
    elif humidity >= 70:
        score += 20
    elif humidity >= 60:
        score += 10

    if rainfall >= 50:
        score += 25
    elif rainfall >= 20:
        score += 15
    elif rainfall >= 5:
        score += 5

    if 18 <= temperature <= 30:
        score += 20
    elif 15 <= temperature <= 35:
        score += 10

    if pest_level == "High":
        score += 25
    elif pest_level == "Medium":
        score += 15
    else:
        score += 5

    if disease.lower() != "healthy":
        score += 15

    score = min(score, 100)

    if score >= 70:
        risk = "High"
    elif score >= 40:
        risk = "Medium"
    else:
        risk = "Low"

    return risk, score


def get_advisory(crop, disease, risk, temperature, humidity, rainfall):
    if disease.lower() == "healthy":
        return (
            f"The AI model currently indicates that the {crop} leaf appears healthy. "
            f"Continue regular monitoring and maintain proper field management."
        )

    advisory = f"The detected condition is {disease}. Current risk is {risk}. "

    if humidity >= 70:
        advisory += "High humidity may increase the risk of disease development. "

    if rainfall > 5:
        advisory += "Recent rainfall can also create favorable conditions for disease spread. "

    advisory += (
        "Inspect nearby plants, remove severely affected leaves where practical, "
        "avoid unnecessary overhead irrigation, and consult a qualified agriculture "
        "expert before applying pesticides or fungicides."
    )
    return advisory


# =========================================================
# LANGUAGE DETECTION
# =========================================================

TELUGU_WORDS = {
    "enduku", "ela", "panta", "mandu", "neellu", "neeru", "varsham",
    "pakutunda", "cheppandi", "undi", "entha", "vachindi", "cheyyali",
    "vadali", "emi", "ayyindi"
}

HINDI_WORDS = {
    "kyun", "kya", "kaise", "meri", "fasal", "dawai", "paani",
    "baarish", "mausam", "khatra", "hai", "kare", "karein", "kitna"
}


def detect_language(text):
    if not text:
        return "English"

    if len(re.findall(r"[\u0C00-\u0C7F]", text)) >= 2:
        return "Telugu"

    if len(re.findall(r"[\u0900-\u097F]", text)) >= 2:
        return "Hindi"

    tokens = set(re.findall(r"[a-z]+", text.lower()))
    telugu_score = len(tokens & TELUGU_WORDS)
    hindi_score = len(tokens & HINDI_WORDS)

    if telugu_score > hindi_score and telugu_score > 0:
        return "Telugu"
    if hindi_score > telugu_score and hindi_score > 0:
        return "Hindi"

    return "English"


# =========================================================
# INITIAL AI EXPLANATION
# =========================================================

def create_initial_assistant_message(result, language):
    crop = result["crop"]
    disease = result["disease"]
    confidence = result["confidence"]
    risk = result["risk"]
    score = result["risk_score"]
    temperature = result["temperature"]
    humidity = result["humidity"]
    rainfall = result["rainfall"]
    healthy = disease.lower() == "healthy"

    if language == "Telugu":
        base = (
            f"Mee {crop} pantani AI analyze chesindi. "
            + ("Prastutam leaf healthy ga kanipistondi. " if healthy
               else f"{disease} possibility kanipistondi. ")
            + f"AI confidence {confidence:.1f}% undi. "
            f"Current risk {risk}, score {score}/100. "
            f"Temperature {temperature:.1f}°C, humidity {humidity:.0f}%, "
            f"rainfall {rainfall:.1f} mm undi. "
        )
        if healthy:
            return base + "Regular ga crop ni monitor cheyyandi. Meeku emaina doubt unte adagandi."
        return base + (
            "Ee conditions disease risk ni penche chance undi. "
            "Severely affected leaves ni remove cheyyadam, nearby plants ni inspect cheyyadam useful. "
            "Specific pesticide or medicine use cheyyadaniki mundu agriculture expert ni consult cheyyandi. "
            "Meeku emaina doubt unte adagandi."
        )

    if language == "Hindi":
        base = (
            f"AI ne aapki {crop} fasal ko analyze kiya hai. "
            + ("Filhaal leaf healthy dikh rahi hai. " if healthy
               else f"{disease} ki possibility dikh rahi hai. ")
            + f"AI confidence {confidence:.1f}% hai. "
            f"Current risk {risk}, score {score}/100 hai. "
            f"Temperature {temperature:.1f}°C, humidity {humidity:.0f}% "
            f"aur rainfall {rainfall:.1f} mm hai. "
        )
        if healthy:
            return base + "Fasal ko regular monitor karein. Aapko koi doubt ho to pooch sakte hain."
        return base + (
            "Ye conditions disease risk ko badha sakti hain. "
            "Prabhavit leaves ko remove karein aur nearby plants ko inspect karein. "
            "Kisi specific pesticide ya medicine ko use karne se pehle agriculture expert se confirm karein. "
            "Aapko koi doubt ho to pooch sakte hain."
        )

    base = (
        f"I analyzed your {crop} crop. "
        + ("The leaf currently appears healthy. " if healthy
           else f"The image indicates a possibility of {disease}. ")
        + f"AI confidence is {confidence:.1f}%. "
        f"The current risk is {risk}, with a score of {score}/100. "
        f"Temperature is {temperature:.1f}°C, humidity is {humidity:.0f}%, "
        f"and rainfall is {rainfall:.1f} mm. "
    )
    if healthy:
        return base + "Continue regular monitoring. If you have any questions, you can ask me."
    return base + (
        "These conditions may increase disease risk. "
        "Inspect nearby plants and remove severely affected leaves where practical. "
        "Please consult an agriculture expert before applying a specific pesticide or fungicide. "
        "If you have any questions, you can ask me."
    )


# =========================================================
# CONTEXT-AWARE FARMER QUESTION ENGINE
# =========================================================

# Romanised + native-script keywords (Google speech-to-text returns native script).
INTENT_KEYWORDS = {
    "greeting": [
        "hello", "hi", "hey", "namaste", "namaskaram",
        "నమస్కారం", "నమస్తే", "नमस्ते", "नमस्कार"
    ],
    "cause": [
        "why", "cause", "reason", "because", "enduku", "enduku vachindi",
        "ela vachindi", "karanam", "kyun", "kaise hua", "karan",
        "ఎందుకు", "కారణం", "क्यों", "कारण", "कैसे हुआ"
    ],
    "treatment": [
        "medicine", "medication", "treatment", "pesticide", "fungicide", "spray",
        "mandu", "em vadali", "em cheyyali", "ilaaj", "dawai", "dawa", "kya kare",
        "మందు", "చికిత్స", "दवा", "दवाई", "इलाज", "स्प्रे"
    ],
    "water": [
        "water", "watering", "irrigation", "neellu", "neeru", "paani",
        "sinchai", "thadapadam",
        "నీరు", "నీళ్లు", "నీటి", "पानी", "सिंचाई"
    ],
    "risk": [
        "risk", "danger", "serious", "severity", "entha risk", "entha",
        "khatra", "gambhir",
        "ప్రమాదం", "రిస్క్", "खतरा", "जोखिम", "गंभीर"
    ],
    "spread": [
        "spread", "spreading", "pakutunda", "pakuthunda", "pakakunda",
        "control", "prevent", "prevention", "aapali", "phail", "rokna", "bachav",
        "వ్యాప్తి", "వ్యాపించ", "పాకు", "నియంత్రణ", "फैल", "रोक", "बचाव"
    ],
    "weather": [
        "weather", "rain", "rainfall", "temperature", "humidity", "varsham",
        "varshalu", "garmi", "nami", "baarish", "mausam",
        "వాతావరణం", "వర్షం", "తేమ", "ఉష్ణోగ్రత", "मौसम", "बारिश", "नमी", "तापमान"
    ],
    "confidence": [
        "confidence", "sure", "accurate", "accuracy", "nammocha", "namvacha",
        "pakka",
        "నమ్మకం", "నమ్మొచ్చా", "भरोसा", "सटीक", "पक्का"
    ],
    "expert": [
        "expert", "agriculture officer", "officer", "agri officer", "lab", "doctor",
        "నిపుణుడు", "అధికారి", "विशेषज्ञ", "अधिकारी", "एक्सपर्ट"
    ],
    "summary": [
        "what happened", "what is wrong", "problem", "problem enti", "em ayyindi",
        "em jarigindi", "disease enti", "what disease", "healthy", "baagundha",
        "kya hua",
        "సమస్య", "ఏమైంది", "ఏం జరిగింది", "समस्या", "क्या हुआ", "बीमारी"
    ],
}

# Order matters: the first matching intent wins.
INTENT_ORDER = [
    "greeting", "cause", "treatment", "water", "risk",
    "spread", "weather", "confidence", "expert", "summary",
]


def contains_any(text, words):
    """Whole-word match for Latin keywords (so 'hi' does not match 'this'),
    plain substring match for native-script keywords."""
    for word in words:
        if word.isascii():
            pattern = r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        elif word in text:
            return True
    return False


def detect_intent(question):
    q = question.lower().strip()
    for intent in INTENT_ORDER:
        if contains_any(q, INTENT_KEYWORDS[intent]):
            return intent
    return "unknown"


def build_answers(result):
    crop = result["crop"]
    disease = result["disease"]
    risk = result["risk"]
    score = result["risk_score"]
    confidence = result["confidence"]
    t = result["temperature"]
    h = result["humidity"]
    r = result["rainfall"]

    if confidence >= 80:
        conf_text = "high"
    elif confidence >= 60:
        conf_text = "moderate"
    else:
        conf_text = "low"

    return {
        "greeting": {
            "Telugu": (
                f"Namaskaram. Mee {crop} crop analysis na daggara undi. "
                f"{disease} possibility mariyu current risk {risk} ga undi. "
                f"Meeku disease cause, treatment, risk, weather or prevention "
                f"gurinchi edaina doubt unte adagandi."
            ),
            "Hindi": (
                f"Namaste. Aapki {crop} crop analysis mere paas hai. "
                f"{disease} ki possibility aur current risk {risk} hai. "
                f"Aap disease, treatment, risk, weather ya prevention ke baare mein pooch sakte hain."
            ),
            "English": (
                f"Hello. I already have the analysis for your {crop} crop. "
                f"The possible condition is {disease} and the current risk is {risk}. "
                f"You can ask me about the cause, treatment, risk, weather or prevention."
            ),
        },
        "cause": {
            "Telugu": (
                f"{disease} risk ki oka reason favorable weather conditions avvachu. "
                f"Ippudu humidity {h:.0f}% mariyu rainfall {r:.1f} mm undi, temperature {t:.1f}°C. "
                f"High humidity mariyu wet conditions konni crop diseases spread avvadaniki favorable ga untayi. "
                f"Kani image analysis okkate exact cause ni confirm cheyyadu. "
                f"Field inspection tho agriculture expert confirmation better."
            ),
            "Hindi": (
                f"{disease} risk ka ek reason favorable weather conditions ho sakta hai. "
                f"Abhi humidity {h:.0f}% aur rainfall {r:.1f} mm hai, temperature {t:.1f}°C hai. "
                f"High humidity aur wet conditions kuch diseases ke spread ko badha sakti hain. "
                f"Lekin image analysis exact cause confirm nahi karta. "
                f"Field inspection aur agriculture expert verification better rahega."
            ),
            "English": (
                f"One possible reason for the {disease} risk is favorable weather conditions. "
                f"Current humidity is {h:.0f}%, rainfall is {r:.1f} mm, and temperature is {t:.1f}°C. "
                f"High humidity and wet conditions can favor the development or spread of some crop diseases. "
                f"However, image analysis alone cannot confirm the exact cause. "
                f"Field inspection and expert verification are recommended."
            ),
        },
        "treatment": {
            "Telugu": (
                "First, affected plants ni inspect cheyyandi. "
                "Severely affected leaves ni possible ayithe remove cheyyandi. "
                "Unnecessary overhead watering avoid cheyyandi. "
                "Specific pesticide or fungicide name ni image result base chesi direct ga suggest "
                "cheyyadam safe kaadu, endukante disease severity, crop stage mariyu local conditions important. "
                "Nearest agriculture officer or expert tho confirm chesi recommended treatment use cheyyandi."
            ),
            "Hindi": (
                "Sabse pehle affected plants ko inspect karein. "
                "Bahut affected leaves ko possible ho to remove karein. "
                "Unnecessary overhead watering avoid karein. "
                "Sirf image result ke basis par specific pesticide ya fungicide batana safe nahi hai, "
                "kyunki disease severity, crop stage aur local conditions important hain. "
                "Agriculture officer ya expert se confirm karke treatment use karein."
            ),
            "English": (
                "First, inspect the affected plants. "
                "Remove severely affected leaves where practical and avoid unnecessary overhead watering. "
                "I would not recommend a specific pesticide or fungicide only from this image result "
                "because disease severity, crop stage and local conditions matter. "
                "Please confirm the treatment with an agriculture officer or qualified expert."
            ),
        },
        "water": {
            "Telugu": (
                f"Ippudu humidity {h:.0f}% mariyu rainfall {r:.1f} mm undi. "
                f"Already field wet ga unte unnecessary watering avoid cheyyandi. "
                f"Overhead irrigation valla konni leaf diseases spread avvachu. "
                f"Crop stage mariyu soil condition batti irrigation maintain cheyyandi."
            ),
            "Hindi": (
                f"Abhi humidity {h:.0f}% aur rainfall {r:.1f} mm hai. "
                f"Agar field already wet hai to unnecessary watering avoid karein. "
                f"Overhead irrigation kuch leaf diseases ko spread kar sakti hai. "
                f"Crop stage aur soil condition ke according irrigation karein."
            ),
            "English": (
                f"Current humidity is {h:.0f}% and rainfall is {r:.1f} mm. "
                f"If the field is already wet, avoid unnecessary watering. "
                f"Overhead irrigation can increase the spread of some leaf diseases. "
                f"Maintain irrigation according to the crop stage and soil condition."
            ),
        },
        "risk": {
            "Telugu": (
                f"Current crop risk {risk}. Risk score {score}/100. "
                f"AI image confidence {confidence:.1f}%. "
                f"Risk calculation lo humidity {h:.0f}%, rainfall {r:.1f} mm, temperature {t:.1f}°C, "
                f"pest level mariyu disease detection ni consider chesam. "
                f"Risk High ante immediate monitoring and expert verification important."
            ),
            "Hindi": (
                f"Current crop risk {risk} hai. Risk score {score}/100 hai. "
                f"AI image confidence {confidence:.1f}% hai. "
                f"Risk calculation mein humidity {h:.0f}%, rainfall {r:.1f} mm, temperature {t:.1f}°C, "
                f"pest level aur disease detection ko consider kiya gaya hai. "
                f"High risk mein immediate monitoring aur expert verification important hai."
            ),
            "English": (
                f"The current crop risk is {risk}, with a risk score of {score}/100. "
                f"The image model confidence is {confidence:.1f}%. "
                f"The risk assessment considers humidity {h:.0f}%, rainfall {r:.1f} mm, "
                f"temperature {t:.1f}°C, pest level and disease detection. "
                f"If the risk is High, immediate monitoring and expert verification are important."
            ),
        },
        "spread": {
            "Telugu": (
                "Disease spread ni reduce cheyyadaniki affected plants ni regular ga inspect cheyyandi. "
                "Severely infected leaves ni remove cheyyandi, healthy plants ki infection transfer "
                "avvakunda tools and hands clean ga maintain cheyyandi. "
                "Unnecessary overhead watering avoid cheyyandi. "
                "Nearby plants lo same symptoms kanipisthe early ga agriculture expert ki report cheyyandi."
            ),
            "Hindi": (
                "Disease spread ko kam karne ke liye affected plants ko regularly inspect karein. "
                "Severely infected leaves remove karein, healthy plants mein infection transfer "
                "na ho isliye tools aur hands clean rakhein. "
                "Unnecessary overhead watering avoid karein. "
                "Nearby plants mein same symptoms dikhein to agriculture expert ko jaldi report karein."
            ),
            "English": (
                "To reduce possible spread, regularly inspect nearby plants. "
                "Remove severely affected leaves where practical and keep tools and hands clean "
                "to reduce transfer between plants. "
                "Avoid unnecessary overhead watering. "
                "If nearby plants show similar symptoms, report the case to an agriculture expert early."
            ),
        },
        "weather": {
            "Telugu": (
                f"Current weather: temperature {t:.1f}°C, humidity {h:.0f}%, rainfall {r:.1f} mm. "
                f"Ee conditions crop disease risk assessment lo use ayyayi."
            ),
            "Hindi": (
                f"Current weather: temperature {t:.1f}°C, humidity {h:.0f}%, rainfall {r:.1f} mm. "
                f"Ye conditions crop disease risk assessment mein use hui hain."
            ),
            "English": (
                f"Current weather conditions are: temperature {t:.1f}°C, humidity {h:.0f}%, "
                f"and rainfall {r:.1f} mm. These conditions are used in the crop risk assessment."
            ),
        },
        "confidence": {
            "Telugu": (
                f"Image model confidence {confidence:.1f}% undi, ante {conf_text} confidence. "
                f"Idi laboratory confirmation kaadu. "
                f"Image quality, field conditions and disease similarity valla prediction change avvachu. "
                f"Uncertain case ayithe agriculture expert verification ki refer cheyyadam best."
            ),
            "Hindi": (
                f"Image model confidence {confidence:.1f}% hai, yani {conf_text} confidence. "
                f"Ye laboratory confirmation nahi hai. "
                f"Image quality, field conditions aur similar symptoms ki wajah se prediction change ho sakta hai. "
                f"Uncertain case mein agriculture expert verification better hai."
            ),
            "English": (
                f"The image model confidence is {confidence:.1f}%, which is {conf_text} confidence. "
                f"This is not laboratory confirmation. "
                f"Image quality, field conditions and visually similar symptoms can affect the prediction. "
                f"If the case is uncertain, expert verification is recommended."
            ),
        },
        "expert": {
            "Telugu": (
                "AI uncertain ga unte case ni agriculture expert or laboratory verification ki "
                "refer cheyyachu. Farmer → AI → uncertain case → Agriculture Expert → final confirmation "
                "ane workflow mana platform lo use chestunnam."
            ),
            "Hindi": (
                "Agar AI uncertain hai to case ko agriculture expert ya laboratory verification ke liye "
                "refer kiya ja sakta hai. Workflow hai: Farmer → AI → uncertain case → Agriculture Expert "
                "→ final confirmation."
            ),
            "English": (
                "If the AI is uncertain, the case can be referred to an agriculture expert "
                "or laboratory for verification. The workflow is: "
                "Farmer → AI → uncertain case → Agriculture Expert → final confirmation."
            ),
        },
        "summary": {
            "Telugu": (
                f"Mee {crop} image lo {disease} possibility identify ayyindi. "
                f"AI confidence {confidence:.1f}%. Current risk {risk}, score {score}/100. "
                f"Weather: {t:.1f}°C, humidity {h:.0f}%, rainfall {r:.1f} mm."
            ),
            "Hindi": (
                f"Aapki {crop} image mein {disease} ki possibility identify hui hai. "
                f"AI confidence {confidence:.1f}% hai. Current risk {risk}, score {score}/100 hai. "
                f"Weather: {t:.1f}°C, humidity {h:.0f}%, rainfall {r:.1f} mm."
            ),
            "English": (
                f"The image analysis indicates a possibility of {disease} in your {crop}. "
                f"AI confidence is {confidence:.1f}%. "
                f"The current risk is {risk}, with a score of {score}/100. "
                f"Weather: {t:.1f}°C, humidity {h:.0f}%, rainfall {r:.1f} mm."
            ),
        },
        "unknown": {
            "Telugu": (
                f"Nenu mee {crop} crop analysis context tho answer cheyyagalanu. "
                f"Current result {disease}, risk {risk}. "
                f"Meeru ila adagavachu: 'Ee disease enduku vachindi?', 'Medicine em vadali?', "
                f"'Risk entha?', 'Disease spread avvakunda ela control cheyyali?', "
                f"or 'Weather ela undi?'"
            ),
            "Hindi": (
                f"Main aapki {crop} crop analysis ke context mein answer kar sakta hoon. "
                f"Current result {disease}, risk {risk} hai. "
                f"Aap pooch sakte hain: 'Ye disease kyun hui?', 'Kaunsi dawa use karein?', "
                f"'Risk kitna hai?', 'Disease ko kaise rokein?', ya 'Weather kaisa hai?'"
            ),
            "English": (
                f"I can answer questions using your current {crop} crop analysis. "
                f"The current result is {disease} with {risk} risk. "
                f"You can ask things like: 'Why did this disease occur?', "
                f"'What treatment should I consider?', 'How serious is the risk?', "
                f"'How can I reduce the spread?', or 'What is the current weather?'"
            ),
        },
    }


def answer_farmer_question(question, result, language):
    if language not in ("Telugu", "Hindi", "English"):
        language = "English"
    intent = detect_intent(question)
    return build_answers(result)[intent][language]


# =========================================================
# BROWSER AI SPEECH WITH INTERRUPT SUPPORT
# =========================================================

SPEECH_COMPONENT_TEMPLATE = """
<div style="padding:12px;border-radius:10px;background:#eef7ee;
            border:1px solid #c8e6c9;font-family:Arial;color:#222;">
  <b>🤖 AI Voice Assistant</b>
  <div style="margin-top:8px;padding:10px;background:white;border-radius:8px;">__TEXT__</div>
  <div id="status" style="margin-top:8px;color:#555;font-size:14px;">🔊 AI is speaking...</div>
  <button id="stopBtn" style="margin-top:8px;padding:7px 14px;border:none;
          border-radius:7px;cursor:pointer;">🛑 Stop AI</button>

<script>
const text = __TEXT_JSON__;
const language = __LANG_JSON__;
let speaking = true;

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) { el.innerText = msg; }
}

function stopAI() {
  try { window.speechSynthesis.cancel(); } catch (e) {}
  speaking = false;
  setStatus("🎤 AI stopped. Farmer can speak.");
}

document.getElementById("stopBtn").onclick = stopAI;

function speakAI() {
  if (!("speechSynthesis" in window)) {
    setStatus("Browser speech is not supported.");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = language;
  utterance.rate = 0.92;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  utterance.onstart = function () {
    speaking = true;
    setStatus("🔊 AI is speaking... If farmer starts speaking, AI will stop.");
  };
  utterance.onend = function () {
    speaking = false;
    setStatus("🎤 AI finished. Farmer can speak.");
  };
  utterance.onerror = function () { speaking = false; };
  window.speechSynthesis.speak(utterance);
}

// Interrupt detection: stop the AI if the microphone hears speech.
// Note: on speakers (not headphones) the mic may hear the AI itself.
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  try {
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;

    recognition.onresult = function (event) {
      if (!speaking) { return; }
      let detected = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        detected += event.results[i][0].transcript;
      }
      if (detected.trim().length >= 2 && speaking) { stopAI(); }
    };

    recognition.onerror = function () {};
    recognition.onend = function () {
      if (speaking) {
        try { recognition.start(); } catch (e) {}
      }
    };
    recognition.start();
  } catch (e) {}
}

setTimeout(speakAI, 250);
</script>
</div>
"""


def speak_with_interrupt(text, language):
    voice_language = {"Telugu": "te-IN", "Hindi": "hi-IN"}.get(language, "en-IN")

    # json.dumps for JS-safe strings; neutralise "</" so text can never close the <script> tag
    text_json = json.dumps(text).replace("</", "<\\/")

    component_html = (
        SPEECH_COMPONENT_TEMPLATE
        .replace("__TEXT_JSON__", text_json)
        .replace("__LANG_JSON__", json.dumps(voice_language))
        .replace("__TEXT__", html.escape(text))
    )

    components.html(component_html, height=280)


# =========================================================
# AUDIO CONVERSION + SPEECH TO TEXT
# =========================================================

def convert_audio_to_wav(audio_bytes):
    if not VOICE_AVAILABLE:
        return None

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

    attempts = [
        [ffmpeg_path, "-y", "-f", "webm", "-i", "pipe:0",
         "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        [ffmpeg_path, "-y", "-i", "pipe:0",
         "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
    ]

    for command in attempts:
        try:
            result = subprocess.run(
                command,
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
            )
            if result.stdout:
                return io.BytesIO(result.stdout)
        except Exception:
            continue

    return None


LANGUAGE_CODES = {"Telugu": "te-IN", "Hindi": "hi-IN", "English": "en-IN"}


def speech_to_text(audio_bytes, language_mode):
    if not VOICE_AVAILABLE:
        return None, None

    wav_buffer = convert_audio_to_wav(audio_bytes)
    if wav_buffer is None:
        return None, None

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)
    except Exception:
        return None, None

    if language_mode in LANGUAGE_CODES:
        try:
            text = recognizer.recognize_google(
                audio, language=LANGUAGE_CODES[language_mode]
            )
            return text, language_mode
        except Exception:
            return None, None

    # Auto detection: try every language and keep the longest transcript
    candidates = []
    for lang_name, lang_code in LANGUAGE_CODES.items():
        try:
            text = recognizer.recognize_google(audio, language=lang_code)
            if text:
                candidates.append((len(text), text, lang_name))
        except Exception:
            pass

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


# =========================================================
# NEARBY SHOP FINDER HELPERS
# =========================================================

GEOAPIFY_ENDPOINT = "https://api.geoapify.com/v2/places"
MAX_COMMERCIAL_PAGES = 2          # 2 x 500 places; raise if results are missing
PAGE_SIZE = 500

AGRI_KEYWORDS = (
    "agri", "agro", "agricultur", "fertil", "pestic", "insecticide",
    "herbicide", "fungicide", "seed", "beej", "krishi", "kisan", "raitu",
    "farm input", "crop protection", "urea", "manure", "irrigation",
    "sprayer", "agrovet", "plant nursery"
)

AGRI_SHOP_TAGS = {
    "agrarian", "agricultural_supplies", "agriculture", "farm", "seeds",
    "fertilizer", "pesticide", "garden_centre", "garden_center"
}


def get_geoapify_key():
    """Read the key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets["GEOAPIFY_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("GEOAPIFY_API_KEY")


def calculate_distance_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    a = min(1.0, max(0.0, a))
    return 2 * earth_radius_km * atan2(sqrt(a), sqrt(1.0 - a))


@st.cache_data(ttl=600, show_spinner=False)
def find_agri_shops(api_key, latitude, longitude, radius_m):
    def search_places(categories, offset, limit):
        params = {
            "categories": categories,
            "filter": f"circle:{longitude},{latitude},{radius_m}",
            "bias": f"proximity:{longitude},{latitude}",
            "limit": limit,
            "offset": offset,
            "lang": "en",
            "apiKey": api_key,
        }
        response = requests.get(GEOAPIFY_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("features", [])

    # 1) Places that Geoapify itself categorises as agricultural shops
    features = []
    try:
        features.extend(search_places("commercial.agrarian", 0, 100))
    except requests.RequestException:
        pass

    # 2) Wider commercial scan, filtered by name/tag keywords
    for page in range(MAX_COMMERCIAL_PAGES):
        page_features = search_places("commercial", page * PAGE_SIZE, PAGE_SIZE)
        features.extend(page_features)
        if len(page_features) < PAGE_SIZE:
            break

    elements = []
    seen = set()

    for feature in features:
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])

        if len(coords) < 2:
            continue

        raw = props.get("datasource", {}).get("raw", {})
        categories = props.get("categories", [])
        name = props.get("name") or props.get("address_line1") or ""

        raw_shop = str(raw.get("shop", "")).lower()
        raw_trade = str(raw.get("trade", "")).lower()

        searchable = " ".join([
            name,
            props.get("formatted", ""),
            " ".join(categories),
            raw_shop,
            raw_trade,
        ]).lower()

        category_match = any(c.startswith("commercial.agrarian") for c in categories)
        tag_match = raw_shop in AGRI_SHOP_TAGS or raw_trade in AGRI_SHOP_TAGS
        name_match = any(word in searchable for word in AGRI_KEYWORDS)

        if not (category_match or tag_match or name_match):
            continue

        lon = props.get("lon", coords[0])
        lat = props.get("lat", coords[1])
        unique_id = props.get("place_id") or f"{name}:{lat}:{lon}"

        if unique_id in seen:
            continue
        seen.add(unique_id)

        contact = props.get("contact", {}) or {}

        elements.append({
            "type": "node",
            "id": unique_id,
            "lat": lat,
            "lon": lon,
            "tags": {
                "name": name or "Agricultural shop",
                "shop": "agrarian",
                "addr:street": props.get("street", ""),
                "addr:city": props.get("city", ""),
                "addr:postcode": str(props.get("postcode", "")),
                "phone": props.get("phone") or contact.get("phone", ""),
                "website": props.get("website", ""),
                "opening_hours": props.get("opening_hours", ""),
            },
        })

    return elements


def build_shop_list(elements, user_lat, user_lon, search_radius):
    shops = []
    seen = set()

    for item in elements:
        tags = item.get("tags", {})
        center = item.get("center", {})
        shop_lat = item.get("lat", center.get("lat"))
        shop_lon = item.get("lon", center.get("lon"))

        if shop_lat is None or shop_lon is None:
            continue

        shop_lat = float(shop_lat)
        shop_lon = float(shop_lon)

        distance = calculate_distance_km(user_lat, user_lon, shop_lat, shop_lon)
        if distance > search_radius:
            continue

        unique_id = (item.get("type"), item.get("id"))
        if unique_id in seen:
            continue
        seen.add(unique_id)

        address = ", ".join(
            str(tags[key])
            for key in ("addr:street", "addr:suburb", "addr:city")
            if tags.get(key)
        )

        shops.append({
            "name": tags.get("name") or "Unnamed agri-input shop",
            "distance": distance,
            "latitude": shop_lat,
            "longitude": shop_lon,
            "phone": tags.get("phone") or tags.get("contact:phone") or "Not listed",
            "address": address or "Address not listed",
            "opening_hours": tags.get("opening_hours") or "Not listed",
        })

    shops.sort(key=lambda shop: shop["distance"])
    return shops


# =========================================================
# HELPFUL LINKS (shown whenever shops are missing or incomplete)
# =========================================================

def build_help_links(user_lat, user_lon):
    """Return (label, url, description) tuples built from the GPS position."""

    def maps_search(query):
        return (
            "https://www.google.com/maps/search/?api=1&query="
            f"{quote_plus(query)}+near+{user_lat},{user_lon}"
        )

    return [
        (
            "🛒 Fertilizer & pesticide shops (Google Maps)",
            maps_search("fertilizer pesticide shop"),
            "Shops that sell fertilizers, pesticides and fungicides near you. "
            "Google Maps shows ratings, photos, phone numbers and opening hours.",
        ),
        (
            "🌾 Seed & agro-input dealers (Google Maps)",
            maps_search("agro input dealer seeds"),
            "Seed stores and general agri-input dealers. Useful when you also "
            "need seeds, sprayers or irrigation items.",
        ),
        (
            "🏛️ Agriculture office (Google Maps)",
            maps_search("agriculture office"),
            "The nearest government agriculture office, where an agriculture "
            "officer can confirm the disease and suggest the right treatment.",
        ),
        (
            "🔬 Krishi Vigyan Kendra - KVK (Google Maps)",
            maps_search("Krishi Vigyan Kendra"),
            "Farm science centres that give crop-disease advice, training and "
            "demonstrations to farmers.",
        ),
        (
            "🤝 Agricultural cooperative society (Google Maps)",
            maps_search("agricultural cooperative society"),
            "Cooperative societies (PACS) often supply fertilizer and seeds to "
            "member farmers.",
        ),
        (
            "🧪 Soil testing laboratory (Google Maps)",
            maps_search("soil testing laboratory"),
            "Labs that test soil health, so fertilizer is used in the right "
            "quantity.",
        ),
        (
            "🏢 District agriculture officer (Google Maps)",
            maps_search("District Agriculture Officer"),
            "Finds the district / mandal agriculture officer office for official "
            "disease and pest guidance in your area.",
        ),
        (
            "🌦️ IMD weather & warnings (official website)",
            "https://mausam.imd.gov.in/",
            "India Meteorological Department: forecasts, rain and heat warnings. "
            "Check it before spraying or irrigating.",
        ),
        (
            "💰 PM-KISAN (official website)",
            "https://pmkisan.gov.in/",
            "Official PM-KISAN portal: check beneficiary status and payment details.",
        ),
        (
            "🌐 Farmers Portal of India (official website)",
            "https://farmer.gov.in/",
            "Government portal with schemes, advisories and agriculture information "
            "for farmers.",
        ),
        (
            "📞 Kisan Call Centre: 1800-180-1551",
            "tel:18001801551",
            "Toll-free national helpline where agriculture experts answer "
            "farmers' questions in local languages. Tap to call on a phone.",
        ),
    ]


def render_help_links(user_lat, user_lon):
    st.markdown("### 🔎 Find shops & help near you")
    st.caption(
        "Map links open Google Maps around your current location. "
        "Official-website links open in your browser. "
        "The phone link works on mobile phones."
    )

    for label, url, description in build_help_links(user_lat, user_lon):
        with st.container(border=True):
            st.markdown(f"**[{label}]({url})**")
            st.caption(description)


# =========================================================
# SMS ALERT HELPERS (Fast2SMS)
# =========================================================

SMS_API_URL = "https://www.fast2sms.com/dev/bulkV2"


def get_sms_key():
    """Read the Fast2SMS key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets["FAST2SMS_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("FAST2SMS_API_KEY")


def clean_phone_number(raw):
    """Return a 10-digit Indian mobile number, or None if it is invalid."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


def build_sms_text(result):
    shops_link = (
        "https://www.google.com/maps/search/?api=1&query="
        f"fertilizer+pesticide+shop+near+{result['latitude']:.5f},{result['longitude']:.5f}"
    )
    return (
        f"Crop alert: {result['crop']} - {result['disease']}. "
        f"Risk {result['risk']} ({result['risk_score']}/100). "
        "Inspect nearby plants. Consult an agriculture officer before spraying. "
        f"Nearby shops: {shops_link} "
        "Kisan helpline: 1800-180-1551 (free)"
    )


def send_sms_fast2sms(phone, message):
    """Send one SMS. Returns (success, info_text). Never raises."""
    api_key = get_sms_key()
    if not api_key:
        return False, "SMS API key not found. Add FAST2SMS_API_KEY to .streamlit/secrets.toml."

    try:
        response = requests.post(
            SMS_API_URL,
            headers={"authorization": api_key},
            data={
                "route": "q",
                "message": message,
                "language": "english",
                "flash": 0,
                "numbers": phone,
            },
            timeout=20,
        )
        data = response.json()

        if data.get("return") is True:
            return True, f"SMS sent to {phone}."

        return False, f"SMS provider error: {data.get('message', data)}"

    except requests.RequestException as error:
        return False, f"Could not reach the SMS provider: {error}"
    except ValueError:
        return False, "SMS provider returned an unexpected response."


# =========================================================
# FEATURE SETTINGS
# =========================================================

UNKNOWN_CONFIDENCE_THRESHOLD = 40.0   # below this % the image is not accepted
HOTSPOT_MIN_CASES = 3                 # High-risk cases needed to alert the officer
HOTSPOT_RADIUS_KM = 10.0              # cases closer than this form one hotspot
HOTSPOT_DAYS = 7                      # only recent cases count
DB_PATH = os.path.join(PROJECT_DIR, "crop_cases.db")


# =========================================================
# IMAGE QUALITY CHECK
# =========================================================

def check_image_quality(image):
    """Return a short warning text, or an empty string if the image looks fine."""
    width, height = image.size
    if min(width, height) < 100:
        return "The image has a very low resolution. Take a closer, clearer photo."

    brightness = ImageStat.Stat(image.convert("L")).mean[0]
    if brightness < 40:
        return "The image is very dark. Take the photo in better light."
    if brightness > 225:
        return "The image is overexposed. Avoid direct bright sunlight on the leaf."
    return ""


# =========================================================
# DATABASE (SQLite) - cases survive app restarts
# =========================================================

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    try:
        with closing(db_connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    crop TEXT,
                    disease TEXT,
                    confidence REAL,
                    risk TEXT,
                    risk_score INTEGER,
                    temperature REAL,
                    humidity REAL,
                    rainfall REAL,
                    pest_level TEXT,
                    latitude REAL,
                    longitude REAL,
                    expert_notes TEXT DEFAULT '',
                    officer_alerted INTEGER DEFAULT 0
                )
                """
            )
        return True
    except sqlite3.Error:
        return False


def save_case(result):
    """Insert a case and return its id (None if the database is unavailable)."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["created_at"] = created_at
    try:
        with closing(db_connect()) as conn, conn:
            cursor = conn.execute(
                """
                INSERT INTO cases (
                    created_at, crop, disease, confidence, risk, risk_score,
                    temperature, humidity, rainfall, pest_level,
                    latitude, longitude, expert_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at, result["crop"], result["disease"],
                    result["confidence"], result["risk"], result["risk_score"],
                    result["temperature"], result["humidity"], result["rainfall"],
                    result["pest_level"], result["latitude"], result["longitude"],
                    result.get("expert_notes", ""),
                ),
            )
            return cursor.lastrowid
    except sqlite3.Error:
        return None


def load_cases():
    try:
        with closing(db_connect()) as conn:
            rows = conn.execute("SELECT * FROM cases ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def update_expert_notes(case_id, notes):
    try:
        with closing(db_connect()) as conn, conn:
            conn.execute("UPDATE cases SET expert_notes = ? WHERE id = ?", (notes, case_id))
        return True
    except sqlite3.Error:
        return False


def get_recent_high_cases(since_iso):
    try:
        with closing(db_connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, latitude, longitude FROM cases
                WHERE risk = 'High' AND officer_alerted = 0 AND created_at >= ?
                """,
                (since_iso,),
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def mark_officer_alerted(case_ids):
    if not case_ids:
        return
    placeholders = ",".join("?" * len(case_ids))
    try:
        with closing(db_connect()) as conn, conn:
            conn.execute(
                f"UPDATE cases SET officer_alerted = 1 WHERE id IN ({placeholders})",
                list(case_ids),
            )
    except sqlite3.Error:
        pass


# =========================================================
# 7-DAY FORECAST -> EARLY WARNING
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast_days(latitude, longitude):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "hourly": "relative_humidity_2m",
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    humidity_by_day = {}
    for stamp, value in zip(hourly.get("time", []), hourly.get("relative_humidity_2m", [])):
        if value is not None:
            humidity_by_day.setdefault(stamp[:10], []).append(value)

    days = []
    for index, date in enumerate(daily.get("time", [])):
        tmax = daily["temperature_2m_max"][index]
        tmin = daily["temperature_2m_min"][index]
        rain = daily["precipitation_sum"][index]
        humidities = humidity_by_day.get(date, [])

        if tmax is None or tmin is None or not humidities:
            continue

        days.append({
            "date": date,
            "temperature": (tmax + tmin) / 2,
            "humidity": sum(humidities) / len(humidities),
            "rainfall": float(rain or 0),
        })

    return days


def get_forecast_risk(latitude, longitude, pest_level, disease):
    """Risk for each of the next days. Returns [] if the forecast is unavailable."""
    try:
        days = fetch_forecast_days(round(latitude, 3), round(longitude, 3))
    except Exception:
        return []

    forecast = []
    for day in days:
        risk, score = calculate_risk(
            day["temperature"], day["humidity"], day["rainfall"], pest_level, disease
        )
        forecast.append({**day, "risk": risk, "risk_score": score})
    return forecast


# =========================================================
# OFFICER HOTSPOT ALERT
# =========================================================

def find_hotspot_cluster(result):
    """IDs of un-alerted High-risk cases near this one in the last few days."""
    since = (datetime.now(timezone.utc) - timedelta(days=HOTSPOT_DAYS)).isoformat(timespec="seconds")
    cluster = []
    for row in get_recent_high_cases(since):
        distance = calculate_distance_km(
            result["latitude"], result["longitude"], row["latitude"], row["longitude"]
        )
        if distance <= HOTSPOT_RADIUS_KM:
            cluster.append(row["id"])
    return cluster


def build_officer_sms(count, result):
    return (
        f"Hotspot alert: {count} High-risk crop cases within {HOTSPOT_RADIUS_KM:.0f} km of "
        f"{result['latitude']:.3f},{result['longitude']:.3f} in the last {HOTSPOT_DAYS} days. "
        f"Latest: {result['crop']} {result['disease']}. Please inspect."
    )


def check_and_alert_officer(result, officer_phone_raw, enabled):
    """Returns a status text ('' when nothing to report)."""
    if not enabled or result.get("risk") != "High" or not result.get("id"):
        return ""

    cluster_ids = find_hotspot_cluster(result)
    count = len(cluster_ids)

    if count < HOTSPOT_MIN_CASES:
        return (
            f"ℹ️ {count} High-risk case(s) near this location in the last "
            f"{HOTSPOT_DAYS} days. The officer is alerted at {HOTSPOT_MIN_CASES}."
        )

    officer_phone = clean_phone_number(officer_phone_raw)
    if not officer_phone:
        return (
            f"⚠️ Hotspot detected ({count} High-risk cases within {HOTSPOT_RADIUS_KM:.0f} km), "
            "but no valid officer mobile number is set in the sidebar."
        )

    sms_ok, sms_info = send_sms_fast2sms(officer_phone, build_officer_sms(count, result))
    if sms_ok:
        mark_officer_alerted(cluster_ids)
        return (
            f"🚨 Hotspot detected: {count} High-risk cases within "
            f"{HOTSPOT_RADIUS_KM:.0f} km. Officer alerted. {sms_info}"
        )
    return (
        f"❌ Hotspot detected ({count} High-risk cases), but the officer SMS failed: {sms_info}"
    )


# Create the database and load saved cases once per browser session
DB_OK = init_db()
if DB_OK and not st.session_state.db_loaded:
    st.session_state.analysis_history = load_cases()
    st.session_state.db_loaded = True


# =========================================================
# SIDEBAR
# =========================================================

LANGUAGES = ["Telugu", "Hindi", "English"]

st.sidebar.title("🌱 Crop AI")
st.sidebar.markdown("### AI Crop Health Early-Warning System")
st.sidebar.markdown("---")

st.sidebar.subheader("🌐 AI Assistant Language")

assistant_language = st.sidebar.selectbox(
    "Select language",
    LANGUAGES,
    index=LANGUAGES.index(st.session_state.assistant_language)
    if st.session_state.assistant_language in LANGUAGES else 0
)
st.session_state.assistant_language = assistant_language

st.sidebar.markdown("---")
st.sidebar.subheader("🐛 Pest Observation")

pest_level = st.sidebar.selectbox("Current pest level", ["Low", "Medium", "High"])

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Field Location")

latitude = st.sidebar.number_input(
    "Latitude", min_value=-90.0, max_value=90.0, value=17.385000, format="%.6f"
)
longitude = st.sidebar.number_input(
    "Longitude", min_value=-180.0, max_value=180.0, value=78.486700, format="%.6f"
)

st.sidebar.caption(
    "Current prototype uses manual coordinates. "
    "Automatic GPS can be added in the mobile version."
)


st.sidebar.markdown("---")
st.sidebar.subheader("📱 SMS Alerts")

farmer_phone = st.sidebar.text_input(
    "Farmer mobile number",
    placeholder="10-digit number",
    max_chars=15
)

auto_sms = st.sidebar.checkbox(
    "Send SMS automatically when risk is High",
    value=True
)

st.sidebar.caption(
    "The SMS is sent only when a valid number is entered and the risk is High."
)

st.sidebar.markdown("---")
st.sidebar.subheader("👮 Officer Alerts")

officer_phone = st.sidebar.text_input(
    "Agriculture officer mobile number",
    placeholder="10-digit number",
    max_chars=15,
    key="officer_phone"
)

officer_alert_on = st.sidebar.checkbox(
    f"Alert officer when {HOTSPOT_MIN_CASES}+ High-risk cases are within {HOTSPOT_RADIUS_KM:.0f} km",
    value=True
)

st.sidebar.caption("Each SMS uses Fast2SMS credits.")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌱 AI-Powered Real-Time Crop Health Early-Warning & Surveillance Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Early disease detection • Risk prediction • Weather analysis • '
    'Farmer voice assistant • Expert validation • Geospatial monitoring</div>',
    unsafe_allow_html=True
)

if MODEL_LOADED:
    st.success("✅ AI image model loaded successfully")
else:
    st.error(
        "❌ AI image model is not loaded. Please check models/crop_disease_model.pth "
        "and models/class_names.json."
    )
    if MODEL_ERROR:
        st.caption(f"Details: {MODEL_ERROR}")


# =========================================================
# MAIN INPUT
# =========================================================

st.subheader("📷 Crop Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a crop/leaf image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Or take a crop image using camera")

selected_image = camera_image if camera_image is not None else uploaded_file

image = None
if selected_image is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(selected_image)).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")

if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Selected crop image", use_container_width=True)

    with col2:
        st.info("Click Analyze Crop to run the AI model.")

    analyze = st.button("🔍 Analyze Crop", type="primary", use_container_width=True)

    if analyze:
        if not MODEL_LOADED:
            st.error("Model is not available.")
        else:
            try:
                with st.spinner("AI is analyzing the crop..."):
                    predicted_class, confidence, top3 = predict_image(image)
                    quality_note = check_image_quality(image)

                if confidence < UNKNOWN_CONFIDENCE_THRESHOLD:
                    # Not confident enough: do not create a case, ask for a better photo
                    st.warning(
                        "⚠️ The AI could not recognise this as a supported crop leaf "
                        "(Tomato, Potato or Maize). The best match has only "
                        f"{confidence:.1f}% confidence. Please take a new photo: one leaf, "
                        "close-up, good light, plain background."
                    )
                    if quality_note:
                        st.caption(f"Photo tip: {quality_note}")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Possible match": item["label"],
                                 "Confidence": f'{item["confidence"]:.1f}%'}
                                for item in top3
                            ]
                        ),
                        use_container_width=True
                    )
                else:
                    with st.spinner("Fetching weather and 7-day forecast..."):
                        crop, disease = extract_crop_and_disease(predicted_class)

                        weather = get_weather(latitude, longitude)

                        risk, risk_score = calculate_risk(
                            weather["temperature"],
                            weather["humidity"],
                            weather["rainfall"],
                            pest_level,
                            disease
                        )

                        forecast = get_forecast_risk(latitude, longitude, pest_level, disease)

                        advisory = get_advisory(
                            crop, disease, risk,
                            weather["temperature"],
                            weather["humidity"],
                            weather["rainfall"]
                        )

                        result = {
                            "crop": crop,
                            "disease": disease,
                            "confidence": confidence,
                            "risk": risk,
                            "risk_score": risk_score,
                            "temperature": weather["temperature"],
                            "humidity": weather["humidity"],
                            "rainfall": weather["rainfall"],
                            "weather_source": weather["source"],
                            "pest_level": pest_level,
                            "latitude": latitude,
                            "longitude": longitude,
                            "advisory": advisory,
                            "expert_notes": "",
                            "top3": top3,
                            "forecast": forecast,
                            "quality_note": quality_note,
                        }

                        # Save to the database (the id is needed for notes and hotspot alerts)
                        result["id"] = save_case(result)

                        st.session_state.last_analysis = result
                        st.session_state.analysis_history.append(result)

                        # New analysis -> new conversation
                        st.session_state.conversation_history = []
                        st.session_state.last_question = ""
                        st.session_state.last_answer = ""

                        intro = create_initial_assistant_message(
                            result, st.session_state.assistant_language
                        )
                        st.session_state.conversation_history.append(
                            {"role": "AI", "text": intro}
                        )
                        st.session_state.pending_speech = {
                            "text": intro,
                            "language": st.session_state.assistant_language,
                        }
                        st.session_state.last_answer = intro
                        st.session_state.last_answer_language = st.session_state.assistant_language

                    # Automatic SMS to the farmer on High risk (once per analysis)
                    st.session_state.sms_status = ""
                    if result["risk"] == "High" and auto_sms:
                        sms_phone = clean_phone_number(farmer_phone)
                        if sms_phone:
                            with st.spinner("Sending High-risk SMS alert..."):
                                sms_ok, sms_info = send_sms_fast2sms(sms_phone, build_sms_text(result))
                            st.session_state.sms_status = ("✅ " if sms_ok else "❌ ") + sms_info
                        else:
                            st.session_state.sms_status = (
                                "❌ High risk detected, but no SMS was sent: "
                                "enter a valid 10-digit mobile number in the sidebar."
                            )

                    # Hotspot check -> SMS to the agriculture officer
                    st.session_state.officer_status = check_and_alert_officer(
                        result, officer_phone, officer_alert_on
                    )

                    st.success("✅ Crop analysis completed.")
            except Exception as error:
                st.error("The analysis failed. Please try another image.")
                st.caption(f"Details: {error}")


# =========================================================
# DISPLAY LAST ANALYSIS
# =========================================================

result = st.session_state.last_analysis

if result is not None:

    st.markdown("---")
    st.subheader("🧠 AI Analysis Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Crop", result["crop"])
    col2.metric("Detected Condition", result["disease"])
    col3.metric("AI Confidence", f'{result["confidence"]:.1f}%')
    col4.metric("Risk", result["risk"])

    if result["confidence"] < 60:
        st.warning("⚠️ AI confidence is relatively low. Consider expert verification.")

    if result.get("quality_note"):
        st.warning(f"📷 Photo tip: {result['quality_note']}")

    if result.get("top3"):
        with st.expander("🔎 Top predictions"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Prediction": item["label"],
                         "Confidence": f'{item["confidence"]:.1f}%'}
                        for item in result["top3"]
                    ]
                ),
                use_container_width=True
            )

    # ---------------- Weather ----------------
    st.subheader("🌦️ Current Weather")

    w1, w2, w3 = st.columns(3)
    w1.metric("Temperature", f'{result["temperature"]:.1f} °C')
    w2.metric("Humidity", f'{result["humidity"]:.0f} %')
    w3.metric("Rainfall", f'{result["rainfall"]:.1f} mm')

    if result.get("weather_source") == "Fallback demo data":
        st.caption(
            "⚠️ Live weather could not be fetched, so demo values are being used. "
            "The risk score may not reflect real conditions."
        )

    # ---------------- Risk box ----------------
    st.subheader("⚠️ Crop Risk Assessment")

    risk_styles = {
        "High": ("risk-high", "🔴 HIGH RISK"),
        "Medium": ("risk-medium", "🟡 MEDIUM RISK"),
        "Low": ("risk-low", "🟢 LOW RISK"),
    }
    risk_class, risk_label = risk_styles.get(result["risk"], risk_styles["Low"])

    st.markdown(
        f"""
        <div class="{risk_class}">
        <h3>{risk_label}</h3>
        <b>Risk Score:</b> {result["risk_score"]}/100
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- 7-day early warning ----------------
    st.subheader("📅 7-Day Early Warning")

    forecast = result.get("forecast") or []

    if forecast:
        first_high = next((d for d in forecast if d["risk"] == "High"), None)
        first_medium = next((d for d in forecast if d["risk"] == "Medium"), None)

        if first_high:
            st.error(
                f"🔴 Early warning: High disease risk is expected on {first_high['date']} "
                f"(risk score {first_high['risk_score']}/100). Plan inspection and "
                "preventive measures before that date."
            )
        elif first_medium:
            st.warning(
                f"🟡 Medium disease risk is expected from {first_medium['date']}. "
                "Keep monitoring the crop."
            )
        else:
            st.success("🟢 No Medium or High risk is expected in the next 7 days.")

        forecast_df = pd.DataFrame(
            [
                {
                    "Date": d["date"],
                    "Temp (°C)": round(d["temperature"], 1),
                    "Humidity (%)": round(d["humidity"]),
                    "Rain (mm)": round(d["rainfall"], 1),
                    "Risk": d["risk"],
                    "Risk score": d["risk_score"],
                }
                for d in forecast
            ]
        )

        st.bar_chart(forecast_df.set_index("Date")["Risk score"])
        st.dataframe(forecast_df, use_container_width=True)
        st.caption(
            "Forecast risk uses the same rules as the current risk, with the "
            "detected condition and pest level you selected. Forecast data: Open-Meteo."
        )
    else:
        st.caption("The 7-day forecast could not be loaded right now.")

    # ---------------- Advisory ----------------
    st.subheader("💡 Management Advisory")
    st.info(result["advisory"])

    st.subheader("🌐 Multilingual Advisory")

    tab1, tab2, tab3 = st.tabs(["English", "తెలుగు", "हिन्दी"])

    telugu_advisory = (
        f"Mee {result['crop']} pantalo {result['disease']} possibility undi. "
        f"Current risk {result['risk']}. "
        f"Humidity {result['humidity']:.0f}% mariyu rainfall {result['rainfall']:.1f} mm undi. "
        f"Crop ni regular ga monitor cheyyandi mariyu "
        f"specific pesticide use cheyyadaniki mundu agriculture expert ni consult cheyyandi."
    )

    hindi_advisory = (
        f"Aapki {result['crop']} fasal mein {result['disease']} ki possibility hai. "
        f"Current risk {result['risk']} hai. "
        f"Humidity {result['humidity']:.0f}% aur rainfall {result['rainfall']:.1f} mm hai. "
        f"Fasal ko regularly monitor karein aur specific pesticide "
        f"use karne se pehle agriculture expert se consult karein."
    )

    with tab1:
        st.write(result["advisory"])
    with tab2:
        st.write(telugu_advisory)
    with tab3:
        st.write(hindi_advisory)

    # =====================================================
    # AI VOICE ASSISTANT
    # =====================================================

    st.markdown("---")
    st.subheader("🎙️ AI Farmer Voice Assistant")

    st.markdown(
        """
        **Natural conversation mode**

        🤖 AI explains the analysis first  
        ↓  
        👨‍🌾 Farmer can interrupt and ask a question  
        ↓  
        🛑 AI stops speaking  
        ↓  
        🧠 AI understands the question using the current crop context  
        ↓  
        🤖 AI answers in the selected language
        """
    )

    # Speak whatever is queued (intro or typed-question answer)
    pending = st.session_state.pending_speech
    if pending:
        speak_with_interrupt(pending["text"], pending["language"])
        st.session_state.pending_speech = None

    # ---------------- Conversation history ----------------
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation")

        for message in st.session_state.conversation_history:
            safe_text = html.escape(message["text"])
            if message["role"] == "Farmer":
                st.markdown(
                    f'<div class="chat-farmer"><b>👨‍🌾 Farmer:</b><br>{safe_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-ai"><b>🤖 AI:</b><br>{safe_text}</div>',
                    unsafe_allow_html=True
                )

    # ---------------- Text question ----------------
    st.markdown("### ⌨️ Ask the AI")

    with st.form("ask_ai_form", clear_on_submit=True):
        typed_question = st.text_input(
            "Type your question",
            placeholder="Example: disease enduku vachindi? Medicine em vadali?"
        )
        text_ask = st.form_submit_button("💬 Ask AI", use_container_width=True)

    if text_ask and typed_question.strip():
        question = typed_question.strip()

        detected_language = detect_language(question)

        # Telugu/Hindi detected -> use it; otherwise fall back to the English default
        response_language = detected_language if detected_language in LANGUAGES \
            else st.session_state.assistant_language

        answer = answer_farmer_question(question, result, response_language)

        st.session_state.conversation_history.append({"role": "Farmer", "text": question})
        st.session_state.conversation_history.append({"role": "AI", "text": answer})

        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.last_answer_language = response_language
        st.session_state.pending_speech = {"text": answer, "language": response_language}

        st.rerun()

    # ---------------- Voice question ----------------
    st.markdown("### 🎤 Speak to AI")

    if VOICE_AVAILABLE:
        voice_language = st.selectbox(
            "Farmer voice language",
            ["Telugu", "Hindi", "English", "Auto Detect"],
            key="voice_language_select"
        )

        st.caption(
            "🎙️ Start recording your question. "
            "AI's current speech can be stopped using the interruption detection."
        )

        audio = mic_recorder(
            start_prompt="🎤 Start Speaking",
            stop_prompt="⏹️ Stop Speaking",
            just_once=True,
            use_container_width=True,
            format="webm",
            key="farmer_voice_recorder"
        )

        if audio is not None:
            audio_bytes = audio.get("bytes")

            if audio_bytes:
                with st.spinner("🎧 Understanding farmer's speech..."):
                    spoken_text, detected_language = speech_to_text(audio_bytes, voice_language)

                if spoken_text:
                    if voice_language == "Auto Detect":
                        response_language = detected_language or st.session_state.assistant_language
                    else:
                        response_language = voice_language

                    answer = answer_farmer_question(spoken_text, result, response_language)

                    st.session_state.conversation_history.append({"role": "Farmer", "text": spoken_text})
                    st.session_state.conversation_history.append({"role": "AI", "text": answer})

                    st.session_state.last_question = spoken_text
                    st.session_state.last_answer = answer
                    st.session_state.last_answer_language = response_language

                    st.success(f"👨‍🌾 Farmer: {spoken_text}")
                    st.info(f"🤖 AI: {answer}")

                    speak_with_interrupt(answer, response_language)
                else:
                    st.warning("I could not understand the audio. Please speak again clearly.")
    else:
        st.warning(
            "Voice packages are not available. Install SpeechRecognition, "
            "streamlit-mic-recorder and imageio-ffmpeg."
        )

    # ---------------- Replay ----------------
    if st.session_state.last_answer:
        st.markdown("---")

        if st.button("🔊 Replay Last AI Answer"):
            speak_with_interrupt(
                st.session_state.last_answer,
                st.session_state.last_answer_language
            )

    # =====================================================
    # EXPERT VERIFICATION
    # =====================================================

    st.markdown("---")
    st.subheader("👨‍🌾 Expert Verification")

    if result["confidence"] < 60:
        verification_status = "⚠️ Recommended"
    else:
        verification_status = "✅ Optional / Recommended for confirmation"

    st.write(f"**Verification Status:** {verification_status}")

    expert_notes = st.text_area(
        "Agriculture expert notes",
        value=result.get("expert_notes", ""),
        placeholder="Enter expert/laboratory verification notes here..."
    )

    if st.button("📋 Save Expert Verification"):
        result["expert_notes"] = expert_notes
        if result.get("id"):
            update_expert_notes(result["id"], expert_notes)
        st.success("Expert verification notes saved for this prototype session.")


    # =====================================================
    # SMS ALERT
    # =====================================================

    st.markdown("---")
    st.subheader("📱 SMS Alert")

    if result["risk"] == "High":
        st.warning("High risk: an SMS alert is sent automatically when a mobile number is entered.")

    if st.session_state.sms_status:
        st.info(st.session_state.sms_status)

    sms_message = st.text_area(
        "SMS message (you can edit it before sending manually)",
        value=build_sms_text(result),
        max_chars=400,
        key=f"sms_text_{len(st.session_state.analysis_history)}"
    )

    st.caption(
        f"{len(sms_message)} characters. Messages over 160 characters are split "
        "into more than one SMS by the provider."
    )

    if st.button("📤 Send this SMS now"):
        target = clean_phone_number(farmer_phone)
        if not target:
            st.session_state.sms_status = "❌ Enter a valid 10-digit mobile number in the sidebar."
        elif not sms_message.strip():
            st.session_state.sms_status = "❌ The SMS message is empty."
        else:
            with st.spinner("Sending SMS..."):
                sms_ok, sms_info = send_sms_fast2sms(target, sms_message.strip())
            st.session_state.sms_status = ("✅ " if sms_ok else "❌ ") + sms_info
        st.rerun()


# =========================================================
# GEOSPATIAL HOTSPOT MAP (cases analysed in this session)
# =========================================================

st.markdown("---")
st.subheader("🗺️ Crop Disease Hotspot Map")

located_cases = [
    case for case in st.session_state.analysis_history
    if case.get("latitude") is not None and case.get("longitude") is not None
]

if located_cases:
    try:
        centre_lat = sum(c["latitude"] for c in located_cases) / len(located_cases)
        centre_lon = sum(c["longitude"] for c in located_cases) / len(located_cases)

        hotspot_map = folium.Map(
            location=[centre_lat, centre_lon],
            zoom_start=9,
            tiles="OpenStreetMap"
        )

        HeatMap(
            [[c["latitude"], c["longitude"], c["risk_score"] / 100] for c in located_cases],
            radius=25
        ).add_to(hotspot_map)

        marker_colours = {"High": "red", "Medium": "orange", "Low": "green"}

        for case in located_cases:
            folium.CircleMarker(
                location=[case["latitude"], case["longitude"]],
                radius=7,
                color=marker_colours.get(case["risk"], "blue"),
                fill=True,
                fill_opacity=0.8,
                tooltip=f'{case["crop"]} - {case["disease"]} ({case["risk"]})'
            ).add_to(hotspot_map)

        st_folium(
            hotspot_map,
            width=None,
            height=400,
            key="hotspot_map",
            returned_objects=[]
        )
    except Exception as error:
        st.warning("The hotspot map could not be drawn.")
        st.caption(f"Details: {error}")
else:
    st.info("Analyze a crop image to see cases appear on the hotspot map.")


# =========================================================
# AUTOMATIC LOCATION + NEARBY PESTICIDE SHOPS
# =========================================================

st.markdown("---")
st.subheader("📍 Your Location & Nearby Pesticide Shops")

st.write(
    "Allow location access to find agricultural-input shops "
    "within 5 km or 10 km of your current location."
)

location = streamlit_geolocation()

user_lat = None
user_lon = None

if isinstance(location, dict):
    try:
        user_lat = float(location["latitude"])
        user_lon = float(location["longitude"])

        if not (-90 <= user_lat <= 90 and -180 <= user_lon <= 180):
            user_lat = None
            user_lon = None
    except (KeyError, TypeError, ValueError):
        user_lat = None
        user_lon = None

search_radius = st.radio(
    "Search distance from your location",
    [5, 10],
    horizontal=True,
    format_func=lambda x: f"Within {x} km",
    key="pesticide_shop_radius"
)

if user_lat is None or user_lon is None:
    st.warning(
        "📍 Location not available yet. Allow location permission "
        "in your browser and press the location button above. "
        "Use localhost or HTTPS for browser GPS access."
    )

else:
    st.success("Current location detected!")
    st.write(f"**Your GPS coordinates:** {user_lat:.5f}, {user_lon:.5f}")
    st.caption(
        "Search results depend on places listed in Geoapify/OpenStreetMap. "
        "A shop's stock and opening status are not guaranteed."
    )

    radius_m = search_radius * 1000
    api_key = get_geoapify_key()

    if not api_key:
        st.error(
            "Geoapify API key not found. Add GEOAPIFY_API_KEY to "
            ".streamlit/secrets.toml (or set it as an environment variable)."
        )
        render_help_links(user_lat, user_lon)

    else:
        try:
            with st.spinner(f"Searching for agricultural shops within {search_radius} km..."):
                elements = find_agri_shops(
                    api_key,
                    round(user_lat, 6),
                    round(user_lon, 6),
                    radius_m
                )

            shops = build_shop_list(elements, user_lat, user_lon, search_radius)

            with st.expander("Debug nearby places"):
                st.write("GPS latitude:", user_lat)
                st.write("GPS longitude:", user_lon)
                st.write("Search radius:", radius_m, "metres")
                st.write("Matching places returned by API:", len(elements))
                st.write("Shops inside radius:", len(shops))

            # ---------------- Map ----------------
            m = folium.Map(
                location=[user_lat, user_lon],
                zoom_start=12 if search_radius == 5 else 11,
                tiles="OpenStreetMap"
            )

            folium.Marker(
                [user_lat, user_lon],
                tooltip="Your current location",
                popup="You are here",
                icon=folium.Icon(color="blue", icon="user")
            ).add_to(m)

            folium.Circle(
                location=[user_lat, user_lon],
                radius=radius_m,
                color="blue",
                fill=True,
                fill_opacity=0.06,
                tooltip=f"{search_radius} km search radius"
            ).add_to(m)

            for shop in shops:
                popup_text = (
                    f"<b>{html.escape(shop['name'])}</b><br>"
                    f"Distance: {shop['distance']:.2f} km<br>"
                    f"Phone: {html.escape(str(shop['phone']))}<br>"
                    f"Address: {html.escape(shop['address'])}<br>"
                    f"Hours: {html.escape(str(shop['opening_hours']))}"
                )

                folium.Marker(
                    [shop["latitude"], shop["longitude"]],
                    tooltip=f"{shop['name']} - {shop['distance']:.2f} km",
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color="green", icon="shopping-cart")
                ).add_to(m)

            st.markdown("### 🗺️ Nearby Shop Map")

            st_folium(
                m,
                width=None,
                height=500,
                key="nearby_pesticide_shop_map",
                returned_objects=[]
            )

            # ---------------- Shop list ----------------
            if shops:
                st.markdown(f"### 🏪 Shops within {search_radius} km")
                st.success(f"Found {len(shops)} listed shop(s).")

                for index, shop in enumerate(shops, start=1):
                    with st.container(border=True):
                        st.markdown(f"**{index}. {shop['name']}**")
                        st.write(f"📏 **Distance:** {shop['distance']:.2f} km")
                        st.write(f"📍 {shop['address']}")
                        st.write(f"📞 {shop['phone']}")
                        st.write(f"🕒 Opening hours: {shop['opening_hours']}")

                        directions_url = (
                            "https://www.google.com/maps/dir/?api=1"
                            f"&destination={shop['latitude']},{shop['longitude']}"
                        )
                        st.markdown(f"[🧭 Get Directions]({directions_url})")

            render_help_links(user_lat, user_lon)

        except requests.RequestException as error:
            st.error(
                "Could not retrieve nearby shops. Check your internet "
                "connection and API key, then try again."
            )
            st.caption(f"Details: {error}")
            render_help_links(user_lat, user_lon)

        except (ValueError, KeyError, TypeError) as error:
            st.error("Could not process the location or shop data.")
            st.caption(f"Details: {error}")


# =========================================================
# CASE HISTORY
# =========================================================

st.markdown("---")
st.subheader("📋 Crop Analysis History")

if st.session_state.analysis_history:
    history_rows = [
        {
            "Case": index,
            "Time": str(case.get("created_at", ""))[:16].replace("T", " "),
            "Crop": case["crop"],
            "Condition": case["disease"],
            "Confidence": f'{case["confidence"]:.1f}%',
            "Risk": case["risk"],
            "Risk Score": case["risk_score"],
            "Temperature": f'{case["temperature"]:.1f} °C',
            "Humidity": f'{case["humidity"]:.0f}%',
            "Rainfall": f'{case["rainfall"]:.1f} mm',
            "Pest Level": case["pest_level"],
        }
        for index, case in enumerate(st.session_state.analysis_history, start=1)
    ]

    st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
else:
    st.info("No crop cases analyzed yet.")


# =========================================================
# OFFICER DASHBOARD
# =========================================================

st.markdown("---")
st.subheader("👮 Agriculture Officer Dashboard")

history = st.session_state.analysis_history

d1, d2, d3, d4 = st.columns(4)
d1.metric("Total Cases", len(history))
d2.metric("High Risk", sum(1 for c in history if c["risk"] == "High"))
d3.metric("Medium Risk", sum(1 for c in history if c["risk"] == "Medium"))
d4.metric("Low Risk", sum(1 for c in history if c["risk"] == "Low"))

if st.session_state.officer_status:
    st.info(st.session_state.officer_status)

st.markdown(
    """
    <div class="info-box">
    <b>Officer workflow:</b><br><br>
    Farmer reports crop →
    AI analyzes image →
    Risk prediction →
    Cases stored →
    Geographical hotspot detection →
    High-risk areas prioritized →
    Agriculture expert intervention
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SYSTEM WORKFLOW
# =========================================================

st.markdown("---")
st.subheader("🔄 System Workflow")

st.code(
    """
Farmer / Field Data
        ↓
Crop Image + Pest Observation + Weather + Location
        ↓
AI / ML Engine
        ↓
Disease Identification
        +
Weather / Pest Risk Analysis
        ↓
Risk Prediction
Low / Medium / High
        ↓
Early Warning
        ↓
Multilingual Farmer Advisory
        ↓
AI Voice Conversation
        ↓
If AI is uncertain
        ↓
Agriculture Expert / Laboratory
        ↓
Final Verification
        ↓
Geospatial Hotspot Dashboard
        ↓
Agriculture Officer Response
""",
    language="text"
)


# =========================================================
# PROJECT BENEFIT + FOOTER
# =========================================================

st.markdown("---")
st.subheader("🎯 Core Benefit")

st.success(
    "Early Prediction → Early Action → Less Crop Damage → "
    "Better Yield → Reduced Farmer Loss"
)

st.markdown("---")

st.caption("🌱 AI-Powered Crop Health Early-Warning & Surveillance Platform | SIH Prototype")
st.caption(
    "Image Model: MobileNetV2 Transfer Learning | "
    "Prototype Classes: Tomato, Potato and Maize"
)
st.caption(
    "⚠️ This prototype is an early-warning and screening system, "
    "not a replacement for professional agricultural diagnosis."
)