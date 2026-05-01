import os
import uuid
import time
import requests as http_requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import whisper
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])

@app.route('/')
def index():
    return render_template('index.html')

# =============================================
# PARTIE 1 — WHISPER (chargé une seule fois)
# =============================================

print("[INFO] Chargement du modele Whisper en cours...")
model = whisper.load_model("tiny")
print("[OK] Modele charge avec succes !")

# =============================================
# OPTIMISATION 1 — Driver Selenium global
# =============================================

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--blink-settings=imagesEnabled=false")  # pas d'images → plus rapide
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

print("[INFO] Démarrage du driver Selenium...")
selenium_driver = create_driver()  # ← créé UNE FOIS au démarrage
print("[OK] Driver Selenium prêt !")

# =============================================
# OPTIMISATION 2 — Cache pharmacies (1 heure)
# =============================================

pharmacy_cache = None
cache_timestamp = 0
CACHE_DURATION = 3600  # 1 heure en secondes

# =============================================
# PARTIE 2 — WEB SCRAPING PHARMACIES
# =============================================

PHARMACIES_FALLBACK = [
    {"name": "Pharmacie Al Amal",    "address": "Bd Mohammed V, Oujda",     "phone": "0536-682-411", "garde": "24h/24", "maps": "", "quartier": "Centre-ville"},
    {"name": "Pharmacie Atlas",      "address": "Av. Hassan II, Oujda",      "phone": "0536-703-122", "garde": "Nuit",   "maps": "", "quartier": "Hay Qods"},
    {"name": "Pharmacie Santé Plus", "address": "Rue Berkane, Oujda",        "phone": "0536-688-900", "garde": "Jour",   "maps": "", "quartier": "Lazaret"},
    {"name": "Pharmacie Al Nour",    "address": "Bd El Maghreb El Arabi",    "phone": "0536-712-344", "garde": "24h/24", "maps": "", "quartier": "Sidi Maâfa"},
]

def scrape_pharmacies_oujda():
    global pharmacy_cache, cache_timestamp, selenium_driver

    # OPTIMISATION 2 — Retourner le cache si valide
    if pharmacy_cache and (time.time() - cache_timestamp) < CACHE_DURATION:
        print("[CACHE] Pharmacies retournées depuis le cache.")
        return pharmacy_cache

    try:
        # OPTIMISATION 1 — Réutiliser le driver existant
        selenium_driver.get("https://oujda.pharmacieenpermanence.ma/")

        WebDriverWait(selenium_driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/pharmacie-']"))
        )
        time.sleep(2)

        soup = BeautifulSoup(selenium_driver.page_source, "html.parser")

        pharmacies = []
        name_links = soup.find_all("a", href=lambda h: h and "/pharmacie-" in h)

        for link in name_links:
            name_tag = link.find("h3")
            if not name_tag:
                continue
            name = name_tag.get_text(strip=True)

            parent = link.parent
            for _ in range(6):
                if parent and len(parent.get_text(strip=True)) > 80:
                    break
                parent = parent.parent if parent else None

            if not parent:
                continue

            phone = "N/A"
            tel_tag = parent.find("a", href=lambda h: h and "tel:" in str(h))
            if tel_tag:
                phone = tel_tag.get("href", "").replace("tel:", "").strip()

            address = "Oujda"
            skip = ["garde", "nuit", "jour", "24h", "maps", "waze",
                    "voir", "détail", "téléphone", name.lower()]
            for tag in parent.find_all(["p", "span", "div", "li"]):
                txt = tag.get_text(strip=True)
                if (8 < len(txt) < 120
                        and not any(w in txt.lower() for w in skip)
                        and not txt.startswith("053")
                        and not txt.startswith("06")):
                    address = txt
                    break

            maps_url = ""
            maps_tag = parent.find("a", href=lambda h: h and "google.com/maps" in str(h))
            if maps_tag:
                maps_url = maps_tag.get("href", "")

            waze_url = ""
            waze_tag = parent.find("a", href=lambda h: h and "waze.com" in str(h))
            if waze_tag:
                waze_url = waze_tag.get("href", "")

            pharmacies.append({
                "name": name,
                "address": address,
                "phone": phone,
                "garde": "24h/24",
                "maps": maps_url,
                "waze": waze_url,
                "quartier": "Oujda"
            })

        print(f"[SCRAPING] {len(pharmacies)} pharmacies trouvées.")

        if pharmacies:
            # OPTIMISATION 2 — Mettre en cache
            pharmacy_cache = pharmacies
            cache_timestamp = time.time()
            return pharmacies
        return None

    except Exception as e:
        print(f"[ERREUR SELENIUM] {str(e)}")
        # Recréer le driver si planté
        try:
            selenium_driver.quit()
        except:
            pass
        selenium_driver = create_driver()
        return None


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"message": "Le microservice IA DOCKT est en ligne !", "status": "success"})


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    unique_id = str(uuid.uuid4())
    temp_path = f"audio_{unique_id}.webm"
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Aucun fichier audio n'a été envoyé"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "Le nom du fichier est vide"}), 400
        audio_file.save(temp_path)
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return jsonify({"error": "Le fichier audio est vide"}), 400
        print(f"[INFO] Analyse audio {unique_id}...")
        start_time = time.time()
        result = model.transcribe(temp_path, language="fr", fp16=False)
        texte_transcrit = result["text"]
        print(f"[OK] Transcription en {round(time.time()-start_time, 2)}s : {texte_transcrit.strip()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"texte_transcrit": texte_transcrit.strip()})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route('/api/pharmacies', methods=['GET'])
def get_pharmacies():
    pharmacies = scrape_pharmacies_oujda()
    if pharmacies:
        return jsonify({"source": "live", "data": pharmacies})
    return jsonify({"source": "fallback", "data": PHARMACIES_FALLBACK})


# =============================================
# PARTIE 3 — CHATBOT (Gemini)
# =============================================

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

MED_KEYWORDS = ["médicament", "posologie", "dose", "dosage", "comprimé",
                 "ordonnance", "antibiotique", "sirop", "gélule", "prescription"]

PHARMA_KEYWORDS = ["pharmacie", "garde", "ouverte", "nuit", "urgence", "صيدلية"]

SYSTEM_PROMPT = """Tu es un assistant médical pour les patients à Oujda, Maroc.
Tu réponds en français ou en arabe selon la langue utilisée.
Tu peux répondre aux questions générales de santé : symptômes, hygiène, premiers secours.
Tu REFUSES toute question sur les médicaments, dosages ou prescriptions.
Tu es concis, bienveillant et professionnel.
En cas d'urgence, rappelle d'appeler le 150 (SAMU)."""


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Champ 'message' manquant"}), 400

    message = data.get("message", "").strip()
    message_lower = message.lower()

    if any(k in message_lower for k in MED_KEYWORDS):
        return jsonify({
            "reply": "⚠️ Je ne suis pas autorisé à donner des conseils sur les médicaments. "
                     "Consultez un pharmacien ou appelez le 150."
        })

    if any(k in message_lower for k in PHARMA_KEYWORDS):
        pharmacies = scrape_pharmacies_oujda() or PHARMACIES_FALLBACK
        result = "🏥 Pharmacies de garde aujourd'hui à Oujda :\n\n"
        for p in pharmacies[:3]:
            maps_link = f"\n🗺️ {p['maps']}" if p.get('maps') else ""
            result += f"🏥 {p['name']}\n📍 {p['address']}\n📞 {p['phone']} — {p['garde']}{maps_link}\n\n"
        result += "⚠️ Appelez avant de vous déplacer. Urgence : 150"
        return jsonify({"reply": result})

    if not GEMINI_KEY:
        return jsonify({"error": "Clé API Gemini non configurée"}), 500

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    )
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": message}]}]
    }
    try:
        res = http_requests.post(url, json=body, timeout=30)
        res.raise_for_status()
        reply = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"reply": reply})
    except http_requests.exceptions.Timeout:
        return jsonify({"reply": "⏱️ Délai dépassé. Réessayez."}), 504
    except Exception as e:
        return jsonify({"reply": f"Erreur API : {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=False, port=8000)  # ← debug=False évite le double chargement de Whisper