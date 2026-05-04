import os
import uuid
import time
import requests
import requests as http_requests
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from deepface import DeepFace
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

# ============================================
# 1. CHARGEMENT WHISPER
# ============================================

print("⏳ Chargement du modèle Whisper 'small'...")
model = whisper.load_model("small")
print("✅ Whisper chargé !")

# ============================================
# 2. DRIVER SELENIUM GLOBAL (réutilisé)
# ============================================

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--blink-settings=imagesEnabled=false")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

print("[INFO] Démarrage du driver Selenium...")
selenium_driver = create_driver()
print("[OK] Driver Selenium prêt !")

# ============================================
# 3. CACHE PHARMACIES (1 heure)
# ============================================

pharmacy_cache    = None
cache_timestamp   = 0
CACHE_DURATION    = 3600

# ============================================
# 4. ÉTAT GLOBAL — RECONNAISSANCE FACIALE
# ============================================

patient_en_attente_de_capture = None
patient_en_attente_tablette2  = None

dernier_patient_reconnu = {
    "status": "none", "patient_id": None,
    "nom": None, "prenom": None, "timestamp": 0
}

# URLs services externes
URL_JAVA_RECONNAITRE    = "http://localhost:8082/api/patients/reconnaitre"
URL_JAVA_BIOMETRIE      = "http://localhost:8082/api/patients"
URL_SPRING_RDV          = "http://localhost:8081/api/rdv/patient"
URL_SPRING_CHECKIN      = "http://localhost:8081/api/file-attente/checkin"

JWT_TOKEN = os.getenv("JWT_TOKEN", "")
HEADERS_SPRING = {
    "Authorization": JWT_TOKEN,
    "Content-Type": "application/json"
}

# ============================================
# 5. FONCTIONS UTILITAIRES — DEEPFACE
# ============================================

def extraire_embedding(image_source):
    tmp_path = None
    try:
        if hasattr(image_source, 'read'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image_source.save(tmp.name)
                tmp_path = tmp.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(bytes(image_source))
                tmp_path = tmp.name

        embedding = DeepFace.represent(
            img_path=tmp_path,
            model_name="Facenet",
            enforce_detection=False
        )
        vecteur = embedding[0]['embedding']
        return vecteur, tmp_path
    except Exception as e:
        print(f"❌ Erreur extraction embedding: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, None


def appeler_java_reconnaitre(vecteur):
    try:
        vecteur_bytes = np.array(vecteur, dtype=np.float64).tobytes()
        response = requests.post(
            URL_JAVA_RECONNAITRE,
            data=vecteur_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        print(f"⚠️ Java reconnaitre → HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Erreur Java reconnaissance: {e}")
        return None


def get_rdv_du_jour(patient_id):
    try:
        url = f"{URL_SPRING_RDV}/{patient_id}"
        response = requests.get(url, headers=HEADERS_SPRING, timeout=10)
        if response.status_code == 200:
            rdvs = response.json()
            aujourd_hui = time.strftime("%Y-%m-%d")
            for rdv in rdvs:
                date_rdv = rdv.get('datePrevue')
                if date_rdv:
                    if isinstance(date_rdv, str):
                        date_rdv = date_rdv.split('T')[0]
                    elif hasattr(date_rdv, 'year'):
                        date_rdv = date_rdv.strftime("%Y-%m-%d")
                    if str(date_rdv) == aujourd_hui:
                        return rdv.get('id')
        elif response.status_code == 403:
            print("❌ Token JWT invalide ou expiré !")
        return None
    except Exception as e:
        print(f"❌ get_rdv_du_jour: {e}")
        return None


def faire_checkin_spring(rdv_id):
    try:
        url = f"{URL_SPRING_CHECKIN}/{rdv_id}"
        response = requests.put(url, headers=HEADERS_SPRING, timeout=10)
        if response.status_code == 200:
            print(f"✅ CHECK-IN Spring Boot OK (RDV {rdv_id})")
            return True
        if response.status_code == 403:
            print("❌ Token JWT invalide ou expiré !")
        return False
    except Exception as e:
        print(f"❌ faire_checkin_spring: {e}")
        return False

# ============================================
# 6. FONCTIONS UTILITAIRES — PHARMACIES
# ============================================

PHARMACIES_FALLBACK = [
    {"name": "Pharmacie Al Amal",    "address": "Bd Mohammed V, Oujda",    "phone": "0536-682-411", "garde": "24h/24", "maps": "", "waze": "", "quartier": "Centre-ville"},
    {"name": "Pharmacie Atlas",      "address": "Av. Hassan II, Oujda",     "phone": "0536-703-122", "garde": "Nuit",   "maps": "", "waze": "", "quartier": "Hay Qods"},
    {"name": "Pharmacie Santé Plus", "address": "Rue Berkane, Oujda",       "phone": "0536-688-900", "garde": "Jour",   "maps": "", "waze": "", "quartier": "Lazaret"},
    {"name": "Pharmacie Al Nour",    "address": "Bd El Maghreb El Arabi",   "phone": "0536-712-344", "garde": "24h/24", "maps": "", "waze": "", "quartier": "Sidi Maâfa"},
]

def scrape_pharmacies_oujda():
    global pharmacy_cache, cache_timestamp

    if pharmacy_cache and (time.time() - cache_timestamp) < CACHE_DURATION:
        print("[CACHE] Pharmacies depuis le cache.")
        return pharmacy_cache

    driver = None
    try:
        driver = create_driver()
        driver.get("https://oujda.pharmacieenpermanence.ma/")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/pharmacie-']"))
        )
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
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

            maps_url, waze_url = "", ""
            maps_tag = parent.find("a", href=lambda h: h and "google.com/maps" in str(h))
            if maps_tag:
                maps_url = maps_tag.get("href", "")
            waze_tag = parent.find("a", href=lambda h: h and "waze.com" in str(h))
            if waze_tag:
                waze_url = waze_tag.get("href", "")

            pharmacies.append({
                "name": name, "address": address, "phone": phone,
                "garde": "24h/24", "maps": maps_url, "waze": waze_url,
                "quartier": "Oujda"
            })

        print(f"[SCRAPING] {len(pharmacies)} pharmacies trouvées.")
        if pharmacies:
            pharmacy_cache = pharmacies
            cache_timestamp = time.time()
            return pharmacies
        return None

    except Exception as e:
        print(f"[ERREUR SELENIUM] {str(e)}")
        return None

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# ============================================
# 7. ROUTES — STATUT & WHISPER
# ============================================

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"message": "Microservice IA DOCKT en ligne !", "status": "success"})


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    unique_id = str(uuid.uuid4())
    temp_path = f"audio_{unique_id}.webm"
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Aucun fichier audio"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "Nom de fichier vide"}), 400
        audio_file.save(temp_path)
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return jsonify({"error": "Fichier audio vide"}), 400
        print(f"\n🎙️ Analyse audio {unique_id}...")
        start = time.time()
        result = model.transcribe(temp_path, language="fr", fp16=False)
        texte = result["text"]
        print(f"[OK] Transcription en {round(time.time()-start,2)}s : {texte.strip()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"texte_transcrit": texte.strip()})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

# ============================================
# 8. ROUTES — PHARMACIES & CHATBOT
# ============================================

@app.route('/api/pharmacies', methods=['GET'])
def get_pharmacies():
    pharmacies = scrape_pharmacies_oujda()
    if pharmacies:
        return jsonify({"source": "live", "data": pharmacies})
    return jsonify({"source": "fallback", "data": PHARMACIES_FALLBACK})


GEMINI_KEY = os.getenv("GEMINI_API_KEY")

MED_KEYWORDS = [
    "médicament", "posologie", "dose", "dosage", "comprimé", "cachet",
    "ordonnance", "antibiotique", "sirop", "gélule", "prescription",
    "traitement", "pilule", "vaccin", "injection", "diagnostic"
]

PHARMA_KEYWORDS = ["pharmacie", "garde", "ouverte", "nuit", "urgence", "صيدلية"]

FAQ_MESSAGE = (
    "🌟 Voici ce que je peux faire pour vous :\n\n"
    "📅 **1. Comment prendre un rendez-vous sur DOCKT ?**\n"
    "📝 **2. Comment créer un compte sur DOCKT ?**\n"
    "❌ **3. Comment annuler ou modifier mon rendez-vous ?**\n"
    "✅ **4. Comment faire mon check-in au cabinet ?**\n"
    "🏥 **5. Comment trouver une pharmacie de garde à Oujda ?**\n"
    "🔒 **6. Comment changer mon mot de passe ?**\n"
    "Posez-moi l'une de ces questions et je serai ravi de vous aider ! 😊\n"
    "⚠️ *Urgence médicale ?* Appelez le **150** (SAMU) immédiatement."
)

SYSTEM_PROMPT = """Tu es un assistant administratif pour la plateforme DOCKT à Oujda, Maroc.
Tu réponds en français ou en arabe selon la langue utilisée par l'utilisateur.

Tu PEUX aider avec :
- Expliquer comment prendre un rendez-vous sur DOCKT
- Expliquer comment créer un compte sur DOCKT
- Expliquer comment annuler ou modifier un rendez-vous
- Expliquer comment faire le check-in au cabinet
- Trouver la pharmacie de garde à Oujda
- Expliquer comment changer mon mot de passe
- Répondre aux questions générales sur la plateforme DOCKT

Si une question concerne les médicaments, dosages, diagnostics ou tout sujet médical sensible,
réponds gentiment que tu ne peux pas aider avec ça et oriente vers un médecin ou le 150.

Sois toujours concis, bienveillant et professionnel.
Pour les urgences, rappelle toujours d'appeler le 150 (SAMU)."""


# ============================================
# KEYWORDS
# ============================================
# ============================================
# RÉPONSES STATIQUES PAR QUESTION
# ============================================

REPONSES_STATIQUES = {
    "rendez-vous": (
        "📅 **Comment prendre un rendez-vous sur DOCKT ?**\n\n"
        "1️⃣ Connectez-vous à votre compte sur l'application DOCKT\n"
        "2️⃣ Cliquez sur **'Prendre un rendez-vous'**\n"
        "3️⃣ Choisissez une **date et une heure** qui vous conviennent\n"
        "4️⃣ Confirmez votre rendez-vous\n\n"
        "✅ Vous recevrez une confirmation par notification."
    ),
    "compte": (
        "📝 **Comment créer un compte sur DOCKT ?**\n\n"
        "1️⃣ Ouvrez l'application **DOCKT**\n"
        "2️⃣ Cliquez sur **'Créer un compte'**\n"
        "3️⃣ Renseignez votre **nom, prénom, email et téléphone**\n"
        "4️⃣ Choisissez un **mot de passe sécurisé**\n"
        "5️⃣ Complétez votre **profil médical** (optionnel)\n\n"
        "✅ Votre compte est prêt à être utilisé !"
    ),
    "annuler": (
        "❌ **Comment annuler ou modifier un rendez-vous ?**\n\n"
        "1️⃣ Connectez-vous à votre compte DOCKT\n"
        "2️⃣ Allez dans **'Mes rendez-vous'**\n"
        "3️⃣ Sélectionnez le rendez-vous concerné\n"
        "4️⃣ Cliquez sur **'Annuler'** ou **'Modifier'**\n"
        "5️⃣ Confirmez votre choix\n\n"
        "⚠️ Pensez à annuler au moins **2 heures avant** votre rendez-vous."
    ),
    "check": (
        "✅ **Comment faire mon check-in au cabinet ?**\n\n"
        "1️⃣ Arrivez au cabinet le jour de votre rendez-vous\n"
        "2️⃣ Approchez-vous de la **tablette DOCKT** à l'accueil\n"
        "3️⃣ Placez votre visage devant la **caméra** pour la reconnaissance\n"
        "4️⃣ Le système vous identifie **automatiquement**\n"
        "5️⃣ Votre check-in est confirmé sur l'écran\n\n"
        "✅ Le médecin est notifié de votre arrivée automatiquement."
    ),

    "mot de passe": (
        "🔒 **Comment changer mon mot de passe ?**\n\n"
        "1️⃣ Connectez-vous à votre compte DOCKT\n"
        "2️⃣ Allez dans **'Mon profil'** → **'Paramètres'**\n"
        "3️⃣ Cliquez sur **'Changer le mot de passe'**\n"
        "4️⃣ Saisissez votre **ancien mot de passe**\n"
        "5️⃣ Entrez et confirmez votre **nouveau mot de passe**\n"
        "6️⃣ Cliquez sur **'Enregistrer'**\n\n"
        "✅ Votre mot de passe est mis à jour."
    ),
}

MED_KEYWORDS   = [
    "médicament", "posologie", "dose", "dosage", "comprimé", "cachet",
    "ordonnance", "antibiotique", "sirop", "gélule", "prescription",
    "pilule", "vaccin", "injection", "diagnostic"
]
PHARMA_KEYWORDS = ["pharmacie", "garde", "ouverte", "nuit", "صيدلية"]
SERVICE_KEYWORDS = ["service", "services", "que peux", "que pouvez", "quels sont vos", "aide"]

# ============================================
# ROUTE CHAT
# ============================================

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"reply": "Voici ce que je peux faire pour vous :", "faq": True}), 200

    message       = data.get("message", "").strip()
    message_lower = message.lower()

    # 1️⃣ Sujets médicaux → refus
    if any(k in message_lower for k in MED_KEYWORDS):
        return jsonify({
            "reply": (
                "Je suis désolé, je ne suis pas autorisé à répondre aux questions "
                "sur les médicaments, traitements ou diagnostics médicaux.\n"
                "Pour votre sécurité, consultez un médecin ou appelez le **150** (SAMU)."
            ),
            "faq": True
        })

    # 2️⃣ Services généraux → boutons FAQ
    if any(k in message_lower for k in SERVICE_KEYWORDS):
        return jsonify({"reply": "Voici ce que je peux faire pour vous :", "faq": True})

    # 3️⃣ Pharmacies → cartes
    if any(k in message_lower for k in PHARMA_KEYWORDS):
        pharmacies = scrape_pharmacies_oujda() or PHARMACIES_FALLBACK
        return jsonify({
            "reply": "Pharmacies de garde aujourd'hui à Oujda :",
            "pharmacies": pharmacies[:5],
            "source": "live" if pharmacy_cache else "fallback"
        })

   # ── Dans la route /api/chat, remplace la boucle de recherche par : ──

    # 4️⃣ Réponses statiques — ordre précis, du plus spécifique au plus général
    if "annuler" in message_lower or "modifier" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["annuler"]})

    if "check" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["check"]})

    if "mot de passe" in message_lower or "password" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["mot de passe"]})

    if "compte" in message_lower or "créer" in message_lower or "inscription" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["compte"]})

    if "prendre" in message_lower or "rendez-vous" in message_lower or "rdv" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["prendre"]})

    # 5️⃣ Rien trouvé → boutons FAQ
    return jsonify({
        "reply": "Je n'ai pas bien compris votre question. Voici ce que je peux faire :",
        "faq": True
    }), 200
# ============================================
# 10. ROUTES — TABLETTES
# ============================================

@app.route('/api/visage/demarrer_capture', methods=['POST'])
def demarrer_capture():
    global patient_en_attente_de_capture
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"status": "error", "message": "patient_id requis"}), 400
    patient_en_attente_de_capture = patient_id
    return jsonify({"status": "waiting", "patient_id": patient_id, "message": "Veuillez vous présenter devant la caméra"})


@app.route('/api/visage/verifier_attente', methods=['GET'])
def verifier_attente():
    global patient_en_attente_de_capture
    if patient_en_attente_de_capture:
        return jsonify({"status": "pending", "patient_id": patient_en_attente_de_capture})
    return jsonify({"status": "none"})


@app.route('/api/visage/consommer_attente', methods=['POST'])
def consommer_attente():
    global patient_en_attente_de_capture
    patient_en_attente_de_capture = None
    return jsonify({"status": "ok"})


@app.route('/api/visage/capture_et_associer', methods=['POST'])
def capture_et_associer():
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        vecteur    = data.get('vecteur')
        if vecteur is None or not patient_id:
            return jsonify({"status": "error", "message": "patient_id et vecteur requis"}), 400
        if isinstance(vecteur, np.ndarray):
            vecteur = vecteur.tolist()
        vecteur_bytes = np.array(vecteur, dtype=np.float64).tobytes()
        url = f"{URL_JAVA_BIOMETRIE}/{patient_id}/biometrie"
        response = requests.put(url, data=vecteur_bytes, headers={"Content-Type": "application/octet-stream"}, timeout=10)
        if response.status_code == 200:
            return jsonify({"status": "success", "message": "Biométrie enregistrée"})
        return jsonify({"status": "error", "message": f"Erreur Java: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/demarrer_capture_tablette2', methods=['POST'])
def demarrer_capture_tablette2():
    global patient_en_attente_tablette2
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"status": "error", "message": "patient_id requis"}), 400
    patient_en_attente_tablette2 = patient_id
    return jsonify({"status": "waiting", "patient_id": patient_id})


@app.route('/api/visage/verifier_attente_tablette2', methods=['GET'])
def verifier_attente_tablette2():
    global patient_en_attente_tablette2
    if patient_en_attente_tablette2:
        patient_id = patient_en_attente_tablette2
        patient_en_attente_tablette2 = None
        return jsonify({"status": "pending", "patient_id": patient_id})
    return jsonify({"status": "none"})

# ============================================
# 11. LANCEMENT
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SERVEUR IA DOCKT — EN LIGNE (port 8000)")
    print("="*50)
    print("🎙️  Whisper         : /api/transcribe")
    print("💊  Pharmacies      : /api/pharmacies")
    print("🤖  Chatbot         : /api/chat")
    print("👤  Reconnaitre     : /api/visage/reconnaitre")
    print("✅  Check-in        : /api/visage/checkin")
    print("📡  Poll            : /api/visage/dernier_checkin")
    print("🆕  Tablette 1      : /api/visage/demarrer_capture")
    print("🩺  Tablette 2      : /api/visage/demarrer_capture_tablette2")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8000)