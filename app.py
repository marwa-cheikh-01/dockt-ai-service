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
        return jsonify({"reply": "⚠️ Je ne suis pas autorisé à donner des conseils sur les médicaments. Consultez un pharmacien ou appelez le 150."})

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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
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

# ============================================
# 9. ROUTES — RECONNAISSANCE FACIALE
# ============================================

@app.route('/api/visage/status', methods=['GET'])
def visage_status():
    return jsonify({"status": "success", "deepface_ready": True})


@app.route('/api/visage/extraire_vecteur', methods=['POST'])
def extraire_vecteur():
    tmp_path = None
    try:
        if 'image' in request.files:
            vecteur, tmp_path = extraire_embedding(request.files['image'])
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({"status": "error", "message": "Aucune image reçue"}), 400
            vecteur, tmp_path = extraire_embedding(data['image'])
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if vecteur is None:
            return jsonify({"status": "error", "message": "Aucun visage détecté"}), 500
        return jsonify({"status": "success", "vecteur": vecteur, "taille": len(vecteur)})
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/reconnaitre', methods=['POST'])
def reconnaitre_visage():
    tmp_path = None
    try:
        if 'image' in request.files:
            vecteur, tmp_path = extraire_embedding(request.files['image'])
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({"status": "error", "message": "Aucune image reçue"}), 400
            vecteur, tmp_path = extraire_embedding(data['image'])
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if vecteur is None:
            return jsonify({"status": "error", "message": "Aucun visage détecté"}), 500
        patient = appeler_java_reconnaitre(vecteur)
        if patient:
            patient_id = patient.get('idPatient') or patient.get('id')
            nom    = patient.get('nom', '')
            prenom = patient.get('prenom', '')
            return jsonify({
                "status": "success", "vecteur": vecteur, "taille": len(vecteur),
                "patient_id": patient_id, "nom": nom, "prenom": prenom
            })
        return jsonify({
            "status": "inconnu", "message": "Patient non reconnu", "vecteur": vecteur
        }), 404
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/checkin', methods=['POST'])
def checkin():
    global dernier_patient_reconnu
    try:
        data = request.get_json()
        if not data or 'patient_id' not in data:
            return jsonify({"status": "error", "message": "patient_id requis"}), 400
        patient_id = data['patient_id']
        nom    = data.get('nom', '')
        prenom = data.get('prenom', '')
        rdv_id = get_rdv_du_jour(patient_id)
        if not rdv_id:
            return jsonify({"status": "no_rdv", "message": "Aucun RDV aujourd'hui", "patient_id": patient_id}), 200
        succes = faire_checkin_spring(rdv_id)
        dernier_patient_reconnu = {
            "status": "reconnu" if succes else "checkin_failed",
            "patient_id": patient_id, "nom": nom, "prenom": prenom,
            "rdv_id": rdv_id, "timestamp": time.time()
        }
        if succes:
            return jsonify({"status": "success", "message": "Check-in effectué !", "patient_id": patient_id, "rdv_id": rdv_id})
        return jsonify({"status": "error", "message": "Erreur check-in Spring Boot"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/dernier_checkin', methods=['GET'])
def dernier_checkin():
    global dernier_patient_reconnu
    resultat = dernier_patient_reconnu.copy()
    if resultat['status'] != 'none' and time.time() - resultat['timestamp'] > 30:
        dernier_patient_reconnu = {"status": "none", "patient_id": None, "nom": None, "prenom": None, "timestamp": 0}
        return jsonify({"status": "none"})
    if resultat['status'] == 'reconnu':
        dernier_patient_reconnu = {"status": "none", "patient_id": None, "nom": None, "prenom": None, "timestamp": 0}
    return jsonify(resultat)


@app.route('/api/visage/verifier', methods=['POST'])
def verifier_visage():
    path1 = path2 = None
    try:
        data = request.get_json()
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({"status": "error", "message": "Deux images requises"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
            tmp1.write(bytes(data['image1'])); path1 = tmp1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
            tmp2.write(bytes(data['image2'])); path2 = tmp2.name
        result = DeepFace.verify(img1_path=path1, img2_path=path2, model_name="Facenet", enforce_detection=False)
        for p in [path1, path2]:
            if p and os.path.exists(p): os.unlink(p)
        return jsonify({"status": "success", "verifie": result['verified'], "distance": result['distance']})
    except Exception as e:
        for p in [path1, path2]:
            if p and os.path.exists(p): os.unlink(p)
        return jsonify({"status": "error", "message": str(e)}), 500

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