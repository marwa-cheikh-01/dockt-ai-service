import os
import uuid
import time
import base64
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

pharmacy_cache  = None
cache_timestamp = 0
CACHE_DURATION  = 3600

# ============================================
# 4. ÉTAT GLOBAL — RECONNAISSANCE FACIALE
# ============================================

patient_en_attente_de_capture = None
patient_en_attente_tablette2  = None

dernier_patient_reconnu = {
    "status": "none", "patient_id": None,
    "nom": None, "prenom": None, "timestamp": 0
}

consultation_en_cours = {}

# Guard anti-doublon + accumulateur de vecteurs
_biometrie_enregistree    = set()   # patients déjà enregistrés cette session
_vecteurs_en_attente      = {}      # { patient_id: [vecteur1, vecteur2, ...] }
_NB_VECTEURS_POUR_MOYENNE = 2       # on attend 2 captures pour faire la moyenne

# ============================================
# 5. URLs & HEADERS
# ============================================

URL_JAVA_RECONNAITRE           = "http://localhost:8082/api/patients/reconnaitre"
URL_JAVA_BIOMETRIE             = "http://localhost:8082/api/patients"
URL_SPRING_RDV                 = "http://localhost:8081/api/rdv/patient"
URL_SPRING_CHECKIN             = "http://localhost:8081/api/file-attente/checkin"
URL_SPRING_STATUT_CONSULTATION = "http://localhost:8081/api/file-attente/statut-consultation"

JWT_TOKEN = os.getenv("JWT_TOKEN", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

HEADERS_SPRING = {
    "Authorization": JWT_TOKEN,
    "Content-Type": "application/json"
}

# ============================================
# 6. FONCTIONS UTILITAIRES — DEEPFACE
# ============================================

def extraire_embedding(image_source):
    """
    Extrait un vecteur Facenet depuis une image.
    Accepte FileStorage Flask ou bytes bruts.
    Retourne (vecteur_list, tmp_path) ou (None, None).
    """
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
    """
    Envoie le vecteur (bytes float64) au MS1 Java sur /reconnaitre.
    Retourne dict patient ou None.
    """
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
        elif response.status_code == 404:
            return None
        else:
            print(f"⚠️ Java reconnaitre → HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erreur Java reconnaissance: {e}")
        return None


def get_rdv_du_jour(patient_id):
    """
    Retourne le rdv_id du RDV d'aujourd'hui pour ce patient.
    """
    try:
        url = f"{URL_SPRING_RDV}/{patient_id}"
        print(f"\n📡 [get_rdv_du_jour] GET {url}")
        response = requests.get(url, headers=HEADERS_SPRING, timeout=10)
        print(f"📡 [get_rdv_du_jour] Status: {response.status_code}")

        if response.status_code == 200:
            rdvs = response.json()
            aujourd_hui = time.strftime("%Y-%m-%d")
            print(f"📅 Aujourd'hui: {aujourd_hui} | RDVs: {len(rdvs)}")

            for rdv in rdvs:
                date_rdv = rdv.get('datePrevue')
                rdv_id   = rdv.get('id')
                if date_rdv:
                    if isinstance(date_rdv, str):
                        date_rdv = date_rdv.split('T')[0]
                    elif hasattr(date_rdv, 'year'):
                        date_rdv = date_rdv.strftime("%Y-%m-%d")
                    if str(date_rdv) == aujourd_hui:
                        print(f"   ✅ RDV DU JOUR TROUVÉ: ID={rdv_id}")
                        return rdv_id

            print("   ⚠️ Aucun RDV aujourd'hui")

        elif response.status_code == 403:
            print("   ❌ ERREUR 403: Token JWT invalide ou expiré !")
        else:
            print(f"   ❌ Erreur HTTP {response.status_code}")
        return None

    except Exception as e:
        print(f"❌ [get_rdv_du_jour] Exception: {e}")
        return None


def faire_checkin_spring(rdv_id):
    """
    Check-in Spring Boot MS2 (tablette 1).
    Retourne (True, 'success') | (False, 'already_checkin') | (False, 'error')
    """
    try:
        url = f"{URL_SPRING_CHECKIN}/{rdv_id}"
        print(f"📡 [faire_checkin_spring] PUT {url}")
        response = requests.put(url, headers=HEADERS_SPRING, timeout=10)
        print(f"📡 [faire_checkin_spring] Status: {response.status_code}")

        if response.status_code == 200:
            print(f"✅ CHECK-IN OK (RDV {rdv_id})")
            return True, "success"

        if response.status_code == 403:
            print("   ❌ ERREUR 403: Token JWT invalide ou expiré !")
            return False, "error"

        if response.status_code == 500:
            try:
                error_body = response.text
                print(f"   ⚠️ Body erreur: {error_body[:200]}")
                if "déjà effectué" in error_body or "already" in error_body.lower():
                    print("   ⚠️ Check-in déjà effectué pour ce patient !")
                    return False, "already_checkin"
            except Exception:
                pass
            return False, "error"

        print(f"⚠️ Erreur check-in → HTTP {response.status_code}")
        return False, "error"

    except Exception as e:
        print(f"❌ [faire_checkin_spring] Exception: {e}")
        return False, "error"


def update_statut_consultation(rdv_id, statut):
    """
    Met à jour le statut de consultation dans MS2.
    """
    try:
        url = f"{URL_SPRING_STATUT_CONSULTATION}/{rdv_id}?statutConsultation={statut}"
        print(f"\n📡 [update_statut] PUT {url}")
        response = requests.put(url, headers=HEADERS_SPRING, timeout=10)
        print(f"📡 [update_statut] HTTP: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Statut → {statut} (RDV {rdv_id})")
            return True
        elif response.status_code == 404:
            print(f"❌ RDV {rdv_id} introuvable dans MS2")
        else:
            print(f"⚠️ HTTP {response.status_code}: {response.text[:200]}")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ MS2 Spring Boot injoignable sur localhost:8081")
        return False
    except Exception as e:
        print(f"❌ [update_statut] Exception: {e}")
        return False


def get_vecteur_depuis_bd(patient_id):
    """
    Récupère imageBiometrique (bytea) du patient depuis MS1.
    Spring Boot retourne le champ byte[] en Base64 automatiquement.
    Retourne np.array float64 ou None.
    """
    try:
        url = f"http://localhost:8082/api/patients/{patient_id}"
        print(f"\n📡 [get_vecteur_bd] GET {url}")
        response = requests.get(url, timeout=10)
        print(f"📡 [get_vecteur_bd] HTTP {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Patient {patient_id} introuvable dans MS1")
            return None

        data = response.json()
        image_bio = data.get('imageBiometrique')

        if not image_bio:
            print(f"⚠️ Pas d'image biométrique pour patient {patient_id}")
            return None

        raw_bytes = base64.b64decode(image_bio)
        vecteur   = np.frombuffer(raw_bytes, dtype=np.float64)
        print(f"✅ Vecteur BDD : {len(vecteur)} dimensions")
        return vecteur

    except Exception as e:
        print(f"❌ [get_vecteur_bd] Exception: {e}")
        return None


def calculer_similarite_cosinus(v1, v2):
    """
    Similarité cosinus entre deux vecteurs numpy.
    1.0 = identiques | 0.0 = perpendiculaires | -1.0 = opposés
    """
    try:
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)
        norme1 = np.linalg.norm(v1)
        norme2 = np.linalg.norm(v2)
        if norme1 == 0 or norme2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norme1 * norme2))
    except Exception as e:
        print(f"❌ [similarite_cosinus] Exception: {e}")
        return 0.0

# ============================================
# 7. FONCTIONS UTILITAIRES — PHARMACIES
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
# 8. ROUTES — STATUT & WHISPER
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
        print(f"\n🎙️ Dictée reçue !")
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
# 9. ROUTES — PHARMACIES & CHATBOT
# ============================================

@app.route('/api/pharmacies', methods=['GET'])
def get_pharmacies():
    pharmacies = scrape_pharmacies_oujda()
    if pharmacies:
        return jsonify({"source": "live", "data": pharmacies})
    return jsonify({"source": "fallback", "data": PHARMACIES_FALLBACK})


MED_KEYWORDS = [
    "médicament", "posologie", "dose", "dosage", "comprimé", "cachet",
    "ordonnance", "antibiotique", "sirop", "gélule", "prescription",
    "traitement", "pilule", "vaccin", "injection", "diagnostic"
]

PHARMA_KEYWORDS  = ["pharmacie", "garde", "ouverte", "nuit", "urgence", "صيدلية"]
SERVICE_KEYWORDS = ["service", "services", "que peux", "que pouvez", "quels sont vos", "aide"]

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
        "⚠️ Pensez à annuler au moins **24 heures avant** votre rendez-vous."
    ),
    "check": (
        "✅ **Comment faire mon check-in au cabinet ?**\n\n"
        "1️⃣ Arrivez au cabinet le jour de votre rendez-vous\n"
        "2️⃣ Approchez-vous de la **tablette DOCKT** à l'accueil\n"
        "3️⃣ Placez votre visage devant la **caméra** pour la reconnaissance\n"
        "4️⃣ Le système vous identifie **automatiquement**\n"
        "5️⃣ Votre check-in est confirmé sur l'écran\n\n"
        "✅ Le médecin est notifié de votre arrivée automatiquement."
    )
}


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

    # 4️⃣ Réponses statiques — ordre précis, du plus spécifique au plus général
    if "annuler" in message_lower or "modifier" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["annuler"]})

    if "check" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["check"]})

    if "compte" in message_lower or "créer" in message_lower or "inscription" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["compte"]})

    if "prendre" in message_lower or "rendez-vous" in message_lower or "rdv" in message_lower:
        return jsonify({"reply": REPONSES_STATIQUES["rendez-vous"]})

    # 5️⃣ Rien trouvé → boutons FAQ
    return jsonify({
        "reply": "Je n'ai pas bien compris votre question. Voici ce que je peux faire :",
        "faq": True
    }), 200

# ============================================
# 10. ROUTES — RECONNAISSANCE FACIALE
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

        print(f"📐 Vecteur extrait ({len(vecteur)} dims)")
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
            print(f"👤 Patient reconnu : {prenom} {nom} (ID: {patient_id})")
            return jsonify({
                "status": "success", "vecteur": vecteur, "taille": len(vecteur),
                "patient_id": patient_id, "nom": nom, "prenom": prenom
            })
        else:
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
        print("\n" + "=" * 50)
        print("✅ CHECK-IN TABLETTE 1")
        print("=" * 50)

        data = request.get_json()
        print(f"📥 Données reçues: {data}")

        if not data or 'patient_id' not in data:
            print("❌ patient_id manquant")
            return jsonify({"status": "error", "message": "patient_id requis"}), 400

        patient_id = data['patient_id']
        nom    = data.get('nom', '')
        prenom = data.get('prenom', '')
        print(f"👤 Patient: {prenom} {nom} (ID: {patient_id})")

        print("🔍 Étape 1: Recherche du RDV du jour...")
        rdv_id = get_rdv_du_jour(patient_id)

        if not rdv_id:
            print("⚠️ Aucun RDV trouvé pour aujourd'hui")
            return jsonify({
                "status": "no_rdv",
                "message": "Aucun RDV aujourd'hui",
                "patient_id": patient_id
            }), 200

        print(f"✅ RDV trouvé: {rdv_id}")

        print("🔍 Étape 2: Appel Spring Boot check-in...")
        succes, raison = faire_checkin_spring(rdv_id)

        dernier_patient_reconnu = {
            "status":     "reconnu" if succes else raison,
            "patient_id": patient_id,
            "nom":        nom,
            "prenom":     prenom,
            "rdv_id":     rdv_id,
            "timestamp":  time.time()
        }

        if succes:
            print("✅ CHECK-IN RÉUSSI")
            return jsonify({
                "status": "success",
                "message": "Check-in effectué !",
                "patient_id": patient_id,
                "rdv_id": rdv_id
            })
        elif raison == "already_checkin":
            print("⚠️ CHECK-IN DÉJÀ EFFECTUÉ")
            return jsonify({
                "status": "already_checkin",
                "message": "Vous avez déjà effectué votre check-in aujourd'hui !",
                "patient_id": patient_id,
                "rdv_id": rdv_id
            }), 200
        else:
            print("❌ CHECK-IN ÉCHOUÉ")
            return jsonify({
                "status": "error",
                "message": "Erreur check-in Spring Boot"
            }), 500

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE checkin: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/dernier_checkin', methods=['GET'])
def dernier_checkin():
    global dernier_patient_reconnu
    resultat = dernier_patient_reconnu.copy()
    if resultat['status'] != 'none' and time.time() - resultat['timestamp'] > 30:
        dernier_patient_reconnu = {
            "status": "none", "patient_id": None,
            "nom": None, "prenom": None, "timestamp": 0
        }
        return jsonify({"status": "none"})
    if resultat['status'] == 'reconnu':
        dernier_patient_reconnu = {
            "status": "none", "patient_id": None,
            "nom": None, "prenom": None, "timestamp": 0
        }
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
        result = DeepFace.verify(
            img1_path=path1, img2_path=path2,
            model_name="Facenet", enforce_detection=False
        )
        for p in [path1, path2]:
            if p and os.path.exists(p): os.unlink(p)
        return jsonify({
            "status": "success", "verifie": result['verified'], "distance": result['distance']
        })
    except Exception as e:
        for p in [path1, path2]:
            if p and os.path.exists(p): os.unlink(p)
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================
# 11. ROUTES — TABLETTE 1 (capture biométrique)
# ============================================

@app.route('/api/visage/demarrer_capture', methods=['POST'])
def demarrer_capture():
    global patient_en_attente_de_capture
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"status": "error", "message": "patient_id requis"}), 400
    patient_en_attente_de_capture = patient_id
    print(f"📱 Patient {patient_id} en attente de capture (tablette 1)")
    return jsonify({"status": "waiting", "patient_id": patient_id,
                    "message": "Veuillez vous présenter devant la caméra"})


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
    global _biometrie_enregistree, _vecteurs_en_attente

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Body JSON requis"}), 400

        patient_id = str(data.get('patient_id'))
        vecteur    = data.get('vecteur')

        print(f"\n{'=' * 50}")
        print(f"📊 CAPTURE ET ASSOCIER - NOUVEAU PATIENT")
        print(f"{'=' * 50}")
        print(f"👤 Patient ID reçu: {patient_id}")
        print(f"📐 Vecteur reçu: {type(vecteur)}")

        if vecteur is None or not patient_id:
            return jsonify({"status": "error", "message": "patient_id et vecteur requis"}), 400

        if isinstance(vecteur, list):
            print(f"   Dimensions: {len(vecteur)}")
            print(f"   Premieres valeurs: {vecteur[:3]}")
        elif isinstance(vecteur, np.ndarray):
            print(f"   Type numpy: {vecteur.dtype}")
            vecteur = vecteur.tolist()

        # Guard anti-doublon
        if patient_id in _biometrie_enregistree:
            print(f"⚠️ Biométrie déjà enregistrée pour patient {patient_id} — ignoré")
            return jsonify({
                "status":  "success",
                "message": "Biométrie déjà enregistrée (doublon ignoré)"
            })

        # Accumuler les vecteurs et calculer la moyenne
        if patient_id not in _vecteurs_en_attente:
            _vecteurs_en_attente[patient_id] = []

        _vecteurs_en_attente[patient_id].append(vecteur)
        nb = len(_vecteurs_en_attente[patient_id])
        print(f"   📦 Vecteurs accumulés pour patient {patient_id}: {nb}/{_NB_VECTEURS_POUR_MOYENNE}")

        if nb < _NB_VECTEURS_POUR_MOYENNE:
            print(f"   ⏳ En attente de plus de captures ({nb}/{_NB_VECTEURS_POUR_MOYENNE})")
            return jsonify({
                "status":  "accumulating",
                "message": f"Capture {nb}/{_NB_VECTEURS_POUR_MOYENNE} reçue, continuez à regarder la caméra",
                "nb":      nb
            })

        # Calculer la moyenne normalisée (L2)
        tous_vecteurs = np.array(_vecteurs_en_attente[patient_id], dtype=np.float64)
        vecteur_moyen = np.mean(tous_vecteurs, axis=0)
        norme = np.linalg.norm(vecteur_moyen)
        if norme > 0:
            vecteur_moyen = vecteur_moyen / norme

        print(f"   📊 Moyenne calculée sur {nb} vecteurs (normalisée)")
        print(f"   Premieres valeurs moyenne: {vecteur_moyen[:3].tolist()}")

        del _vecteurs_en_attente[patient_id]

        vecteur_bytes = vecteur_moyen.astype(np.float64).tobytes()
        print(f"   Taille en bytes: {len(vecteur_bytes)}")

        if len(vecteur_bytes) == 0:
            return jsonify({"status": "error", "message": "vecteur vide"}), 400

        url = f"{URL_JAVA_BIOMETRIE}/{patient_id}/biometrie"
        print(f"   URL: {url}")

        response = requests.put(
            url, data=vecteur_bytes,
            headers={"Content-Type": "application/octet-stream"}, timeout=10
        )
        print(f"   Reponse Java: HTTP {response.status_code}")

        if response.status_code == 200:
            _biometrie_enregistree.add(patient_id)
            print(f"✅ Biométrie enregistrée (moyenne {nb} captures) pour patient {patient_id}")
            return jsonify({
                "status":  "success",
                "message": f"Biométrie enregistrée (moyenne de {nb} captures)"
            })
        else:
            print(f"❌ Erreur MS1 → HTTP {response.status_code}: {response.text[:200]}")
            return jsonify({"status": "error", "message": f"MS1 HTTP {response.status_code}"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/reinitialiser_biometrie', methods=['POST'])
def reinitialiser_biometrie():
    """
    Réinitialise le guard pour un patient
    (utile si on veut re-enregistrer sa biométrie).
    """
    global _biometrie_enregistree, _vecteurs_en_attente
    try:
        data = request.get_json()
        patient_id = str(data.get('patient_id', ''))
        if not patient_id:
            return jsonify({"status": "error", "message": "patient_id requis"}), 400

        _biometrie_enregistree.discard(patient_id)
        _vecteurs_en_attente.pop(patient_id, None)

        print(f"🔄 Biométrie réinitialisée pour patient {patient_id}")
        return jsonify({"status": "success", "message": f"Biométrie réinitialisée pour patient {patient_id}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================
# 12. ROUTES — TABLETTE 2
# ============================================

@app.route('/api/visage/demarrer_capture_tablette2', methods=['POST'])
def demarrer_capture_tablette2():
    global patient_en_attente_tablette2
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"status": "error", "message": "patient_id requis"}), 400
    patient_en_attente_tablette2 = patient_id
    print(f"🩺 Patient {patient_id} en attente tablette 2")
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
# 13. BLUEPRINT TABLETTE 2 (consultation)
# ============================================
from tablette2_consultation import tablette2_bp, init_tablette2

app.register_blueprint(tablette2_bp)

init_tablette2({
    "HEADERS_SPRING":                 HEADERS_SPRING,
    "consultation_en_cours":          consultation_en_cours,
    "URL_SPRING_STATUT_CONSULTATION": URL_SPRING_STATUT_CONSULTATION,
    "extraire_embedding":             extraire_embedding,
    "appeler_java_reconnaitre":       appeler_java_reconnaitre,
    "get_rdv_du_jour":                get_rdv_du_jour,
    "get_vecteur_depuis_bd":          get_vecteur_depuis_bd,
    "calculer_similarite_cosinus":    calculer_similarite_cosinus,
})

# ============================================
# 14. LANCEMENT
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("🚀 SERVEUR IA DOCKT — EN LIGNE (port 8000)")
    print("=" * 55)
    print("📡 Status              : GET  /api/status")
    print("🎙️  Transcription       : POST /api/transcribe")
    print("─" * 55)
    print("💊 Pharmacies          : GET  /api/pharmacies")
    print("🤖 Chatbot             : POST /api/chat")
    print("─" * 55)
    print("✅ Tablette 1")
    print("   Reconnaitre         : POST /api/visage/reconnaitre")
    print("   Extraire vecteur    : POST /api/visage/extraire_vecteur")
    print("   Check-in            : POST /api/visage/checkin")
    print("   Poll résultat       : GET  /api/visage/dernier_checkin")
    print("   Démarrer capture    : POST /api/visage/demarrer_capture")
    print("   Associer biométrie  : POST /api/visage/capture_et_associer")
    print("   Réinit biométrie    : POST /api/visage/reinitialiser_biometrie")
    print("─" * 55)
    print("🩺 Tablette 2")
    print("   Consultation        : POST /api/visage/consulter")
    print("   Status              : GET  /api/visage/tablette2/status")
    print("   Démarrer capture    : POST /api/visage/demarrer_capture_tablette2")
    print("=" * 55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)