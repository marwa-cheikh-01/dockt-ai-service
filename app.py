import os
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import requests

app = Flask(__name__)
CORS(app)

# ============================================
# 1. CHARGEMENT WHISPER (modèle base - plus précis)
# ============================================
print("⏳ Chargement du modèle Whisper 'base'... (plus précis que tiny)")
model = whisper.load_model("small")  # ← small pour plus de precision voix 
print("✅ Modèle chargé !")

# ============================================
# 2. ÉTAT GLOBAL
# ============================================
patient_en_attente_de_capture = None
patient_en_attente_tablette2  = None

dernier_patient_reconnu = {
    "status": "none", "patient_id": None,
    "nom": None, "prenom": None, "timestamp": 0
}

# URLs
URL_JAVA_RECONNAITRE = "http://localhost:8082/api/patients/reconnaitre"
URL_JAVA_BIOMETRIE   = "http://localhost:8082/api/patients"
URL_JAVA_AVEC_BIOMETRIE = "http://localhost:8082/api/patients/avec-biometrie"
URL_SPRING_RDV       = "http://localhost:8081/api/rdv/patient"
URL_SPRING_CHECKIN   = "http://localhost:8081/api/file-attente/checkin"


# ============================================
# FONCTIONS UTILITAIRES
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
        print(f"\n📡 [get_rdv_du_jour] Appel: {url}")
        

        print(f"📡 [get_rdv_du_jour] Status: {response.status_code}")
        
        if response.status_code == 200:
            rdvs = response.json()
            aujourd_hui = time.strftime("%Y-%m-%d")
            print(f"📅 [get_rdv_du_jour] Aujourd'hui: {aujourd_hui}")
            print(f"📋 [get_rdv_du_jour] Nombre de RDVs: {len(rdvs)}")
            
            for rdv in rdvs:
                date_rdv = rdv.get('datePrevue')
                rdv_id = rdv.get('id')
                
                if date_rdv:
                    if isinstance(date_rdv, str):
                        date_rdv = date_rdv.split('T')[0]
                    elif hasattr(date_rdv, 'year'):
                        date_rdv = date_rdv.strftime("%Y-%m-%d")
                    
                    if str(date_rdv) == aujourd_hui:
                        print(f"   ✅ RDV DU JOUR TROUVÉ: ID={rdv_id}")
                        return rdv_id
            
            print("   ⚠️ Aucun RDV ne correspond à aujourd'hui")
        elif response.status_code == 403:
            print("   ❌ ERREUR 403: Token JWT invalide ou expiré !")
        else:
            print(f"   ❌ Erreur HTTP {response.status_code}")
        return None
        
    except Exception as e:
        print(f"❌ [get_rdv_du_jour] Exception: {e}")
        return None


def faire_checkin_spring(rdv_id):
    try:
        url = f"{URL_SPRING_CHECKIN}/{rdv_id}"
        print(f"📡 [faire_checkin_spring] Appel: PUT {url}")
        
        print(f"📡 [faire_checkin_spring] Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ CHECK-IN Spring Boot OK (RDV {rdv_id})")
            return True
        
        if response.status_code == 403:
            print("   ❌ ERREUR 403: Token JWT invalide ou expiré !")
        else:
            print(f"⚠️ Erreur check-in Spring → HTTP {response.status_code}")
        return False
        
    except Exception as e:
        print(f"❌ [faire_checkin_spring] Exception: {e}")
        return False


# ============================================
# 3. RECONNAISSANCE VOCALE
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
        result = model.transcribe(temp_path, language="fr", fp16=False)
        texte = result["text"]
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"texte_transcrit": texte.strip()})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


# ============================================
# 4. RECONNAISSANCE FACIALE
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
        return jsonify({
            "status": "success",
            "vecteur": vecteur,
            "taille": len(vecteur)
        })

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
                "status": "success",
                "vecteur": vecteur,
                "taille": len(vecteur),
                "patient_id": patient_id,
                "nom": nom,
                "prenom": prenom
            })
        else:
            return jsonify({
                "status": "inconnu",
                "message": "Patient non reconnu",
                "vecteur": vecteur
            }), 404

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/visage/checkin', methods=['POST'])
def checkin():
    global dernier_patient_reconnu
    try:
        print("\n" + "="*50)
        print("✅ CHECK-IN DEMANDÉ")
        print("="*50)
        
        data = request.get_json()
        print(f"📥 Données reçues: {data}")
        
        if not data or 'patient_id' not in data:
            print("❌ patient_id manquant")
            return jsonify({"status": "error", "message": "patient_id requis"}), 400

        patient_id = data['patient_id']
        nom    = data.get('nom', '')
        prenom = data.get('prenom', '')
        
        print(f"👤 Patient: {prenom} {nom} (ID: {patient_id}")

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
        succes = faire_checkin_spring(rdv_id)

        dernier_patient_reconnu = {
            "status": "reconnu" if succes else "checkin_failed",
            "patient_id": patient_id,
            "nom": nom,
            "prenom": prenom,
            "rdv_id": rdv_id,
            "timestamp": time.time()
        }

        if succes:
            print("✅ CHECK-IN RÉUSSI")
            return jsonify({
                "status": "success",
                "message": "Check-in effectué !",
                "patient_id": patient_id,
                "rdv_id": rdv_id
            })
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
            tmp1.write(bytes(data['image1']))
            path1 = tmp1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
            tmp2.write(bytes(data['image2']))
            path2 = tmp2.name
        result = DeepFace.verify(img1_path=path1, img2_path=path2,
                                  model_name="Facenet", enforce_detection=False)
        for p in [path1, path2]:
            if p and os.path.exists(p): os.unlink(p)
        return jsonify({"status": "success", "verifie": result['verified'],
                        "distance": result['distance']})
    except Exception as e:
        for p in [path1, path2]:
            if p and os.path.exists(p): os.unlink(p)
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# 5. ENDPOINTS TABLETTE 1
# ============================================

@app.route('/api/visage/demarrer_capture', methods=['POST'])
def demarrer_capture():
    global patient_en_attente_de_capture
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"status": "error", "message": "patient_id requis"}), 400
    patient_en_attente_de_capture = patient_id
    print(f"📱 Nouveau patient {patient_id} en attente de capture")
    return jsonify({"status": "waiting", "patient_id": patient_id,
                    "message": "Veuillez vous présenter devant la caméra"})


@app.route('/api/visage/verifier_attente', methods=['GET'])
def verifier_attente():
    global patient_en_attente_de_capture
    if patient_en_attente_de_capture:
        return jsonify({"status": "pending",
                        "patient_id": patient_en_attente_de_capture})
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
        
        print(f"\n{'='*50}")
        print(f"📊 CAPTURE ET ASSOCIER - NOUVEAU PATIENT")
        print(f"{'='*50}")
        print(f"👤 Patient ID reçu: {patient_id}")
        print(f"📐 Vecteur reçu: {type(vecteur)}")
        
        if vecteur is None:
            print("❌ ERREUR: vecteur est NULL!")
            return jsonify({"status": "error", "message": "vecteur null"}), 400
        
        if isinstance(vecteur, list):
            print(f"   Dimensions: {len(vecteur)}")
            print(f"   Premieres valeurs: {vecteur[:3]}")
        elif isinstance(vecteur, np.ndarray):
            print(f"   Type numpy: {vecteur.dtype}")
            print(f"   Dimensions: {vecteur.shape}")
        
        if not patient_id or not vecteur:
            return jsonify({"status": "error", "message": "patient_id et vecteur requis"}), 400
        
        # Conversion en bytes
        if isinstance(vecteur, np.ndarray):
            vecteur = vecteur.tolist()
        
        vecteur_bytes = np.array(vecteur, dtype=np.float64).tobytes()
        print(f"   Taille en bytes: {len(vecteur_bytes)}")
        
        if len(vecteur_bytes) == 0:
            print("❌ ERREUR: vecteur_bytes est vide!")
            return jsonify({"status": "error", "message": "vecteur vide"}), 400
        
        url = f"{URL_JAVA_BIOMETRIE}/{patient_id}/biometrie"
        print(f"   URL: {url}")
        
        response = requests.put(
            url,
            data=vecteur_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10
        )
        
        print(f"   Reponse Java: HTTP {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Biométrie enregistrée pour patient {patient_id}")
            return jsonify({"status": "success", "message": "Biométrie enregistrée"})
        else:
            print(f"⚠️ Erreur: {response.text[:200]}")
            return jsonify({"status": "error", "message": f"HTTP {response.status_code}"}), 500
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# 6. ENDPOINTS TABLETTE 2
# ============================================

@app.route('/api/visage/demarrer_capture_tablette2', methods=['POST'])
def demarrer_capture_tablette2():
    global patient_en_attente_tablette2
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"status": "error", "message": "patient_id requis"}), 400
    patient_en_attente_tablette2 = patient_id
    print(f"🩺 Patient {patient_id} en attente Tablette 2")
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
# 7. LANCEMENT
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 SERVEUR IA DOCKT - EN LIGNE")
    print("=" * 50)
    print("👤 Reconnaitre    : /api/visage/reconnaitre")
    print("📐 Extraire       : /api/visage/extraire_vecteur")
    print("✅ Check-in       : /api/visage/checkin")
    print("📡 Poll           : /api/visage/dernier_checkin")
    print("🆕 Tablette 1     : /api/visage/demarrer_capture")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8000)