# ================================================================
# tablette2_consultation.py — Tablette 2 DOCKT
#
# Logique métier complète :
#
# DÉBUT CONSULTATION (statutConsultation = EN_ATTENTE) :
#   [1] Extraire vecteur Facenet depuis la caméra
#   [2] Reconnaître patient via MS1 (comparaison BDD)
#   [3] Vérifier check_in = true dans MS2
#   [4] Vérification biométrique : cosinus(caméra, BDD) >= 0.60  ← FIX seuil abaissé
#   [5] Vérifier que c'est son tour (prochain patient file)
#   [6] MS2 → EN_CONSULTATION + heureEffective = now()
#   → Afficher "Veuillez entrer dans la salle du médecin"
#
# FIN CONSULTATION (statutConsultation = EN_CONSULTATION) :
#   [1] Extraire vecteur Facenet depuis la caméra
#   [2] Reconnaître patient via MS1
#   [3] Vérifier check_in = true dans MS2
#   [4] Vérification biométrique : cosinus(caméra, BDD) >= 0.60  ← FIX seuil abaissé
#   [5] Vérifier que ce patient a statut EN_CONSULTATION
#   [6] MS2 → TERMINE + heureFin = now()
#   → Afficher "Bonne journée !"
#
# Si visage non reconnu / mauvais patient → "Patient non reconnu"
#
# Endpoint unique : POST /api/visage/consulter
# Monitoring      : GET  /api/visage/tablette2/status
# ================================================================

import os
import time
import requests
from flask import Blueprint, request, jsonify

tablette2_bp = Blueprint('tablette2', __name__)

_deps = {}

def init_tablette2(deps: dict):
    _deps.update(deps)

def _get(key):
    if key not in _deps:
        raise RuntimeError(
            f"[tablette2] clé manquante : '{key}'. "
            f"Vérifiez init_tablette2() dans app.py."
        )
    return _deps[key]


# ================================================================
# CONSTANTES
# FIX 2 : seuil abaissé de 0.70 → 0.60 pour conditions réelles
# (lumière variable, angle de caméra, distance)
# ================================================================

SEUIL_SIMILARITE     = 0.60   # ← FIX : était 0.70
URL_MS2_RDV_PATIENT  = "http://localhost:8081/api/rdv/patient"
URL_MS2_PROCHAIN     = "http://localhost:8081/api/file-attente/prochain-patient"


# ================================================================
# FONCTIONS UTILITAIRES INTERNES
# ================================================================

def changer_statut_consultation(rdv_id, nouveau_statut):
    """
    PUT /api/file-attente/statut-consultation/{rdv_id}?statutConsultation={statut}
    EN_CONSULTATION → MS2 enregistre heureEffective = now()
    TERMINE         → MS2 enregistre heureFin = now()
    """
    try:
        url = (
            f"{_get('URL_SPRING_STATUT_CONSULTATION')}"
            f"/{rdv_id}?statutConsultation={nouveau_statut}"
        )
        print(f"\n📡 [changer_statut] PUT {url}")
        response = requests.put(
            url, headers=_get('HEADERS_SPRING'), timeout=10
        )
        print(f"📡 [changer_statut] HTTP {response.status_code}")

        if response.status_code == 200:
            print(f"✅ MS2 → statut = {nouveau_statut} (RDV {rdv_id})")
            return True
        elif response.status_code == 404:
            print(f"❌ RDV {rdv_id} introuvable dans MS2")
        else:
            print(f"❌ Erreur {response.status_code} : {response.text[:200]}")
        return False

    except requests.exceptions.ConnectionError:
        print("❌ MS2 Spring Boot (port 8081) inaccessible !")
        return False
    except Exception as e:
        print(f"❌ Exception changer_statut: {e}")
        return False


def verifier_checkin_et_statut(patient_id):
    """
    Appelle MS2 GET /api/rdv/patient/{id}.
    Cherche le RDV d'aujourd'hui.
    Retourne (rdv_id, statut_consultation) si check_in = true.
    Retourne (None, None) si check_in = false ou pas de RDV.
    """
    try:
        url = f"{URL_MS2_RDV_PATIENT}/{patient_id}"
        print(f"\n📡 [verifier_checkin] GET {url}")
        response = requests.get(
            url, headers=_get('HEADERS_SPRING'), timeout=10
        )
        if response.status_code != 200:
            print(f"   ❌ HTTP {response.status_code}")
            return None, None

        rdvs        = response.json()
        aujourd_hui = time.strftime("%Y-%m-%d")

        for rdv in rdvs:
            date_rdv = rdv.get('datePrevue', '')
            if isinstance(date_rdv, str):
                date_rdv = date_rdv.split('T')[0]
            if str(date_rdv) == aujourd_hui:
                rdv_id   = rdv.get('id')
                check_in = rdv.get('checkIn', False)
                statut   = rdv.get('statutConsultation', '')
                print(f"   RDV {rdv_id} | checkIn={check_in} | statut={statut}")
                if check_in:
                    print(f"   ✅ check_in = true")
                    return rdv_id, statut
                else:
                    print(f"   ❌ check_in = false")
                    return None, None

        print(f"   ⚠️ Aucun RDV aujourd'hui pour patient {patient_id}")
        return None, None

    except Exception as e:
        print(f"❌ [verifier_checkin] Exception: {e}")
        return None, None


def verifier_identite_biometrique(vecteur_camera, patient_id):
    """
    Compare le vecteur caméra avec le vecteur stocké en BDD.
    Retourne (True, score) si similarité >= SEUIL_SIMILARITE (0.60).
    Retourne (False, score) sinon.
    Si pas de vecteur BDD → (True, None) — on laisse passer.

    FIX 2 : seuil abaissé à 0.60 pour tenir compte des conditions réelles.
    FIX 4 : log détaillé du score pour faciliter le diagnostic.
    """
    print(f"\n🔐 [verif_bio] Vérification biométrique patient {patient_id}...")
    print(f"   🎯 Seuil de reconnaissance : {SEUIL_SIMILARITE}")

    vecteur_bd = _get('get_vecteur_depuis_bd')(patient_id)

    if vecteur_bd is None or len(vecteur_bd) == 0:
        print(f"   ⚠️ Pas de vecteur BDD → vérification ignorée (patient autorisé)")
        return True, None

    similarite = _get('calculer_similarite_cosinus')(vecteur_camera, vecteur_bd)

    # FIX 4 : log détaillé avec indication de résultat
    if similarite >= SEUIL_SIMILARITE:
        print(f"   ✅ IDENTITÉ CONFIRMÉE — score={similarite:.4f} >= seuil={SEUIL_SIMILARITE}")
        return True, similarite
    elif similarite >= 0.50:
        # Zone grise — log pour diagnostic
        print(f"   ⚠️ Score PROCHE du seuil : {similarite:.4f} (seuil={SEUIL_SIMILARITE})")
        print(f"   ❌ IDENTITÉ REJETÉE — différence : {SEUIL_SIMILARITE - similarite:.4f}")
        return False, similarite
    else:
        print(f"   ❌ IDENTITÉ REJETÉE — score={similarite:.4f} << seuil={SEUIL_SIMILARITE}")
        return False, similarite


def get_prochain_patient_attendu():
    """
    GET /api/file-attente/prochain-patient
    Retourne dict { idPatient, idRdv, position } ou None.
    """
    try:
        print(f"\n📡 [prochain_patient] GET {URL_MS2_PROCHAIN}")
        response = requests.get(
            URL_MS2_PROCHAIN,
            headers=_get('HEADERS_SPRING'),
            timeout=10
        )
        print(f"📡 [prochain_patient] HTTP {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data and data.get('idPatient') is not None:
                print(f"   ✅ Prochain patient : ID {data.get('idPatient')}")
                return data
            print("   ⚠️ Aucun patient en attente")
            return None
        return None

    except Exception as e:
        print(f"❌ [prochain_patient] Exception: {e}")
        return None


# ================================================================
# ENDPOINT UNIQUE : POST /api/visage/consulter
#
# Gère automatiquement début ET fin selon statutConsultation du RDV.
# Angular n'a pas besoin de gérer le mode — c'est automatique.
# ================================================================

@tablette2_bp.route('/api/visage/consulter', methods=['POST'])
def consulter():
    tmp_path = None
    try:
        print("\n" + "=" * 65)
        print("🩺  TABLETTE 2 — CONSULTATION")
        print("=" * 65)

        # ── [1] Extraction du vecteur ───────────────────────────
        print("\n[1/6] Extraction vecteur Facenet...")

        if 'image' not in request.files:
            return jsonify({
                "status":  "error",
                "message": "Aucune image reçue."
            }), 400

        vecteur_camera, tmp_path = _get('extraire_embedding')(
            request.files['image']
        )

        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            tmp_path = None

        if vecteur_camera is None:
            print("   ⚠️ Aucun visage détecté")
            return jsonify({
                "status":  "no_face",
                "message": "Aucun visage détecté. Regardez bien la caméra."
            }), 200

        print(f"   ✅ Vecteur extrait : {len(vecteur_camera)} dimensions")

        # ── [2] Reconnaissance patient via MS1 ──────────────────
        print("\n[2/6] Reconnaissance patient via MS1...")
        patient = _get('appeler_java_reconnaitre')(vecteur_camera)

        if not patient:
            print("   ❌ Patient non reconnu par MS1")
            print("   💡 Vérifiez les logs Java pour voir les scores de similarité")
            return jsonify({
                "status":  "inconnu",
                "message": "Patient non reconnu. Contactez la secrétaire."
            }), 200

        patient_id = str(patient.get('idPatient') or patient.get('id'))
        nom        = patient.get('nom', '')
        prenom     = patient.get('prenom', '')
        print(f"   ✅ Patient reconnu : {prenom} {nom} (ID: {patient_id})")

        # ── [3] Vérifier check_in = true et récupérer statut ────
        print(f"\n[3/6] Vérification check_in et statut consultation...")
        rdv_id, statut_actuel = verifier_checkin_et_statut(patient_id)

        if rdv_id is None:
            print(f"   ❌ check_in=false ou pas de RDV → patient non autorisé")
            return jsonify({
                "status":     "patient_non_reconnu",
                "message":    "Patient non reconnu.",
                "patient_id": patient_id,
                "nom":        nom,
                "prenom":     prenom
            }), 200

        print(f"   ✅ RDV={rdv_id} | statut={statut_actuel}")

        # ── [4] Vérification biométrique double ─────────────────
        print(f"\n[4/6] Vérification biométrique (cosinus caméra vs BDD)...")
        identite_ok, similarite = verifier_identite_biometrique(
            vecteur_camera, patient_id
        )

        if not identite_ok:
            score = f"{similarite:.4f}" if similarite is not None else "N/A"
            print(f"   ❌ Identité rejetée pour {prenom} {nom} (score={score}, seuil={SEUIL_SIMILARITE})")
            return jsonify({
                "status":     "patient_non_reconnu",
                "message":    "Patient non reconnu.",
                "patient_id": patient_id,
                "nom":        nom,
                "prenom":     prenom
            }), 200

        score_str = f"{similarite:.4f}" if similarite is not None else "ignorée (pas de BDD)"
        print(f"   ✅ Identité confirmée (score={score_str})")

        # ── [5/6] Logique selon statut actuel ───────────────────

        # ══════════ CAS 1 : DÉBUT DE CONSULTATION ══════════
        if statut_actuel in (None, '', 'EN_ATTENTE'):
            print(f"\n[5/6] Début de consultation — vérification du tour...")

            prochain = get_prochain_patient_attendu()

            if prochain is None:
                return jsonify({
                    "status":     "patient_non_reconnu",
                    "message":    "Patient non reconnu.",
                    "patient_id": patient_id,
                    "nom":        nom,
                    "prenom":     prenom
                }), 200

            prochain_id = str(prochain.get('idPatient', ''))

            if prochain_id != patient_id:
                print(
                    f"   ⚠️ Mauvais patient !"
                    f" Attendu ID={prochain_id} / Présenté ID={patient_id}"
                )
                return jsonify({
                    "status":     "patient_non_reconnu",
                    "message":    "Patient non reconnu.",
                    "patient_id": patient_id,
                    "nom":        nom,
                    "prenom":     prenom
                }), 200

            print(f"   ✅ C'est bien son tour")
            print(f"\n[6/6] MS2 → EN_CONSULTATION (enregistre heureEffective)...")

            succes = changer_statut_consultation(rdv_id, "EN_CONSULTATION")

            if not succes:
                return jsonify({
                    "status":  "error",
                    "message": "Erreur lors du démarrage. Contactez la secrétaire."
                }), 500

            _get('consultation_en_cours')[patient_id] = {
                "rdv_id": rdv_id,
                "debut":  time.time(),
                "nom":    nom,
                "prenom": prenom
            }

            print(f"\n✅ DÉBUT CONSULTATION OK — {prenom} {nom} | RDV {rdv_id}")

            return jsonify({
                "status":     "success",
                "action":     "debut",
                "message":    "Veuillez entrer dans la salle du médecin.",
                "patient_id": patient_id,
                "nom":        nom,
                "prenom":     prenom,
                "rdv_id":     rdv_id,
                "timestamp":  time.strftime("%H:%M:%S")
            })

        # ══════════ CAS 2 : FIN DE CONSULTATION ══════════
        elif statut_actuel == 'EN_CONSULTATION':
            print(f"\n[5/6] Fin de consultation pour {prenom} {nom}...")

            consultation = _get('consultation_en_cours')
            duree = 0
            if patient_id in consultation:
                duree = int(time.time() - consultation[patient_id]["debut"])
                print(f"   ✅ Consultation trouvée en mémoire (durée {duree}s)")
            else:
                print(f"   ⚠️ Patient absent mémoire Flask — rdv_id connu via MS2")

            print(f"\n[6/6] MS2 → TERMINE (enregistre heureFin)...")
            succes = changer_statut_consultation(rdv_id, "TERMINE")

            if not succes:
                return jsonify({
                    "status":  "error",
                    "message": "Erreur lors de la clôture. Contactez la secrétaire."
                }), 500

            _get('consultation_en_cours').pop(patient_id, None)

            print(f"\n✅ FIN CONSULTATION OK — {prenom} {nom} | RDV {rdv_id}")

            return jsonify({
                "status":         "success",
                "action":         "fin",
                "message":        "Bonne journée !",
                "patient_id":     patient_id,
                "nom":            nom,
                "prenom":         prenom,
                "rdv_id":         rdv_id,
                "duree_secondes": duree,
                "timestamp":      time.strftime("%H:%M:%S")
            })

        # ══════════ CAS 3 : STATUT INATTENDU ══════════
        else:
            print(f"   ⚠️ Statut inattendu : {statut_actuel}")
            return jsonify({
                "status":     "patient_non_reconnu",
                "message":    "Patient non reconnu.",
                "patient_id": patient_id,
                "nom":        nom,
                "prenom":     prenom
            }), 200

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"❌ ERREUR CRITIQUE consulter: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ================================================================
# GET /api/visage/tablette2/status — monitoring
# ================================================================

@tablette2_bp.route('/api/visage/tablette2/status', methods=['GET'])
def tablette2_status():
    consultation = _get('consultation_en_cours')
    detail = {}
    for pid, info in consultation.items():
        duree = int(time.time() - info["debut"])
        detail[pid] = {
            "rdv_id":         info["rdv_id"],
            "nom":            info["nom"],
            "prenom":         info["prenom"],
            "duree_secondes": duree,
            "duree_minutes":  duree // 60
        }
    return jsonify({
        "status":                "online",
        "service":               "Tablette 2 - Consultation",
        "seuil_similarite":      SEUIL_SIMILARITE,
        "consultations_actives": len(consultation),
        "patients_en_cours":     detail,
        "timestamp":             time.strftime("%H:%M:%S")
    })