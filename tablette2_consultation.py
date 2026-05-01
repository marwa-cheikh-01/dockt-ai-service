import cv2
import requests
import numpy as np
import time

print("=" * 60)
print("🩺 TABLETTE 2 - SUIVI DE CONSULTATION")
print("=" * 60)

# Configuration des URLs
URL_CAPTURE_VECTEUR = "http://localhost:8000/api/visage/capture"
URL_RECONNAITRE = "http://localhost:8082/api/patients/reconnaitre"
URL_RDV_PATIENT = "http://localhost:8081/api/rdv/patient"
URL_STATUT_CONSULTATION = "http://localhost:8081/api/file-attente/statut-consultation"

# Variables d'état
patient_en_cours = None
rdv_en_cours = None
patient_id_en_cours = None
dernier_scan_timestamp = 0
DELAI_ENTRE_SCANS = 5  # secondes entre deux scans

def capturer_vecteur():
    """Capture un visage et retourne son vecteur"""
    try:
        response = requests.get(URL_CAPTURE_VECTEUR, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return data['vecteur']
        return None
    except Exception as e:
        return None

def reconnaitre_patient(vecteur):
    """Envoie le vecteur au backend pour reconnaissance"""
    try:
        vecteur_bytes = np.array(vecteur, dtype=np.float64).tobytes()
        response = requests.post(
            URL_RECONNAITRE,
            data=vecteur_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_rdv_aujourdhui(patient_id):
    """Récupère le RDV du jour pour un patient"""
    try:
        response = requests.get(f"{URL_RDV_PATIENT}/{patient_id}", timeout=10)
        if response.status_code == 200:
            rdvs = response.json()
            aujourd_hui = time.strftime("%Y-%m-%d")
            for rdv in rdvs:
                if rdv.get('datePrevue') == aujourd_hui:
                    statut = rdv.get('statutConsultation')
                    return rdv.get('id'), statut
        return None, None
    except Exception as e:
        return None, None

def mettre_a_jour_statut(rdv_id, nouveau_statut):
    """Met à jour le statut de consultation"""
    try:
        response = requests.put(
            f"{URL_STATUT_CONSULTATION}/{rdv_id}",
            params={"statutConsultation": nouveau_statut},
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        return False

def afficher_etat():
    """Affiche l'état actuel du système"""
    print("\n" + "=" * 50)
    if patient_en_cours:
        print(f"🩺 En consultation : {patient_en_cours.get('prenom')} {patient_en_cours.get('nom')}")
    else:
        print("🟢 En attente d'un patient...")
    print("=" * 50)

# ============================================
# BOUCLE PRINCIPALE - SCAN CONTINU
# ============================================

print("\n🎯 Service Tablette 2 en fonctionnement...")
print("   - Patient arrivant → Début consultation")
print("   - Patient sortant → Fin consultation")
print("   - Appuyez sur CTRL+C pour arrêter")
print("=" * 60)

afficher_etat()

while True:
    try:
        temps_actuel = time.time()
        
        # Scanner le visage
        vecteur = capturer_vecteur()
        
        if vecteur:
            patient = reconnaitre_patient(vecteur)
            
            if patient:
                patient_id = patient.get('idPatient')
                patient_nom = f"{patient.get('prenom')} {patient.get('nom')}"
                
                # Vérifier si c'est le patient déjà en cours
                if patient_id_en_cours == patient_id:
                    # Patient déjà en consultation, ne rien faire
                    pass
                else:
                    # Nouveau patient détecté
                    rdv_id, statut_actuel = get_rdv_aujourdhui(patient_id)
                    
                    if rdv_id:
                        if statut_actuel == 'EN_ATTENTE' or statut_actuel is None:
                            print(f"\n👤 Patient détecté : {patient_nom}")
                            print(f"📅 Statut actuel : {statut_actuel}")
                            
                            if mettre_a_jour_statut(rdv_id, 'EN_CONSULTATION'):
                                patient_en_cours = patient
                                patient_id_en_cours = patient_id
                                rdv_en_cours = rdv_id
                                print(f"✅ Consultation COMMENCÉE pour {patient_nom}")
                                afficher_etat()
                            else:
                                print(f"❌ Erreur début consultation")
                        
                        elif statut_actuel == 'EN_CONSULTATION' and patient_id_en_cours == patient_id:
                            print(f"\n👤 Patient détecté : {patient_nom}")
                            print(f"📅 Statut actuel : EN_CONSULTATION")
                            
                            if mettre_a_jour_statut(rdv_id, 'TERMINE'):
                                patient_en_cours = None
                                patient_id_en_cours = None
                                rdv_en_cours = None
                                print(f"✅ Consultation TERMINÉE pour {patient_nom}")
                                afficher_etat()
                            else:
                                print(f"❌ Erreur fin consultation")
                        else:
                            print(f"\n👤 Patient détecté : {patient_nom} (statut: {statut_actuel})")
                    else:
                        print(f"\n⚠️ Patient {patient_nom} détecté mais AUCUN RDV aujourd'hui")
        
        time.sleep(2)  # Pause entre les scans
        
    except KeyboardInterrupt:
        print("\n\n👋 Service Tablette 2 arrêté")
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")
        time.sleep(5)

print("\n📌 Application terminée")