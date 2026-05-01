import cv2
import requests
import numpy as np
import time

print("=" * 60)
print("🏥 TABLETTE 1 - SYSTÈME DE CHECK-IN AUTOMATIQUE")
print("=" * 60)

URL_IA = "http://localhost:8000/api/visage/capture"
URL_BACKEND_RECONNAITRE = "http://localhost:8082/api/patients/reconnaitre"
URL_BACKEND_CREER = "http://localhost:8082/api/patients/avec-biometrie"

def capturer_vecteur():
    print("\n📷 Veuillez vous placer devant la caméra...")
    try:
        resp = requests.get(URL_IA, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                vecteur = data["vecteur"]
                print(f"✅ Visage capturé ! Vecteur : {len(vecteur)} nombres")
                return vecteur
        print("❌ Erreur lors de la capture")
        return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

def reconnaitre_patient(vecteur):
    try:
        vecteur_bytes = np.array(vecteur, dtype=np.float64).tobytes()
        resp = requests.post(
            URL_BACKEND_RECONNAITRE,
            data=vecteur_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10
        )
        if resp.status_code == 200:
            patient = resp.json()
            print(f"\n👤 Patient reconnu : {patient.get('nom')} {patient.get('prenom')}")
            return patient
        elif resp.status_code == 404:
            print("\n❓ Patient non reconnu (nouveau patient)")
            return None
        else:
            print(f"❌ Erreur backend : {resp.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

def creer_patient(nom, prenom, cin, vecteur):
    try:
        vecteur_bytes = np.array(vecteur, dtype=np.float64).tobytes()
        payload = {
            "nom": nom,
            "prenom": prenom,
            "cin": cin,
            "vecteur": list(vecteur_bytes)
        }
        resp = requests.post(URL_BACKEND_CREER, json=payload, timeout=10)
        if resp.status_code == 201:
            data = resp.json()
            print(f"✅ Nouveau patient créé (ID: {data.get('idPatient')})")
            return data.get('idPatient')
        else:
            print(f"❌ Échec création : {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

# Programme principal
if __name__ == "__main__":
    print("\n🎯 Mode : Check-in automatique + enregistrement nouveau patient")
    print("=" * 60)

    while True:
        input("\n🔘 Appuyez sur ENTER pour scanner un visage (ou 'q' pour quitter)...")
        if input().strip().lower() == 'q':
            print("👋 Au revoir.")
            break

        vecteur = capturer_vecteur()
        if not vecteur:
            continue

        patient = reconnaitre_patient(vecteur)
        if patient:
            print(f"\n✅ CHECK-IN AUTOMATIQUE pour {patient['nom']} {patient['prenom']}")
        else:
            print("\n📝 NOUVEAU PATIENT DÉTECTÉ")
            nom = input("   Nom : ").strip()
            prenom = input("   Prénom : ").strip()
            cin = input("   CIN : ").strip()
            if nom and prenom and cin:
                pid = creer_patient(nom, prenom, cin, vecteur)
                if pid:
                    # Re-connaître le patient (vérification)
                    print("🔍 Vérification de l'enregistrement...")
                    patient2 = reconnaitre_patient(vecteur)
                    if patient2:
                        print(f"✅ CHECK-IN effectué pour {patient2['nom']} {patient2['prenom']}")
                    else:
                        print("⚠️ Patient enregistré mais la reconnaissance immédiate a échoué.")
            else:
                print("❌ Informations manquantes, patient non enregistré.")

    print("\n📌 Application terminée")