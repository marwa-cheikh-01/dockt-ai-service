import requests
import time

print("=" * 60)
print("🏥 TABLETTE 1 - SERVICE DE CHECK-IN AUTOMATIQUE")
print("=" * 60)

URL_VERIFIER_CHECKIN = "http://localhost:8000/api/visage/dernier_checkin"
URL_RDV_PATIENT      = "http://localhost:8081/api/rdv/patient"
URL_CHECKIN          = "http://localhost:8081/api/file-attente/checkin"

dernier_patient_checkin = None
dernier_timestamp       = 0
DELAI_ENTRE_SCANS       = 10


def get_rdv_du_jour(patient_id):
    try:
        response = requests.get(f"{URL_RDV_PATIENT}/{patient_id}", timeout=15)
        if response.status_code == 200:
            rdvs = response.json()
            aujourd_hui = time.strftime("%Y-%m-%d")
            for rdv in rdvs:
                if rdv.get('datePrevue') == aujourd_hui:
                    return rdv.get('id')
        return None
    except Exception as e:
        print(f"❌ Erreur RDV: {e}")
        return None


def faire_checkin(rdv_id):
    try:
        response = requests.put(f"{URL_CHECKIN}/{rdv_id}", timeout=15)
        if response.status_code == 200:
            print(f"✅ CHECK-IN effectué (RDV {rdv_id})")
            return True
        print(f"⚠️ Erreur check-in HTTP {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Erreur check-in: {e}")
        return False


print("\n🎯 Service Tablette 1 en fonctionnement...")
print("   - La caméra est gérée par Angular")
print("   - Ce service coordonne les check-ins Spring Boot")
print("   - Appuyez sur CTRL+C pour arrêter")
print("=" * 60)

while True:
    try:
        # Vérifier si Angular a reconnu un patient
        try:
            response = requests.get(URL_VERIFIER_CHECKIN, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'reconnu':
                    patient_id   = data.get('patient_id')
                    prenom       = data.get('prenom', '')
                    nom          = data.get('nom', '')
                    temps_actuel = time.time()

                    if (dernier_patient_checkin != patient_id or
                            temps_actuel - dernier_timestamp > DELAI_ENTRE_SCANS):

                        dernier_patient_checkin = patient_id
                        dernier_timestamp       = temps_actuel

                        print(f"\n👤 Patient reconnu : {prenom} {nom} (ID: {patient_id})")

                        rdv_id = get_rdv_du_jour(patient_id)
                        if rdv_id:
                            faire_checkin(rdv_id)
                        else:
                            print(f"   ℹ️ Aucun RDV aujourd'hui")

        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"⚠️ Erreur polling: {e}")

        time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n👋 Service arrêté")
        break
    except Exception as e:
        print(f"❌ Erreur: {e}")
        time.sleep(5)

print("\n📌 Terminé")