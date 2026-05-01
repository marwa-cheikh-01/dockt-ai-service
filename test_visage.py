import cv2
from deepface import DeepFace
import numpy as np

print("=" * 50)
print("🔬 TEST DE RECONNAISSANCE FACIALE")
print("=" * 50)

# 1. Capture un visage depuis la caméra
print("\n📷 1. Capture de votre visage...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Impossible de capturer l'image")
    exit()

print(f"✅ Image capturée : {frame.shape}")

# 2. Sauvegarde temporaire
temp_image = "temp_visage.jpg"
cv2.imwrite(temp_image, frame)
print(f"✅ Image sauvegardée : {temp_image}")

# 3. Analyse avec DeepFace
print("\n🧠 2. Analyse du visage avec DeepFace...")
try:
    # Extraction des caractéristiques (vecteur facial)
    embedding = DeepFace.represent(
        img_path=temp_image,
        model_name="Facenet",  # Modèle de reconnaissance
        enforce_detection=False  # Continue même si visage non détecté
    )
    
    print("✅ Visage détecté et analysé !")
    print(f"   Vecteur facial : {len(embedding[0]['embedding'])} nombres")
    print(f"   Premières valeurs : {embedding[0]['embedding'][:5]}...")
    
except Exception as e:
    print(f"⚠️ Erreur : {e}")
    print("   Assurez-vous qu'un visage est bien visible")

# 4. Test de vérification (optionnel)
print("\n👤 3. Test de vérification...")
try:
    # Vérifie en comparant avec lui-même (doit retourner True)
    result = DeepFace.verify(
        img1_path=temp_image,
        img2_path=temp_image,
        model_name="Facenet",
        enforce_detection=False
    )
    print(f"✅ Vérification : {result['verified']} (similarité: {result['distance']:.3f})")
except Exception as e:
    print(f"⚠️ Vérification non effectuée : {e}")

print("\n" + "=" * 50)
print("📌 Test terminé")
print("=" * 50)