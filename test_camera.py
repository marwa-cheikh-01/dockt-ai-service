import cv2

print("📷 Test de la caméra...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Caméra non détectée")
    print("   Si vous êtes sur Antigravity, autorisez la caméra dans le navigateur")
else:
    print("✅ Caméra détectée !")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Image capturée - Taille : {frame.shape}")
        print(f"   Hauteur: {frame.shape[0]} pixels")
        print(f"   Largeur: {frame.shape[1]} pixels")
    else:
        print("❌ Impossible de capturer une image")
    cap.release()

print("\n📌 Test terminé")