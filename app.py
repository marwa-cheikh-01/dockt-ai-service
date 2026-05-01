import os
import uuid
import time
import requests as http_requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import whisper

app = Flask(__name__)
# On autorise Angular (qui est sur le port 4200) à communiquer avec Python (sur le port 8000)
CORS(app)

# 1. Chargement du modèle d'IA Whisper au démarrage
print("⏳ Chargement du modèle d'IA Whisper en cours... (Cela peut prendre quelques instants)")
# Utilisation du modèle 'tiny' pour une analyse très rapide sur PC portable
model = whisper.load_model("tiny")
print("✅ Modèle chargé avec succès ! L'IA est prête à écouter.")

# 2. Route de test pour vérifier que le serveur tourne bien
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "message": "Le microservice IA DOCKT est en ligne !", 
        "status": "success"
    })


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    unique_id = str(uuid.uuid4())
    temp_path = f"audio_{unique_id}.webm"
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Aucun fichier audio n'a été envoyé"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "Le nom du fichier est vide"}), 400

        # On sauvegarde l'enregistrement d'Angular sur le PC
        audio_file.save(temp_path)
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return jsonify({"error": "Le fichier audio est vide"}), 400

        # --- DÉMARRAGE DU CHRONOMÈTRE ---
        print(f"\n🎙️ Nouvelle dictée reçue ! Début de l'analyse pour {unique_id}...")
        start_time = time.time()

        # L'IA écoute et transcrit le fichier audio
        # fp16=False évite les avertissements si on n'a pas de carte graphique
        result = model.transcribe(temp_path, language="fr", fp16=False)
        texte_transcrit = result["text"]

        # --- FIN DU CHRONOMÈTRE ---
        end_time = time.time()
        duree = round(end_time - start_time, 2)
        
        # On affiche le résultat dans le terminal noir pour surveiller
        print(f"✅ Analyse terminée en {duree} secondes !")
        print(f"📝 Ordonnance dictée : {texte_transcrit.strip()}")

        # On supprime le fichier audio de l'ordinateur pour faire le ménage
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # On renvoie le texte exact à Angular pour qu'il l'affiche dans le carré blanc
        return jsonify({"texte_transcrit": texte_transcrit.strip()})

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Le serveur Python tourne sur le port 8000
    app.run(debug=True, port=8000)