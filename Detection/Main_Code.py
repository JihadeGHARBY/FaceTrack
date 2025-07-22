"""
Main_Code.py :: Détection faciale en temps réel & message audio personnalisé
============================================================================
Ce script autonome ouvre votre webcam, détecte les visages en temps réel,
les compare à une petite base locale d’images de référence et, lorsqu’une
correspondance est suffisamment fiable, salue la personne reconnue par son
nom via les haut-parleurs.

Fonctionnement (vue d’ensemble) :
  1.   Charge deux modèles *dlib* pré-entraînés :
         • `shape_predictor_68_face_landmarks.dat` – repérage des points caractéristiques du visage.
         • `dlib_face_recognition_resnet_model_v1.dat` – embeddings faciaux en 128 dimensions.
  2.   Parcourt chaque image trouvée dans `../IMAGES_FOLDER/` (chemin relatif à ce fichier)
       et construit une **base vectorielle en mémoire** des encodings de visages connus ainsi que
       leurs étiquettes (nom de fichier sans extension).
  3.   Si un fichier audio `<name>.mp3` est absent dans `../AUDIO_FOLDER/`, le script
       en génère automatiquement un via Google-TTS afin que chaque personne dispose de son
       message vocal de bienvenue.
  4.   Lance la capture de trames depuis une source vidéo (0 = webcam par défaut),
       détecte les visages, calcule leurs encodings et cherche la correspondance la plus proche dans
       la base. Si la meilleure distance est < 0.6, le visage est considéré comme reconnu.
  5.   Trace un cadre et le score de similarité sur la trame **et** joue
       le fichier audio correspondant. Un court délai (cooldown) empêche le même message
       de se répéter en boucle tant que le visage reste à l’écran.

Structure des dossiers (attendue) :
  votre_projet/
  ├─ Detection/
  │  ├─ Main_Code.py               # <— vous êtes ici
  │  └─ models/
  │     ├─ shape_predictor_68_face_landmarks.dat
  │     └─ dlib_face_recognition_resnet_model_v1.dat
  ├─ IMAGES_FOLDER/                # Visages de référence (ex. alice.jpg)
  └─ AUDIO_FOLDER/                 # Salutations pré-enregistrées (optionnel)

Démarrage rapide
-----------
1.  Installez les dépendances (de préférence dans un virtualenv) :
        pip install opencv-python dlib pygame numpy gTTS
    Sous Windows, vous aurez peut-être besoin des outils de build Visual C++ pour *dlib*.
2.  Placez au moins une image de visage dans `../IMAGES_FOLDER/`, par ex. `bob.jpg`.
3.  Lancez le script :
        python Main_Code.py
4.  Appuyez sur **q** dans la fenêtre vidéo pour quitter.

Ajustements
-----------
•   Modifiez `VIDEO_SOURCE` avec le chemin d’un fichier vidéo ou une URL de flux pour traiter autre chose que la webcam.
•   Ajustez `COOLDOWN_SECONDS` et le seuil de distance 0.6 pour rendre le système plus ou moins strict.

Author: <your-name-here>
"""

# Installez les dépendances si nécessaire :
# pip install opencv-python dlib pygame numpy pandas gTTS

import cv2
import dlib
import numpy as np
import time
import threading
import os
import pygame


# --- CONFIGURATION ---
VIDEO_SOURCE = 0  # 0 for webcam

# Détermine la racine du projet (un dossier au-dessus de ce script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Chemins vers les dossiers images et audio (personnalisables)
IMAGES_FOLDER = os.path.join(PROJECT_ROOT, 'IMAGES_FOLDER')  # Dossier où sont stockées les images
COOLDOWN_SECONDS = 5
AUDIO_FOLDER = os.path.join(PROJECT_ROOT, 'AUDIO_FOLDER')

# --- INITIALISATION ---
print("Loading known people...")

if not os.path.exists(IMAGES_FOLDER):
    raise FileNotFoundError(f"Images folder not found at {IMAGES_FOLDER}")

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Initialize dlib's face detector and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

# --- CHARGEMENT DES PERSONNES À PARTIR DES IMAGES ---
# On construit la base des visages connus en analysant chaque fichier image dans
# `IMAGES_FOLDER`. Le nom de fichier (sans extension) est utilisé comme
# nom de la personne et sert également à rechercher (ou générer) le
# fichier `.mp3` correspondant dans `AUDIO_FOLDER`.

known_encodings = []  # Liste[np.ndarray] de descripteurs faciaux 128-D
known_names = []      # Liste parallèle des noms associés à chaque encodage

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

for file_name in os.listdir(IMAGES_FOLDER):
    # Ignorer les fichiers qui ne sont pas des images
    if not file_name.lower().endswith(VALID_EXTS):
        continue

    name = os.path.splitext(file_name)[0]            # label
    image_path = os.path.join(IMAGES_FOLDER, file_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector(rgb_img, 1)
    if len(detections) == 0:
        print(f"Warning: No face found in image {image_path}. Skipping.")
        continue

    shape = shape_predictor(rgb_img, detections[0])
    descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_img, shape))

    known_encodings.append(descriptor)
    known_names.append(name)

    # Vérifie qu’un fichier audio existe pour cette personne
    audio_path = os.path.join(AUDIO_FOLDER, f"{name}.mp3")
    if not os.path.exists(audio_path):
        print(f"Warning: Greeting audio for {name} not found at {audio_path}. This person will be recognised silently.")

print(f"Loaded {len(known_names)} known people.")

# Initialise le mixer pygame pour l’audio
pygame.mixer.init()

cooldown = False

# ---------------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------------

# Fonction pour jouer le son de manière asynchrone
def play_audio_alert(nickname):
    """Lit la salutation `<nickname>.mp3` sur un thread en arrière-plan."""
    try:
        audio_path = os.path.join(AUDIO_FOLDER, f"{nickname}.mp3")
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play audio for {nickname}: {e}")

# Gestion du cooldown
def start_cooldown():
    """Empêche d’annoncer plusieurs fois le même nom pendant quelques secondes."""
    global cooldown
    cooldown = True
    time.sleep(COOLDOWN_SECONDS)
    cooldown = False

# Démarre la capture vidéo
print("Démarrage du flux vidéo …")  # Informe l’utilisateur que la capture webcam va commencer
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

try:
    # Boucle de traitement infinie – capture une image, détecte les visages et salue les correspondances
    while True:
        # Capture une image vidéo ; `ret` est False si l’acquisition échoue (ex. caméra déconnectée)
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Convertit BGR (par défaut OpenCV) en RGB pour compatibilité dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Détecte les visages dans la trame (suréchantillonne une fois pour plus de précision)
        detections = face_detector(rgb_frame, 1)

        # Parcourt chaque visage détecté dans cette trame
        for det in detections:
            shape = shape_predictor(rgb_frame, det)
            # Calcule un embedding 128-D (descripteur facial) pour le visage détecté
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

            # Calcule la distance euclidienne entre ce visage et tous les visages connus dans la base
            distances = [np.linalg.norm(face_descriptor - known_encoding) for known_encoding in known_encodings]
            if len(distances) == 0:
                continue

            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]  # Plus petite distance → meilleure correspondance

            # N’accepte la correspondance que si la distance est inférieure au seuil empirique de 0,6
            if min_distance < 0.6:
                detected_name = known_names[min_distance_idx]

                # Dessine le rectangle et le nom
                # Dessine un cadre vert autour du visage reconnu
                cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)
                # Affiche le nom de la personne et le score de similarité au-dessus du cadre
                cv2.putText(frame, f"{detected_name} ({1 - min_distance:.2f})", (det.left(), det.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Joue le son si le cooldown est terminé
                if not cooldown:
                    threading.Thread(target=play_audio_alert, args=(detected_name,)).start()
                    threading.Thread(target=start_cooldown).start()

        # Affiche la trame résultante
        # Affiche la trame annotée dans une fenêtre intitulée "Video"
        cv2.imshow('Video', frame)

        # Appuyez sur 'q' pour quitter
        # Quitte proprement lorsque l’utilisateur appuie sur la touche q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Libération des ressources
    video_capture.release()  # Libère le périphérique webcam
    cv2.destroyAllWindows()
    print("Video stream stopped.")