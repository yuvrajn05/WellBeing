import os
from flask import Flask, render_template, request, redirect, url_for
from retinaface import RetinaFace
from deepface import DeepFace
import uuid
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALIGNED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALIGNED_FOLDER'] = ALIGNED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ALIGNED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = str(uuid.uuid4()) + ".jpg"
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(img_path)

            try:
                faces_aligned = RetinaFace.extract_faces(img_path=img_path, align=True)

                if not faces_aligned:
                    return render_template('result.html', error="No face detected.")

                # Save aligned face to static/processed
                aligned_filename = "aligned_" + filename
                aligned_path = os.path.join(app.config['ALIGNED_FOLDER'], aligned_filename)

                # Ensure the array is in uint8 and RGB format
                face_array = faces_aligned[0]
                if face_array.dtype != np.uint8:
                    # If values are in [0, 1], scale to [0, 255]
                    if face_array.max() <= 1.0:
                        face_array = (face_array * 255).astype(np.uint8)
                    else:
                        face_array = face_array.astype(np.uint8)

                # Save the aligned face
                aligned_image_pil = Image.fromarray(face_array)
                aligned_image_pil.save(aligned_path)

                # Analyze emotion
                result = DeepFace.analyze(face_array, actions=['emotion'])
                emotion = result[0]['dominant_emotion']

                # Convert and sort scores by value descending
                emotion_scores = {k: float(v) for k, v in result[0]['emotion'].items()}
                emotion_scores = dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True))



                return render_template('result.html',
                                       emotion=emotion,
                                       emotion_scores=emotion_scores,
                                       image_filename=os.path.basename(img_path).replace("\\", "/"),
                                       aligned_filename=os.path.basename(aligned_path).replace("\\", "/"))

            except Exception as e:
                return render_template('result.html', error=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
