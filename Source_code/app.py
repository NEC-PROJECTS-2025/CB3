'''
import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, render_template, jsonify

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask application
app = Flask(__name__)

# ========== MODEL CONFIGURATION ==========
INPUT_SIZE = (100, 100)        # Must match training dimensions
MODEL_PATH = 'retinal_disease_resnet101_32class.h5'  # Trained model file
CLASS_COUNT = 32               # Number of disease classes

# ========== LOAD AND VERIFY MODEL ==========
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Compile model (required for some TensorFlow versions)
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# ========== DISEASE LABEL MAPPING ==========
# IMPORTANT: Must match training label order exactly!
LABEL_MAP = {
    0: 'AMN Macular Neuroretinopathy',
    1: "Adult Coats' Disease",
    2: 'Adult Foveomacular Dystrophy Pattern',
    3: 'Age-Related Macular Degeneration With Pattern Dystrophy Appearance',
    4: 'Antiphospholipid Antibody Syndrome',
    5: "Behcet's",
    6: 'Bilateral Macular Dystrophy',
    7: "Bull's Eye Maculopathy Chloroquine",
    8: 'CMV Chorioretinitis',
    9: 'Central Serous Chorioretinopathy',
    10: 'Choroidal Nevus',
    11: 'Cone - Rod Dystrophy',
    12: 'Congenital Syphillis',
    13: 'Diabetic Maculopathy Multiple Myeloma with Retinal Detachment',
    14: 'Giant Retinal Tear',
    15: 'Juxtafoveal Telangiectasis DM Diabetes',
    16: "Leber's Stellate Maculopathy",
    17: 'Macular Dystrophy',
    18: 'Multifocal Exudative Detachments Due to VKH',
    19: 'Myelinated Nerve Fibers',
    20: 'North Carolina Dystrophy',
    21: 'Optic Disc Drusen',
    22: 'Pattern Dystrophy Simulating Fundus Flavimaculatus',
    23: 'Reticular Pattern Dystrophy',
    24: 'Retinal Folds Following Retinal Reattachment Surgery',
    25: 'Retrohyaloid Hemorrhage',
    26: 'Roth Spot',
    27: 'Self-Applied Retinal Detachment',
    28: 'Solar Retinopathy Familial',
    29: "Susac's Syndrome",
    30: "Terson's Syndrome",
    31: 'Wyburn-Mason Syndrome'
}

# ========== IMAGE PROCESSING ==========
def preprocess_image(image_stream):
    """Convert uploaded file to model-ready input"""
    try:
        # Read and decode image
        img = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 
                          cv2.IMREAD_COLOR)
        
        # Convert color space and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SIZE)
        
        # Normalize and add batch dimension
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
        
    except Exception as e:
        app.logger.error(f"Image processing failed: {str(e)}")
        return None

# ========== FLASK ROUTES ==========
@app.route('/')
def home():
    """Render main page"""
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')
@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')
@app.route('/prediction')
def predection():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file submitted'}), 400

        # Preprocess image
        processed_img = preprocess_image(file)
        if processed_img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Get predictions
        predictions = model.predict(processed_img)[0]
        
        # Verify prediction dimensions
        if len(predictions) != CLASS_COUNT:
            return jsonify({'error': 'Model output mismatch'}), 500

        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        # Format response
        results = [{
            'class': LABEL_MAP[int(idx)],
            'confidence': float(predictions[idx]),
            'confidence_percent': f"{predictions[idx]*100:.2f}%"
        } for idx in top_indices]

        return jsonify({'predictions': results})

    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': 'Processing error'}), 500

# ========== START APPLICATION ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 
'''
import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, render_template, jsonify

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask application
app = Flask(__name__)

# ========== MODEL CONFIGURATION ==========
INPUT_SIZE = (100, 100)        # Must match training dimensions
MODEL_PATH = 'retinal_disease_resnet101_32class.h5'  # Trained model file
CLASS_COUNT = 32               # Number of disease classes
MIN_CONFIDENCE = 0.03          # 3% minimum confidence threshold

# ========== LOAD AND VERIFY MODEL ==========
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Compile model (required for some TensorFlow versions)
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# ========== DISEASE LABEL MAPPING ==========
LABEL_MAP = {
    0: 'AMN Macular Neuroretinopathy',
    1: "Adult Coats' Disease",
    2: 'Adult Foveomacular Dystrophy Pattern',
    3: 'Age-Related Macular Degeneration With Pattern Dystrophy Appearance',
    4: 'Antiphospholipid Antibody Syndrome',
    5: "Behcet's",
    6: 'Bilateral Macular Dystrophy',
    7: "Bull's Eye Maculopathy Chloroquine",
    8: 'CMV Chorioretinitis',
    9: 'Central Serous Chorioretinopathy',
    10: 'Choroidal Nevus',
    11: 'Cone - Rod Dystrophy',
    12: 'Congenital Syphillis',
    13: 'Diabetic Maculopathy Multiple Myeloma with Retinal Detachment',
    14: 'Giant Retinal Tear',
    15: 'Juxtafoveal Telangiectasis DM Diabetes',
    16: "Leber's Stellate Maculopathy",
    17: 'Macular Dystrophy',
    18: 'Multifocal Exudative Detachments Due to VKH',
    19: 'Myelinated Nerve Fibers',
    20: 'North Carolina Dystrophy',
    21: 'Optic Disc Drusen',
    22: 'Pattern Dystrophy Simulating Fundus Flavimaculatus',
    23: 'Reticular Pattern Dystrophy',
    24: 'Retinal Folds Following Retinal Reattachment Surgery',
    25: 'Retrohyaloid Hemorrhage',
    26: 'Roth Spot',
    27: 'Self-Applied Retinal Detachment',
    28: 'Solar Retinopathy Familial',
    29: "Susac's Syndrome",
    30: "Terson's Syndrome",
    31: 'Wyburn-Mason Syndrome'
}

# ========== IMAGE PROCESSING ==========
def preprocess_image(image_stream):
    """Convert uploaded file to model-ready input"""
    try:
        img = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 
                          cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SIZE)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        app.logger.error(f"Image processing failed: {str(e)}")
        return None

# ========== FLASK ROUTES ==========
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file submitted'}), 400

        processed_img = preprocess_image(file)
        if processed_img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        predictions = model.predict(processed_img)[0]
        
        if len(predictions) != CLASS_COUNT:
            return jsonify({'error': 'Model output mismatch'}), 500

        max_confidence = np.max(predictions)
        if max_confidence < MIN_CONFIDENCE:
            return jsonify({
                'error': 'Invalid retinal image',
                'message': 'Low confidence prediction (minimum 3% required)',
                'max_confidence': f"{max_confidence*100:.2f}%",
                'confidence_scale': round(float(max_confidence)*10, 1)
            }), 400

        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[idx])
            results.append({
                'class': LABEL_MAP[int(idx)],
                'confidence': confidence,
                'confidence_percent': f"{confidence*100:.2f}%",
                'confidence_scale_1_10': round(confidence * 10, 1)
            })

        return jsonify({
            'predictions': results,
            'status': 'Valid retinal image',
            'confidence_check': f"Minimum confidence threshold ({MIN_CONFIDENCE*100}%) passed"
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': 'Processing error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)