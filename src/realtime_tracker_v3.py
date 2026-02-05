""" 
Live Demonstration UI for MPIIGaze Project 
Shows real-time predictions and visualizations 
""" 
import sys
import os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src')) 
import flask 
from flask import Flask, render_template_string, jsonify, request 
import numpy as np 
import json 
from pathlib import Path 
import tensorflow as tf 
import pickle 
from src.config import * 
from src.feature_engineering import FeatureExtractor 
app = Flask(__name__) 
# Global variables to store models and data 
MODELS = {} 
TEST_DATA = None 
CURRENT_SEQUENCE_IDX = 0 
TEST_SEQUENCES = [] 
# ============================================================================ 
# INITIALIZATION 
# ============================================================================ 
def load_models_and_data(): 
    """Load trained models and test data""" 
    global MODELS, TEST_DATA, TEST_SEQUENCES, CURRENT_SEQUENCE_IDX 
    print("Loading models...") 
    try: 
        # Load models 
        MODELS['attention'] = tf.keras.models.load_model( 
            str(MODELS_DIR / 'attention_classifier.h5') 
        ) 
        MODELS['gaze'] = tf.keras.models.load_model( 
            str(MODELS_DIR / 'gaze_estimator.h5') 
        ) 
        print("✓ Models loaded") 
    except Exception as e: 
        print(f"⚠ Could not load trained models: {e}") 
        print("  Using dummy models for demo...") 
        return False 
    try: 
        # Load test data 
        with open(PROCESSED_DATA_ROOT / "sequences.pkl", 'rb') as f: 
            sequences = pickle.load(f) 
        # Use last 20 sequences as "test" data for demo 
        TEST_SEQUENCES = sequences[-20:] if len(sequences) > 20 else sequences 
        CURRENT_SEQUENCE_IDX = 0 
        print(f"✓ Loaded {len(TEST_SEQUENCES)} test sequences") 
        return True 
    except Exception as e: 
        print(f"⚠ Could not load test data: {e}") 
        return False 
# ============================================================================ 
# PREDICTION FUNCTIONS 
# ============================================================================ 
def predict_attention(features): 
    """Predict attention level from features""" 
    try: 
        # Add batch dimension 
        features = np.expand_dims(features, axis=0) 
        # Predict 
        probs = MODELS['attention'].predict(features, verbose=0)[0] 
        pred_class = np.argmax(probs) 
        labels = ['Focused', 'Distracted', 'Sleeping'] 
        confidence = float(probs[pred_class]) 
        return { 
            'class': labels[pred_class], 
            'class_id': int(pred_class), 
            'confidence': confidence, 
            'all_probs': { 
                'focused': float(probs[0]), 
                'distracted': float(probs[1]), 
                'sleeping': float(probs[2]) 
            } 
        } 
    except Exception as e: 
        print(f"Error in attention prediction: {e}") 
        return { 
            'class': 'Unknown', 
            'class_id': -1, 
            'confidence': 0.0, 
            'all_probs': { 
                'focused': 0.33, 
                'distracted': 0.33, 
                'sleeping': 0.34 
            } 
        } 
def predict_gaze(features): 
    """Predict gaze coordinates from features""" 
    try: 
        # Add batch dimension 
        features = np.expand_dims(features, axis=0) 
        # Predict 
        gaze_coords = MODELS['gaze'].predict(features, verbose=0)[0] 
        return { 
            'x': float(np.clip(gaze_coords[0], 0, 1)), 
            'y': float(np.clip(gaze_coords[1], 0, 1)), 
            'x_pixel': int(gaze_coords[0] * SCREEN_WIDTH_PIXEL), 
            'y_pixel': int(gaze_coords[1] * SCREEN_HEIGHT_PIXEL) 
        } 
    except Exception as e: 
        print(f"Error in gaze prediction: {e}") 
        return { 
            'x': 0.5, 
            'y': 0.5, 
            'x_pixel': SCREEN_WIDTH_PIXEL // 2, 
            'y_pixel': SCREEN_HEIGHT_PIXEL // 2 
        } 
# ============================================================================ 
# FLASK ROUTES 
# ============================================================================ 
HTML_TEMPLATE = """ 
<!DOCTYPE html> 
<html lang="en"> 
<head> 
    <meta charset="UTF-8"> 
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
    <title>MPIIGaze Live Demonstration</title> 
    <style> 
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        } 
         
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
            padding: 20px; 
        } 
         
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
        } 
         
        .header { 
            text-align: center; 
            color: white; 
            margin-bottom: 30px; 
        } 
         
        .header h1 { 
            font-size: 2.5em; 
            margin-bottom: 10px; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3); 
        } 
         
        .header p { 
            font-size: 1.1em; 
            opacity: 0.9; 
        } 
         
        .main-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin-bottom: 20px; 
        } 
         
        .card { 
            background: white; 
            border-radius: 15px; 
            padding: 25px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            transition: transform 0.3s ease; 
        } 
         
        .card:hover { 
            transform: translateY(-5px); 
        } 
         
        .card h2 { 
            color: #667eea; 
            margin-bottom: 20px; 
            font-size: 1.5em; 
            border-bottom: 2px solid #667eea; 
            padding-bottom: 10px; 
        } 
         
        .metric { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 15px 0; 
            border-bottom: 1px solid #eee; 
        } 
         
        .metric:last-child { 
            border-bottom: none; 
        } 
         
        .metric-label { 
            font-weight: 600; 
            color: #333; 
        } 
         
        .metric-value { 
            font-size: 1.3em; 
            font-weight: bold; 
            color: #667eea; 
        } 
         
        .confidence-bar { 
            width: 100%; 
            height: 8px; 
            background: #eee; 
            border-radius: 4px; 
            overflow: hidden; 
            margin-top: 5px; 
        } 
         
        .confidence-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #667eea, #764ba2); 
            transition: width 0.3s ease; 
        } 
         
        .status { 
            display: inline-block; 
            padding: 8px 15px; 
            border-radius: 20px; 
            font-weight: 600; 
            font-size: 0.9em; 
        } 
         
        .status.focused { 
            background: #d4edda; 
            color: #155724; 
        } 
         
        .status.distracted { 
            background: #fff3cd; 
            color: #856404; 
        } 
         
        .status.sleeping { 
            background: #f8d7da; 
            color: #721c24; 
        } 
         
        .gaze-screen { 
            background: #f0f0f0; 
            border: 2px solid #ddd; 
            border-radius: 10px; 
            position: relative; 
            width: 100%; 
            aspect-ratio: 16/9; 
            overflow: hidden; 
            margin-top: 15px; 
        } 
         
        .gaze-point { 
            position: absolute; 
            width: 30px; 
            height: 30px; 
            background: radial-gradient(circle, rgba(102,126,234,0.8) 0%, rgba(102,126,234,0.3) 70%); 
            border: 2px solid #667eea; 
            border-radius: 50%; 
            transform: translate(-50%, -50%); 
            transition: all 0.2s ease; 
            box-shadow: 0 0 10px rgba(102,126,234,0.5); 
        } 
         
        .prediction-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 10px; 
            margin-top: 15px; 
        } 
         
        .prediction-item { 
            background: #f8f9fa; 
            padding: 12px; 
            border-radius: 8px; 
            text-align: center; 
        } 
         
        .prediction-item label { 
            font-size: 0.9em; 
            color: #666; 
            display: block; 
            margin-bottom: 5px; 
        } 
         
        .prediction-item value { 
            font-size: 1.3em; 
            font-weight: bold; 
            color: #667eea; 
            display: block; 
        } 
         
        .controls { 
            display: flex; 
            gap: 10px; 
            margin-bottom: 20px; 
        } 
         
        .btn { 
            padding: 12px 25px; 
            border: none; 
            border-radius: 8px; 
            font-size: 1em; 
            font-weight: 600; 
            cursor: pointer; 
            transition: all 0.3s ease; 
            color: white; 
        } 
         
        .btn-primary { 
            background: #667eea; 
        } 
         
        .btn-primary:hover { 
            background: #5568d3; 
            transform: scale(1.05); 
        } 
         
        .btn-secondary { 
            background: #764ba2; 
        } 
         
        .btn-secondary:hover { 
            background: #633a8b; 
            transform: scale(1.05); 
        } 
         
        .info-section { 
            background: white; 
            border-radius: 15px; 
            padding: 25px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
        } 
         
        .info-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
        } 
         
        .info-item { 
            padding: 15px; 
            background: #f8f9fa; 
            border-radius: 8px; 
            border-left: 4px solid #667eea; 
        } 
         
        .info-item h3 { 
            color: #667eea; 
            margin-bottom: 10px; 
            font-size: 1.1em; 
        } 
         
        .info-item p { 
            color: #666; 
            line-height: 1.6; 
            font-size: 0.95em; 
        } 
         
        .sequence-info { 
            text-align: center; 
            padding: 15px; 
            background: #667eea; 
            color: white; 
            border-radius: 8px; 
            margin-bottom: 20px; 
        } 
         
        .sequence-info h3 { 
        } 
            margin-bottom: 10px; 
        .sequence-counter { 
            font-size: 1.5em; 
            font-weight: bold; 
        } 
        @media (max-width: 1024px) { 
            .main-grid { 
                grid-template-columns: 1fr; 
            } 
            .info-grid { 
                grid-template-columns: 1fr; 
            } 
            .header h1 { 
                font-size: 1.8em; 
            } 
        } 
    </style> 
</head> 
<body> 
    <div class="container"> 
        <!-- Header --> 
        <div class="header"> 
            <h1> MPIIGaze Live Demonstration</h1> 
            <p>Real-time Eye Gaze Tracking & Attention Classification</p> 
        </div> 
        <!-- Sequence Info --> 
        <div class="sequence-info"> 
            <h3>Current Sequence</h3> 
            <div class="sequence-counter"> 
                <span id="current-seq">1</span> / <span id="total-seq">20</span> 
            </div> 
            <div style="margin-top: 10px;"> 
                <button class="btn btn-secondary" onclick="previousSequence()">← Previous</button> 
                <button class="btn btn-secondary" onclick="nextSequence()">Next →</button> 
            </div> 
        </div> 
        <!-- Main Content --> 
        <div class="main-grid"> 
            <!-- Attention Level Card --> 
            <div class="card"> 
                <h2> 
 Attention Level</h2> 
                <div class="metric"> 
                    <span class="metric-label">Predicted State</span> 
                    <span class="status" id="attention-status">Loading...</span> 
                </div> 
                <div class="metric"> 
                    <span class="metric-label">Confidence</span> 
                    <span class="metric-value" id="attention-conf">--</span> 
                    <div class="confidence-bar"> 
                        <div class="confidence-fill" id="attention-bar" style="width: 0%"></div> 
                    </div> 
                </div> 
                <div style="margin-top: 15px;"> 
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 10px;">Probability Distribution:</div> 
                    <div class="prediction-grid"> 
                        <div class="prediction-item"> 
                            <label>Focused</label> 
                            <value id="prob-focused">--</value> 
                        </div> 
                        <div class="prediction-item"> 
                            <label>Distracted</label> 
                            <value id="prob-distracted">--</value> 
                        </div> 
                        <div class="prediction-item" style="grid-column: 1/-1;"> 
                            <label>Sleeping</label> 
                            <value id="prob-sleeping">--</value> 
                        </div> 
                    </div> 
                </div> 
            </div> 
            <!-- Gaze Estimation Card --> 
            <div class="card"> 
                <h2> Gaze Estimation</h2> 
                <div class="prediction-grid"> 
                    <div class="prediction-item"> 
                        <label>X Coordinate</label> 
                        <value id="gaze-x">--</value> 
                    </div> 
                    <div class="prediction-item"> 
                        <label>Y Coordinate</label> 
                        <value id="gaze-y">--</value> 
                    </div> 
                </div> 
                <div style="margin-top: 15px;"> 
                    <label style="color: #666; font-weight: 600;">Screen Visualization:</label> 
                    <div class="gaze-screen"> 
                        <div class="gaze-point" id="gaze-indicator"></div> 
                    </div> 
                </div> 
                <div class="prediction-grid" style="margin-top: 15px;"> 
                    <div class="prediction-item"> 
                        <label>X (pixels)</label> 
                        <value id="gaze-x-pixel">--</value> 
                    </div> 
                    <div class="prediction-item"> 
                        <label>Y (pixels)</label> 
                        <value id="gaze-y-pixel">--</value> 
                    </div> 
                </div> 
            </div> 
        </div> 
        <!-- Information Section --> 
        <div class="info-section"> 
            <h2 style="color: #667eea; margin-bottom: 20px;">ℹ  About This Demo</h2> 
            <div class="info-grid"> 
                <div class="info-item"> 
                    <h3> Attention Level Classifier</h3> 
                    <p>Classifies eye gaze patterns into three states: Focused (high gaze velocity), Distracted (medium velocity), an
                </div> 
                <div class="info-item"> 
                    <h3> Gaze Estimator</h3> 
                    <p>Predicts the point on screen where the user is looking based on 32 timesteps of temporal gaze and head pose da
                </div> 
                <div class="info-item"> 
                    <h3> CNN-LSTM Architecture</h3> 
                    <p>Uses Conv1D layer for spatial pattern capture, followed by stacked LSTM layers for temporal dependency learnin
                </div> 
                <div class="info-item"> 
                    <h3> Dataset: MPIIGaze</h3> 
                    <p>213,659 images from 15 participants. Contains eye landmarks, gaze coordinates, head pose, and camera calibrati
                </div> 
            </div> 
        </div> 
    </div>
    <script> 
        let totalSequences = 20; 
        let currentIdx = 0; 
        // Load initial prediction 
        window.onload = function() { 
            updatePrediction(); 
        };
        function updatePrediction() { 
            fetch(`/predict/${currentIdx}`) 
                .then(response => response.json()) 
                .then(data => { 
                    // Update attention metrics 
                    const attentionClass = data.attention.class; 
                    document.getElementById('attention-status').textContent = attentionClass; 
                    document.getElementById('attention-status').className = `status ${attentionClass.toLowerCase()}`; 
                    document.getElementById('attention-conf').textContent = (data.attention.confidence * 100).toFixed(1) + '%'; 
                    document.getElementById('attention-bar').style.width = (data.attention.confidence * 100) + '%'; 
                    // Update probabilities 
                    document.getElementById('prob-focused').textContent = (data.attention.all_probs.focused * 100).toFixed(1) + '%'; 
                    document.getElementById('prob-distracted').textContent = (data.attention.all_probs.distracted * 100).toFixed(1) +
                    document.getElementById('prob-sleeping').textContent = (data.attention.all_probs.sleeping * 100).toFixed(1) + '%
                    // Update gaze metrics 
                    document.getElementById('gaze-x').textContent = data.gaze.x.toFixed(3); 
                    document.getElementById('gaze-y').textContent = data.gaze.y.toFixed(3); 
                    document.getElementById('gaze-x-pixel').textContent = data.gaze.x_pixel; 
                    document.getElementById('gaze-y-pixel').textContent = data.gaze.y_pixel; 
                    // Update gaze indicator 
                    const screen = document.querySelector('.gaze-screen'); 
                    const indicator = document.getElementById('gaze-indicator'); 
                    const x = data.gaze.x * 100; 
                    const y = data.gaze.y * 100; 
                    indicator.style.left = x + '%'; 
                    indicator.style.top = y + '%'; 
                    // Update sequence counter 
                    document.getElementById('current-seq').textContent = currentIdx + 1; 
                }) 
                .catch(error => console.error('Error:', error)); 
        } 
        function nextSequence() { 
            if (currentIdx < totalSequences - 1) { 
                currentIdx++; 
                updatePrediction(); 
            } 
        } 
        function previousSequence() { 
            if (currentIdx > 0) { 
                currentIdx--; 
                updatePrediction(); 
            } 
        } 
    </script> 
</body> 
</html> 
""" 
@app.route('/') 
def index(): 
    """Main page""" 
    return render_template_string(HTML_TEMPLATE) 
@app.route('/predict/<int:seq_idx>') 
def predict(seq_idx): 
    """Get predictions for a specific sequence""" 
    try: 
        # Get sequence 
        if seq_idx < 0 or seq_idx >= len(TEST_SEQUENCES): 
            seq_idx = 0 
        sequence = TEST_SEQUENCES[seq_idx] 
        # Extract features 
        features = FeatureExtractor.extract_temporal_features(sequence) 
        # Get predictions 
        attention_pred = predict_attention(features) 
        gaze_pred = predict_gaze(features) 
        return jsonify({ 
            'attention': attention_pred, 
            'gaze': gaze_pred, 
            'sequence_info': { 
                'participant': sequence['participant_id'], 
                'day': sequence['day'], 
                'index': seq_idx 
            } 
        })
    except Exception as e: 
        print(f"Error in prediction: {e}") 
        return jsonify({ 
            'error': str(e), 
            'attention': { 
                'class': 'Error', 
                'confidence': 0.0, 
                'all_probs': {'focused': 0.33, 'distracted': 0.33, 'sleeping': 0.34} 
            }, 
            'gaze': { 
                'x': 0.5, 
                'y': 0.5, 
                'x_pixel': SCREEN_WIDTH_PIXEL // 2, 
                'y_pixel': SCREEN_HEIGHT_PIXEL // 2 
            } 
        }), 500 
@app.route('/info') 
def info(): 
    """Get model information""" 
    return jsonify({ 
        'models_loaded': len(MODELS) > 0, 
        'test_sequences': len(TEST_SEQUENCES), 
        'model_config': { 
            'attention_classes': ATTENTION_NUM_CLASSES, 
            'gaze_output_dim': GAZE_OUTPUT_DIM, 
            'sequence_length': SEQUENCE_LENGTH, 
            'num_features': NUM_FEATURES 
        } 
    }) 
# ============================================================================ 
# MAIN 
# ============================================================================ 
if __name__ == '__main__': 
    print("=" * 80) 
    print("MPIIGaze Live Demonstration UI") 
    print("=" * 80) 
    # Load models and data 
    models_loaded = load_models_and_data() 
    if not models_loaded: 
        print("\n⚠ Warning: Could not load trained models") 
        print("Make sure you have run the training pipeline first:") 
        print("  python main.py") 
        print("\nYou can still view the UI, but predictions will be random.\n") 
    print(f"\nStarting Flask server...") 
    print(f"Open your browser and navigate to: http://localhost:5000") 
    print(f"Press Ctrl+C to stop the server\n") 
    app.run(debug=True, host='localhost', port=5000) 