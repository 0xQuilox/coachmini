
import os
import warnings
import logging

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import json
import time
from datetime import datetime
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

from sports_modules.soccer import SoccerAnalyzer
from sports_modules.baseball import BaseballAnalyzer
from sports_modules.football import FootballAnalyzer
from sports_modules.fitness import FitnessAnalyzer
from sports_modules.gym import GymAnalyzer
from sports_modules.basketball import BasketballAnalyzer
from gemini_analyzer import GeminiVideoAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['AR_FOLDER'] = 'ar_videos'

# Production security configurations
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') != 'development'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Configure logging for production
if not app.debug:
    logging.basicConfig(level=logging.INFO)

@app.after_request
def after_request(response):
    """Add security headers for production"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f'Server Error: {e}')
    return jsonify({'error': 'Internal server error'}), 500

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['AR_FOLDER'], exist_ok=True)

# Initialize analyzers
analyzers = {
    'soccer': SoccerAnalyzer(),
    'baseball': BaseballAnalyzer(),
    'football': FootballAnalyzer(),
    'fitness': FitnessAnalyzer(),
    'gym': GymAnalyzer(),
    'basketball': BasketballAnalyzer()
}

gemini_analyzer = GeminiVideoAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for production monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gemini_configured': bool(os.getenv('GEMINI_API_KEY'))
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    sport_type = request.form.get('sport_type', 'soccer')
    create_ar = request.form.get('create_ar', 'false').lower() == 'true'
    
    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check for supported video formats
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Unsupported video format. Supported: {", ".join(allowed_extensions)}'}), 400
    
    if video_file:
        try:
            filename = secure_filename(video_file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            
            # Validate video file after saving
            if not os.path.exists(video_path):
                return jsonify({'error': 'Failed to save video file'}), 500
            
            # Load basketball shot data if available and sport is basketball
            if sport_type == 'basketball':
                ball_json_path = 'ball.json'
                if os.path.exists(ball_json_path):
                    try:
                        analyzers[sport_type].load_shot_data(ball_json_path)
                    except Exception as e:
                        app.logger.warning(f'Failed to load basketball shot data: {e}')
            
            # Process video with Gemini at 1fps
            try:
                analysis_result = gemini_analyzer.analyze_video(video_path, sport_type)
                
                # Process with sport-specific analyzer
                if sport_type in analyzers:
                    sport_analysis = analyzers[sport_type].analyze_video(video_path, analysis_result)
                    
                    # Create processed video with overlays
                    processed_video_path = analyzers[sport_type].create_processed_video(
                        video_path, sport_analysis, app.config['PROCESSED_FOLDER']
                    )
                    
                    response_data = {
                        'success': True,
                        'analysis': sport_analysis,
                        'processed_video': os.path.basename(processed_video_path),
                        'original_video': os.path.basename(video_path)
                    }
                    
                    # Create AR version if requested
                    if create_ar and hasattr(analyzers[sport_type], 'create_ar_video'):
                        try:
                            ar_video_path = analyzers[sport_type].create_ar_video(
                                video_path, sport_analysis, app.config['AR_FOLDER'], show_corrections=True
                            )
                            response_data['ar_video'] = os.path.basename(ar_video_path)
                        except Exception as e:
                            app.logger.warning(f'Failed to create AR video: {e}')
                    
                    return jsonify(response_data)
                else:
                    return jsonify({'error': 'Unsupported sport type'}), 400
                    
            except Exception as e:
                app.logger.error(f'Analysis failed: {str(e)}')
                # Clean up uploaded file on analysis failure
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except:
                        pass
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
                
        except Exception as e:
            app.logger.error(f'Upload processing failed: {str(e)}')
            return jsonify({'error': f'Upload processing failed: {str(e)}'}), 500

@app.route('/processed/<filename>')
def processed_video(filename):
    # Security: Validate filename
    if not filename or '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 400
    
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_video(filename):
    # Security: Validate filename
    if not filename or '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ar/<filename>')
def ar_video(filename):
    # Security: Validate filename
    if not filename or '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 400
    
    file_path = os.path.join(app.config['AR_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(app.config['AR_FOLDER'], filename)

if __name__ == '__main__':
    # Validate environment setup
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY not set. Using mock analysis.")
    
    # Validate required directories exist
    required_dirs = [
        app.config['UPLOAD_FOLDER'],
        app.config['PROCESSED_FOLDER'], 
        app.config['AR_FOLDER']
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
            except Exception as e:
                print(f"Failed to create directory {dir_path}: {e}")
                exit(1)
    
    # Production configuration
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Sports Analysis Pro on port {port}")
    print(f"Debug mode: {debug_mode}")
    print(f"Gemini API configured: {bool(os.getenv('GEMINI_API_KEY'))}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
    except Exception as e:
        print(f"Failed to start application: {e}")
        exit(1)
