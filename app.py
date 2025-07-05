
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import json
import time
from datetime import datetime
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
from sports_modules.soccer import SoccerAnalyzer
from sports_modules.baseball import BaseballAnalyzer
from sports_modules.football import FootballAnalyzer
from sports_modules.fitness import FitnessAnalyzer
from sports_modules.gym import GymAnalyzer
from gemini_analyzer import GeminiVideoAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize analyzers
analyzers = {
    'soccer': SoccerAnalyzer(),
    'baseball': BaseballAnalyzer(),
    'football': FootballAnalyzer(),
    'fitness': FitnessAnalyzer(),
    'gym': GymAnalyzer()
}

gemini_analyzer = GeminiVideoAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    sport_type = request.form.get('sport_type', 'soccer')
    
    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if video_file:
        filename = secure_filename(video_file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
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
                
                return jsonify({
                    'success': True,
                    'analysis': sport_analysis,
                    'processed_video': processed_video_path,
                    'original_video': video_path
                })
            else:
                return jsonify({'error': 'Unsupported sport type'}), 400
                
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/processed/<filename>')
def processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
