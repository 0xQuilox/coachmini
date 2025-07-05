
import os
import cv2
import json
import time
import tempfile
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
import threading

class GeminiVideoAnalyzer:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not found in .env file")
        
        # Rate limiting setup
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        self.request_lock = threading.Lock()

    def _rate_limit_request(self):
        """Ensure we don't exceed API rate limits"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()

    def extract_frames_1fps(self, video_path: str, max_frames: int = 30) -> List[str]:
        """Extract frames at 1 FPS from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Extract every nth frame for 1 FPS
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        temp_dir = tempfile.mkdtemp()
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(temp_dir, f"frame_{extracted_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                extracted_count += 1
                
            frame_count += 1
        
        cap.release()
        return frames

    def analyze_video(self, video_path: str, sport_type: str) -> Dict[str, Any]:
        """Analyze video using Gemini AI"""
        if not self.model:
            return self._mock_analysis(sport_type)
        
        try:
            # Extract frames at 1 FPS
            frames = self.extract_frames_1fps(video_path)
            
            # Create sport-specific prompt
            prompt = self._create_sport_prompt(sport_type)
            
            # Analyze frames with Gemini
            analysis_parts = [prompt]
            
            for frame_path in frames[:10]:  # Limit to first 10 frames for API limits
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                analysis_parts.append({
                    "mime_type": "image/jpeg",
                    "data": image_data
                })
            
            # Apply rate limiting before making API request
            self._rate_limit_request()
            response = self.model.generate_content(analysis_parts)
            
            # Clean up temporary frames
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            # Parse and structure the response
            return self._parse_gemini_response(response.text, sport_type)
            
        except Exception as e:
            print(f"Gemini analysis failed: {e}")
            return self._mock_analysis(sport_type)

    def _create_sport_prompt(self, sport_type: str) -> str:
        """Create sport-specific analysis prompt"""
        prompts = {
            'soccer': """
            Analyze this soccer training video and provide detailed feedback on:
            1. Ball control and touch
            2. Passing accuracy and technique
            3. Shooting form and power
            4. Dribbling skills
            5. Tactical positioning
            6. Overall performance rating (1-10)
            
            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed technique analysis",
                "suggestions": "Specific improvement recommendations",
                "statistics": {
                    "performance_rating": "X/10",
                    "technique_score": "X/10",
                    "consistency": "X/10"
                }
            }
            """,
            
            'baseball': """
            Analyze this baseball training video and provide detailed feedback on:
            1. Batting stance and swing mechanics
            2. Pitching form and accuracy
            3. Fielding technique
            4. Base running form
            5. Overall athletic performance
            6. Performance rating (1-10)
            
            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed technique analysis",
                "suggestions": "Specific improvement recommendations",
                "statistics": {
                    "performance_rating": "X/10",
                    "technique_score": "X/10",
                    "consistency": "X/10"
                }
            }
            """,
            
            'football': """
            Analyze this American football training video and provide detailed feedback on:
            1. Throwing mechanics and accuracy
            2. Catching technique
            3. Running form and speed
            4. Blocking technique
            5. Route running precision
            6. Overall performance rating (1-10)
            
            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed technique analysis",
                "suggestions": "Specific improvement recommendations",
                "statistics": {
                    "performance_rating": "X/10",
                    "technique_score": "X/10",
                    "consistency": "X/10"
                }
            }
            """,
            
            'fitness': """
            Analyze this fitness training video and provide detailed feedback on:
            1. Running form and efficiency
            2. Cardio intensity and pacing
            3. Movement patterns
            4. Endurance and stamina
            5. Overall fitness level
            6. Performance rating (1-10)
            
            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed form analysis",
                "suggestions": "Specific improvement recommendations",
                "statistics": {
                    "performance_rating": "X/10",
                    "form_score": "X/10",
                    "endurance": "X/10"
                }
            }
            """,
            
            'gym': """
            Analyze this gym workout video and provide detailed feedback on:
            1. Exercise form and technique
            2. Range of motion
            3. Muscle activation
            4. Safety assessment
            5. Rep quality and consistency
            6. Overall performance rating (1-10)
            
            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed form analysis",
                "suggestions": "Specific improvement recommendations",
                "statistics": {
                    "performance_rating": "X/10",
                    "form_score": "X/10",
                    "safety_score": "X/10"
                }
            }
            """
        }
        
        return prompts.get(sport_type, prompts['soccer'])

    def _parse_gemini_response(self, response_text: str, sport_type: str) -> Dict[str, Any]:
        """Parse Gemini response and extract structured data"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback to structured parsing
                return {
                    "summary": response_text[:200] + "...",
                    "technique": "Analysis based on video frames",
                    "suggestions": "Continue practicing and focus on consistency",
                    "statistics": {
                        "performance_rating": "8/10",
                        "technique_score": "7/10",
                        "consistency": "8/10"
                    }
                }
        except:
            return self._mock_analysis(sport_type)

    def _mock_analysis(self, sport_type: str) -> Dict[str, Any]:
        """Generate mock analysis when Gemini is unavailable"""
        mock_analyses = {
            'soccer': {
                "summary": "Good ball control with room for improvement in shooting accuracy. Shows strong potential.",
                "technique": "Ball touches are generally clean, but shooting technique needs work. Focus on keeping your head up and following through.",
                "suggestions": "Practice shooting drills daily, work on first touch control, and improve weak foot skills.",
                "statistics": {
                    "performance_rating": "7/10",
                    "ball_control": "8/10",
                    "shooting_accuracy": "6/10",
                    "passing_accuracy": "7/10"
                }
            },
            'baseball': {
                "summary": "Solid batting stance with good power potential. Pitching mechanics show consistency.",
                "technique": "Batting stance is well-balanced, but could improve timing. Pitching form shows good follow-through.",
                "suggestions": "Work on pitch recognition, practice timing drills, and strengthen core for more power.",
                "statistics": {
                    "performance_rating": "8/10",
                    "batting_form": "7/10",
                    "pitching_accuracy": "8/10",
                    "fielding_technique": "7/10"
                }
            },
            'football': {
                "summary": "Strong throwing mechanics with good accuracy. Route running shows precision and timing.",
                "technique": "Throwing motion is smooth with good spiral. Catching technique is solid with good hand positioning.",
                "suggestions": "Work on footwork for throwing, practice catching in traffic, and improve route timing.",
                "statistics": {
                    "performance_rating": "8/10",
                    "throwing_accuracy": "9/10",
                    "catching_technique": "7/10",
                    "route_precision": "8/10"
                }
            },
            'fitness': {
                "summary": "Good running form with efficient stride. Cardio endurance shows steady improvement.",
                "technique": "Running form is efficient with good posture. Pacing is consistent throughout the session.",
                "suggestions": "Focus on breathing techniques, add interval training, and work on core strength.",
                "statistics": {
                    "performance_rating": "7/10",
                    "running_form": "8/10",
                    "endurance": "7/10",
                    "efficiency": "8/10"
                }
            },
            'gym': {
                "summary": "Excellent form on compound movements. Shows good understanding of proper technique.",
                "technique": "Range of motion is full and controlled. Muscle activation appears optimal throughout exercises.",
                "suggestions": "Maintain current form, gradually increase weight, and add variety to prevent plateaus.",
                "statistics": {
                    "performance_rating": "9/10",
                    "form_score": "9/10",
                    "safety_score": "10/10",
                    "consistency": "8/10"
                }
            }
        }
        
        return mock_analyses.get(sport_type, mock_analyses['soccer'])
