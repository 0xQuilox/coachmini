
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
import tempfile

class BaseballAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configure Gemini API
        api_key = os.environ.get('GEMINI_API_KEY', "AIzaSyAo_0NUZ3PYViVUgSiEO3IfJdleGbdSTJM")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze baseball video with batting and pitching mechanics"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if gemini_analysis is None:
            gemini_analysis = self._analyze_with_gemini(video_path)
        
        analysis_data = {
            "sport": "baseball",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }
        
        frame_count = 0
        swing_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % 3 == 0:
                results = self._analyze_frame(frame)
                
                if results:
                    timestamp = frame_count / fps
                    
                    # Detect swing mechanics
                    swing_analysis = self._analyze_swing_mechanics(results, timestamp)
                    if swing_analysis:
                        swing_sequence.append(swing_analysis)
                        analysis_data["performance_data"].append(swing_analysis)
                    
                    # Detect pitching mechanics
                    pitch_analysis = self._analyze_pitching_mechanics(results, timestamp)
                    if pitch_analysis:
                        analysis_data["performance_data"].append(pitch_analysis)
                        analysis_data["key_moments"].append({
                            'type': 'pitch',
                            'timestamp': timestamp,
                            'mechanics_score': pitch_analysis['mechanics_score']
                        })
        
        cap.release()
        
        analysis_data["technical_metrics"] = self._calculate_baseball_metrics(
            swing_sequence, analysis_data["key_moments"]
        )
        
        analysis_data["recommendations"] = self._generate_baseball_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )
        
        return analysis_data

    def _analyze_with_gemini(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with Gemini AI at 1 FPS"""
        try:
            frames = self._extract_frames_1fps(video_path, max_frames=30)
            
            prompt = """
            Analyze this baseball training video and provide detailed feedback on:
            1. Batting stance and swing mechanics
            2. Pitching form and accuracy
            3. Fielding technique and positioning
            4. Base running form and speed
            5. Overall athletic performance
            6. Performance rating (1-10)
            
            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed technique analysis",
                "suggestions": "Specific improvement recommendations",
                "common_mistakes": "List of observed mistakes",
                "statistics": {
                    "performance_rating": "X/10",
                    "batting_form": "X/10",
                    "pitching_accuracy": "X/10",
                    "fielding_technique": "X/10"
                }
            }
            """
            
            analysis_parts = [prompt]
            
            for frame_path in frames[:10]:
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                analysis_parts.append({
                    "mime_type": "image/jpeg",
                    "data": image_data
                })
            
            response = self.model.generate_content(analysis_parts)
            
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            return self._parse_gemini_response(response.text)
            
        except Exception as e:
            print(f"Gemini analysis failed: {e}")
            return self._mock_baseball_analysis()

    def _extract_frames_1fps(self, video_path: str, max_frames: int = 30) -> List[str]:
        """Extract frames at 1 FPS from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)
        
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

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response and extract structured data"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return self._mock_baseball_analysis()

    def _mock_baseball_analysis(self) -> Dict[str, Any]:
        """Generate mock analysis when Gemini is unavailable"""
        return {
            "summary": "Solid batting stance with good power potential. Pitching mechanics show consistency.",
            "technique": "Batting stance is well-balanced, but could improve timing. Pitching form shows good follow-through.",
            "suggestions": "Work on pitch recognition, practice timing drills, and strengthen core for more power.",
            "common_mistakes": ["Late swing timing", "Inconsistent stride", "Poor pitch selection"],
            "statistics": {
                "performance_rating": "8/10",
                "batting_form": "7/10",
                "pitching_accuracy": "8/10",
                "fielding_technique": "7/10"
            }
        }

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for baseball-specific pose data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            results['pose_data'] = {
                'head': (landmarks[0].x, landmarks[0].y),
                'shoulders': {
                    'left': (landmarks[11].x, landmarks[11].y),
                    'right': (landmarks[12].x, landmarks[12].y)
                },
                'hands': {
                    'left': (landmarks[15].x, landmarks[15].y),
                    'right': (landmarks[16].x, landmarks[16].y)
                },
                'hips': {
                    'left': (landmarks[23].x, landmarks[23].y),
                    'right': (landmarks[24].x, landmarks[24].y)
                },
                'feet': {
                    'left': (landmarks[31].x, landmarks[31].y),
                    'right': (landmarks[32].x, landmarks[32].y)
                }
            }
        
        return results

    def _analyze_swing_mechanics(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze batting swing mechanics"""
        pose_data = results.get('pose_data')
        if not pose_data:
            return None
            
        stance_score = self._calculate_batting_stance(pose_data)
        swing_path = self._calculate_swing_path(pose_data)
        follow_through = self._calculate_follow_through(pose_data)
        
        return {
            'type': 'swing_mechanics',
            'timestamp': timestamp,
            'stance_score': stance_score,
            'swing_path': swing_path,
            'follow_through': follow_through,
            'overall_swing': (stance_score + swing_path + follow_through) / 3
        }

    def _analyze_pitching_mechanics(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze pitching mechanics"""
        pose_data = results.get('pose_data')
        if not pose_data:
            return None
            
        balance = self._calculate_pitching_balance(pose_data)
        arm_position = self._calculate_arm_position(pose_data)
        leg_drive = self._calculate_leg_drive(pose_data)
        
        return {
            'type': 'pitching_mechanics',
            'timestamp': timestamp,
            'balance_score': balance,
            'arm_position': arm_position,
            'leg_drive': leg_drive,
            'mechanics_score': (balance + arm_position + leg_drive) / 3
        }

    def _calculate_batting_stance(self, pose_data: Dict[str, Any]) -> float:
        """Calculate batting stance score"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']
        
        # Good stance when feet are properly spaced
        foot_spacing = abs(left_foot[0] - right_foot[0])
        optimal_spacing = 0.25
        
        stance_score = 1.0 - abs(foot_spacing - optimal_spacing)
        return max(0, min(1, stance_score))

    def _calculate_swing_path(self, pose_data: Dict[str, Any]) -> float:
        """Calculate swing path score"""
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']
        
        # Good swing when hands are level
        hand_level = 1.0 - abs(left_hand[1] - right_hand[1])
        return max(0, min(1, hand_level))

    def _calculate_follow_through(self, pose_data: Dict[str, Any]) -> float:
        """Calculate follow-through score"""
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']
        left_shoulder = pose_data['shoulders']['left']
        
        # Good follow-through when hands finish high
        if left_hand[1] < left_shoulder[1] and right_hand[1] < left_shoulder[1]:
            return 1.0
        return 0.5

    def _calculate_pitching_balance(self, pose_data: Dict[str, Any]) -> float:
        """Calculate pitching balance score"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']
        
        balance = 1.0 - abs(left_foot[1] - right_foot[1])
        return max(0, min(1, balance))

    def _calculate_arm_position(self, pose_data: Dict[str, Any]) -> float:
        """Calculate arm position score for pitching"""
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']
        left_shoulder = pose_data['shoulders']['left']
        right_shoulder = pose_data['shoulders']['right']
        
        # Good arm position when throwing hand is above shoulder
        throwing_hand = right_hand  # Assuming right-handed
        throwing_shoulder = right_shoulder
        
        if throwing_hand[1] < throwing_shoulder[1]:
            return 1.0
        return 0.5

    def _calculate_leg_drive(self, pose_data: Dict[str, Any]) -> float:
        """Calculate leg drive score for pitching"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']
        
        # Good leg drive when there's proper weight transfer
        leg_separation = abs(left_foot[0] - right_foot[0])
        if leg_separation > 0.3:
            return 1.0
        return 0.5

    def _calculate_baseball_metrics(self, swing_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate baseball-specific metrics"""
        metrics = {
            'total_swings': len(swing_data),
            'avg_swing_score': 0,
            'stance_consistency': 0,
            'pitches_thrown': 0,
            'pitching_mechanics': 0
        }
        
        if swing_data:
            swing_scores = [s['overall_swing'] for s in swing_data]
            stance_scores = [s['stance_score'] for s in swing_data]
            
            metrics['avg_swing_score'] = np.mean(swing_scores)
            metrics['stance_consistency'] = 1.0 - np.std(stance_scores)
        
        pitch_moments = [m for m in key_moments if m['type'] == 'pitch']
        metrics['pitches_thrown'] = len(pitch_moments)
        
        if pitch_moments:
            pitch_scores = [m['mechanics_score'] for m in pitch_moments]
            metrics['pitching_mechanics'] = np.mean(pitch_scores)
        
        return metrics

    def _generate_baseball_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate baseball-specific recommendations"""
        recommendations = []
        
        if metrics['avg_swing_score'] < 0.6:
            recommendations.append("Work on batting fundamentals - practice tee work and timing drills")
        
        if metrics['stance_consistency'] < 0.7:
            recommendations.append("Focus on consistent batting stance and setup")
        
        if metrics['pitching_mechanics'] < 0.7:
            recommendations.append("Improve pitching mechanics - work on balance and follow-through")
        
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")
        
        return recommendations

    def create_ar_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str, show_corrections: bool = True) -> str:
        """Create AR version of video with improvement suggestions overlaid"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f"baseball_ar_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        key_moments = analysis_data.get('key_moments', [])
        recommendations = analysis_data.get('recommendations', [])
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            if show_corrections:
                frame = self._add_ar_overlays(frame, timestamp, key_moments, recommendations, analysis_data)
            else:
                frame = self._add_basic_overlays(frame, timestamp, key_moments, analysis_data)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        return output_path

    def _add_ar_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], recommendations: List[str], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add AR overlays with corrections and suggestions"""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, "BASEBALL ANALYSIS - AR MODE", (20, 30), font, 0.8, (255, 215, 0), 2)
        
        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Swing Score: {metrics.get('avg_swing_score', 0):.1f}/1.0", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Swings: {metrics.get('total_swings', 0)}", (250, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitches: {metrics.get('pitches_thrown', 0)}", (350, 55), font, 0.5, (255, 255, 255), 1)
        
        # Show improvement suggestions
        if recommendations:
            rec_index = int(timestamp / 3) % len(recommendations)
            self._draw_feedback_text(frame, recommendations[rec_index], width, height)
        
        return frame

    def _add_basic_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add basic overlays without corrections"""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "BASEBALL ANALYSIS", (20, 35), font, 0.6, (255, 215, 0), 2)
        
        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Swings: {metrics.get('total_swings', 0)}", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitches: {metrics.get('pitches_thrown', 0)}", (120, 55), font, 0.5, (255, 255, 255), 1)
        
        return frame

    def _draw_feedback_text(self, frame: np.ndarray, text: str, width: int, height: int):
        """Draw feedback text at bottom of frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        words = text.split()
        lines = []
        current_line = []
        max_width = width - 40
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_size = cv2.getTextSize(test_line, font, 0.6, 2)[0]
            
            if text_size[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        total_height = len(lines) * 30 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - total_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, 0.6, 2)[0]
            x = (width - text_size[0]) // 2
            y = height - total_height + 30 + (i * 30)
            
            cv2.putText(frame, line, (x, y), font, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, line, (x, y), font, 0.6, (255, 255, 255), 2)

    def create_processed_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str) -> str:
        """Create processed video with standard analysis overlays"""
        return self.create_ar_video(video_path, analysis_data, output_dir, show_corrections=False)
