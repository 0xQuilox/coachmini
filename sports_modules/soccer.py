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
import threading

class SoccerAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not found in environment variables")

        # Rate limiting setup
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.request_lock = threading.Lock()

        # Soccer-specific tracking
        self.ball_touches = []
        self.passes = []
        self.shots = []

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze soccer video with pose detection and ball tracking"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get Gemini analysis if not provided
        if gemini_analysis is None:
            gemini_analysis = self._analyze_with_gemini(video_path)

        analysis_data = {
            "sport": "soccer",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }

        frame_count = 0
        ball_control_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                results = self._analyze_frame(frame)

                if results and results.get('pose_data'):
                    # Analyze ball control
                    control_analysis = self._analyze_ball_control(results['pose_data'], timestamp)
                    if control_analysis:
                        ball_control_data.append(control_analysis)
                        analysis_data["performance_data"].append(control_analysis)

                    # Detect key moments
                    key_moment = self._detect_key_moments(results, timestamp)
                    if key_moment:
                        analysis_data["key_moments"].append(key_moment)

        cap.release()

        # Calculate metrics
        analysis_data["technical_metrics"] = self._calculate_soccer_metrics(
            ball_control_data, analysis_data["key_moments"]
        )

        # Generate recommendations
        analysis_data["recommendations"] = self._generate_soccer_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )

        return analysis_data

    def _analyze_with_gemini(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with Gemini AI at 1 FPS"""
        if not self.model:
            print("Gemini model not initialized, skipping analysis.")
            return self._mock_soccer_analysis()
        try:
            frames = self._extract_frames_1fps(video_path, max_frames=30)

            prompt = """
            Analyze this soccer training video and provide detailed feedback on:
            1. Ball control and first touch
            2. Passing accuracy and technique
            3. Shooting form and power
            4. Dribbling skills and agility
            5. Tactical positioning and awareness
            6. Overall performance rating (1-10)

            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed technique analysis",
                "suggestions": "Specific improvement recommendations",
                "common_mistakes": "List of observed mistakes",
                "statistics": {
                    "performance_rating": "X/10",
                    "ball_control": "X/10",
                    "passing_accuracy": "X/10",
                    "shooting_technique": "X/10"
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

            with self.request_lock:
                if time.time() - self.last_request_time < self.min_request_interval:
                    time.sleep(self.min_request_interval - (time.time() - self.last_request_time))

                response = self.model.generate_content(analysis_parts)
                self.last_request_time = time.time()

            # Clean up temporary frames
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

            return self._parse_gemini_response(response.text)

        except Exception as e:
            print(f"Gemini analysis failed: {e}")
            return self._mock_soccer_analysis()

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

        return self._mock_soccer_analysis()

    def _mock_soccer_analysis(self) -> Dict[str, Any]:
        """Generate mock analysis when Gemini is unavailable"""
        return {
            "summary": "Good ball control with room for improvement in shooting accuracy. Shows strong potential.",
            "technique": "Ball touches are generally clean, but shooting technique needs work. Focus on keeping your head up.",
            "suggestions": "Practice shooting drills daily, work on first touch control, and improve weak foot skills.",
            "common_mistakes": ["Heavy first touch", "Low shooting accuracy", "Poor weak foot control"],
            "statistics": {
                "performance_rating": "7/10",
                "ball_control": "8/10",
                "passing_accuracy": "7/10",
                "shooting_technique": "6/10"
            }
        }

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for soccer-specific pose data"""
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

    def _analyze_ball_control(self, pose_data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze ball control from pose data"""
        balance_score = self._calculate_balance(pose_data)
        foot_positioning = self._calculate_foot_positioning(pose_data)
        body_posture = self._calculate_body_posture(pose_data)

        return {
            'type': 'ball_control',
            'timestamp': timestamp,
            'balance_score': balance_score,
            'foot_positioning': foot_positioning,
            'body_posture': body_posture,
            'overall_control': (balance_score + foot_positioning + body_posture) / 3
        }

    def _calculate_balance(self, pose_data: Dict[str, Any]) -> float:
        """Calculate balance score from pose"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        balance = 1.0 - abs(left_foot[1] - right_foot[1])
        return max(0, min(1, balance))

    def _calculate_foot_positioning(self, pose_data: Dict[str, Any]) -> float:
        """Calculate foot positioning score"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        # Good positioning when feet are shoulder-width apart
        foot_distance = abs(left_foot[0] - right_foot[0])
        optimal_distance = 0.3  # Normalized shoulder width

        score = 1.0 - abs(foot_distance - optimal_distance)
        return max(0, min(1, score))

    def _calculate_body_posture(self, pose_data: Dict[str, Any]) -> float:
        """Calculate body posture score"""
        head = pose_data['head']
        left_hip = pose_data['hips']['left']
        right_hip = pose_data['hips']['right']

        # Good posture when head is above hips
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        posture_score = 1.0 if head[1] < hip_center_y else 0.5

        return posture_score

    def _detect_key_moments(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Detect key soccer moments"""
        # Simplified key moment detection
        pose_data = results.get('pose_data')
        if not pose_data:
            return None

        # Detect potential shooting motion
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        if abs(left_foot[1] - right_foot[1]) > 0.2:  # One foot significantly higher
            return {
                'type': 'shooting_motion',
                'timestamp': timestamp,
                'confidence': 0.7
            }

        return None

    def _calculate_soccer_metrics(self, control_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate soccer-specific metrics"""
        metrics = {
            'total_touches': len(control_data),
            'avg_ball_control': 0,
            'balance_consistency': 0,
            'key_moments': len(key_moments),
            'shooting_attempts': 0
        }

        if control_data:
            control_scores = [d['overall_control'] for d in control_data]
            balance_scores = [d['balance_score'] for d in control_data]

            metrics['avg_ball_control'] = np.mean(control_scores)
            metrics['balance_consistency'] = 1.0 - np.std(balance_scores)

        shooting_moments = [m for m in key_moments if m['type'] == 'shooting_motion']
        metrics['shooting_attempts'] = len(shooting_moments)

        return metrics

    def _generate_soccer_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate soccer-specific recommendations"""
        recommendations = []

        if metrics['avg_ball_control'] < 0.6:
            recommendations.append("Practice ball control drills - juggling and first touch exercises")

        if metrics['balance_consistency'] < 0.7:
            recommendations.append("Work on balance and core strength for better stability")

        if metrics['shooting_attempts'] < 3:
            recommendations.append("Be more aggressive in shooting - practice finishing drills")

        # Add Gemini insights
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")

        return recommendations

    def create_ar_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str, show_corrections: bool = True) -> str:
        """Create AR version of video with improvement suggestions overlaid"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"soccer_ar_{int(time.time())}.mp4"
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

        # Semi-transparent overlay background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(frame, "SOCCER ANALYSIS - AR MODE", (20, 30), font, 0.8, (255, 215, 0), 2)

        # Current metrics
        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Ball Control: {metrics.get('avg_ball_control', 0):.1f}/1.0", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Balance: {metrics.get('balance_consistency', 0):.1f}/1.0", (250, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Touches: {metrics.get('total_touches', 0)}", (450, 55), font, 0.5, (255, 255, 255), 1)

        # Real-time feedback
        current_feedback = None
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 2.0:
                if moment['type'] == 'shooting_motion':
                    cv2.putText(frame, "SHOOTING MOTION DETECTED", (20, 90), font, 0.6, (0, 255, 255), 2)

        # Show improvement suggestions
        if recommendations:
            rec_index = int(timestamp / 3) % len(recommendations)
            self._draw_feedback_text(frame, recommendations[rec_index], width, height)

        return frame

    def _add_basic_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add basic overlays without corrections"""
        height, width = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "SOCCER ANALYSIS", (20, 35), font, 0.6, (255, 215, 0), 2)

        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Touches: {metrics.get('total_touches', 0)}", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Shots: {metrics.get('shooting_attempts', 0)}", (120, 55), font, 0.5, (255, 255, 255), 1)

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