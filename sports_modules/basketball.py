import cv2
import json
import numpy as np
import mediapipe as mp
import os
import tempfile
import time
import threading
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Load environment variables
load_dotenv()

class BasketballAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Load environment variables and configure Gemini API
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not found in .env file")

        # Rate limiting setup
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.request_lock = threading.Lock()

        # Basketball-specific tracking
        self.shot_data = []
        self.last_shot_time = None
        self.animation_duration = 1.25
        self.current_color = (255, 255, 255)

    def load_shot_data(self, json_path: str):
        """Load shot data from JSON file"""
        try:
            with open(json_path, 'r') as f:
                self.shot_data = json.load(f).get('shots', [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.shot_data = []

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze basketball video with pose detection and shot tracking"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get Gemini analysis if not provided
        if gemini_analysis is None:
            gemini_analysis = self._analyze_with_gemini(video_path)

        analysis_data = {
            "sport": "basketball",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }

        frame_count = 0
        last_head = None
        shot_attempts = []
        shooting_form_data = []

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
                    # Analyze shooting form
                    shooting_analysis = self._analyze_shooting_form(results['pose_data'], timestamp)
                    if shooting_analysis:
                        shooting_form_data.append(shooting_analysis)
                        analysis_data["performance_data"].append(shooting_analysis)

                    # Track shot attempts from data
                    shot_moment = self._check_shot_moments(timestamp)
                    if shot_moment:
                        analysis_data["key_moments"].append(shot_moment)

        cap.release()

        # Calculate metrics
        analysis_data["technical_metrics"] = self._calculate_basketball_metrics(
            shooting_form_data, analysis_data["key_moments"]
        )

        # Generate recommendations
        analysis_data["recommendations"] = self._generate_basketball_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )

        return analysis_data

    def _analyze_with_gemini(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with Gemini AI at 1 FPS"""
        try:
            # Extract frames at 1 FPS
            frames = self._extract_frames_1fps(video_path, max_frames=30)

            prompt = """
            Analyze this basketball training video and provide detailed feedback on:
            1. Shooting form and technique
            2. Body positioning and balance
            3. Follow-through mechanics
            4. Footwork and stance
            5. Overall performance rating (1-10)

            Focus on identifying specific areas for improvement in shooting mechanics.

            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed shooting technique analysis",
                "suggestions": "Specific improvement recommendations",
                "common_mistakes": "List of observed mistakes",
                "statistics": {
                    "performance_rating": "X/10",
                    "shooting_form": "X/10",
                    "consistency": "X/10"
                }
            }
            """

            # Analyze frames with Gemini
            analysis_parts = [prompt]

            for frame_path in frames[:10]:  # Limit for API constraints
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                analysis_parts.append({
                    "mime_type": "image/jpeg",
                    "data": image_data
                })

            with self.request_lock:
                self._rate_limit()
                if self.model:
                    response = self.model.generate_content(analysis_parts)
                else:
                    return self._mock_basketball_analysis()

                self.last_request_time = time.time()
            # Clean up temporary frames
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

            return self._parse_gemini_response(response.text)

        except Exception as e:
            print(f"Gemini analysis failed: {e}")
            return self._mock_basketball_analysis()

    def _rate_limit(self):
        """Simple rate limiting to prevent API overuse"""
        current_time = time.time()
        elapsed_time = current_time - self.last_request_time

        if elapsed_time < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed_time)

    def _extract_frames_1fps(self, video_path: str, max_frames: int = 30) -> List[str]:
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

        return self._mock_basketball_analysis()

    def _mock_basketball_analysis(self) -> Dict[str, Any]:
        """Generate mock analysis when Gemini is unavailable"""
        return {
            "summary": "Good shooting form with consistent follow-through. Shows potential for improvement.",
            "technique": "Shooting stance is balanced, but could improve arc and release timing.",
            "suggestions": "Focus on consistent follow-through, practice free throws daily, work on shot arc.",
            "common_mistakes": ["Inconsistent follow-through", "Low shot arc", "Poor balance"],
            "statistics": {
                "performance_rating": "7/10",
                "shooting_form": "8/10",
                "consistency": "6/10"
            }
        }

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for basketball-specific pose data"""
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

    def _analyze_shooting_form(self, pose_data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze shooting form from pose data"""
        # Calculate shooting metrics
        balance_score = self._calculate_balance(pose_data)
        arm_extension = self._calculate_arm_extension(pose_data)
        follow_through = self._calculate_follow_through(pose_data)

        return {
            'type': 'shooting_form',
            'timestamp': timestamp,
            'balance_score': balance_score,
            'arm_extension': arm_extension,
            'follow_through': follow_through,
            'overall_form': (balance_score + arm_extension + follow_through) / 3
        }

    def _calculate_balance(self, pose_data: Dict[str, Any]) -> float:
        """Calculate balance score from pose"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        # Balance is better when feet are level
        balance = 1.0 - abs(left_foot[1] - right_foot[1])
        return max(0, min(1, balance))

    def _calculate_arm_extension(self, pose_data: Dict[str, Any]) -> float:
        """Calculate arm extension score"""
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']

        # Good extension when hands are above shoulders
        left_shoulder = pose_data['shoulders']['left']
        right_shoulder = pose_data['shoulders']['right']

        extension_score = 0
        if left_hand[1] < left_shoulder[1]:  # Hand above shoulder
            extension_score += 0.5
        if right_hand[1] < right_shoulder[1]:
            extension_score += 0.5

        return extension_score

    def _calculate_follow_through(self, pose_data: Dict[str, Any]) -> float:
        """Calculate follow-through score"""
        # Simplified follow-through based on hand positions
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']

        # Good follow-through when hands are extended forward
        return min(1.0, abs(left_hand[0] - right_hand[0]))

    def _check_shot_moments(self, timestamp: float) -> Dict[str, Any]:
        """Check if current timestamp matches a shot moment"""
        for shot in self.shot_data:
            shot_time = self._parse_timestamp(shot.get('timestamp_of_outcome', '0:00'))
            if abs(timestamp - shot_time) < 0.5:  # Within 0.5 seconds
                return {
                    'type': 'shot_attempt',
                    'timestamp': timestamp,
                    'result': shot.get('result', 'unknown'),
                    'feedback': shot.get('feedback', ''),
                    'shots_made': shot.get('total_shots_made_so_far', 0),
                    'shots_missed': shot.get('total_shots_missed_so_far', 0)
                }
        return None

    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert timestamp (e.g., "0:07.5") to seconds"""
        try:
            minutes, seconds = timestamp.split(':')
            return float(minutes) * 60 + float(seconds)
        except:
            return 0.0

    def _calculate_basketball_metrics(self, shooting_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate basketball-specific metrics"""
        shot_attempts = [k for k in key_moments if k['type'] == 'shot_attempt']

        metrics = {
            'total_shots': len(shot_attempts),
            'shots_made': 0,
            'shots_missed': 0,
            'shooting_percentage': 0,
            'avg_form_score': 0,
            'balance_consistency': 0
        }

        if shot_attempts:
            made_shots = [s for s in shot_attempts if s['result'] == 'made']
            metrics['shots_made'] = len(made_shots)
            metrics['shots_missed'] = len(shot_attempts) - len(made_shots)
            metrics['shooting_percentage'] = len(made_shots) / len(shot_attempts) * 100

        if shooting_data:
            form_scores = [s['overall_form'] for s in shooting_data]
            balance_scores = [s['balance_score'] for s in shooting_data]

            metrics['avg_form_score'] = np.mean(form_scores)
            metrics['balance_consistency'] = 1.0 - np.std(balance_scores)

        return metrics

    def _generate_basketball_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate basketball-specific recommendations"""
        recommendations = []

        # Form-based recommendations
        if metrics['avg_form_score'] < 0.6:
            recommendations.append("Focus on shooting form fundamentals - practice BEEF technique")
            recommendations.append("Work on consistent follow-through and arc")

        if metrics['balance_consistency'] < 0.7:
            recommendations.append("Practice shooting with better balance and foot positioning")

        if metrics['shooting_percentage'] < 50:
            recommendations.append("Increase practice frequency and focus on form over speed")

        # Add Gemini insights
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")

        if gemini_analysis and 'common_mistakes' in gemini_analysis:
            for mistake in gemini_analysis['common_mistakes'][:2]:  # Limit to 2 mistakes
                recommendations.append(f"Avoid: {mistake}")

        return recommendations

    def create_ar_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str, show_corrections: bool = True) -> str:
        """Create AR version of video with improvement suggestions overlaid"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"basketball_ar_{int(time.time())}.mp4"
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
        cv2.putText(frame, "BASKETBALL ANALYSIS - AR MODE", (20, 30), font, 0.8, (255, 215, 0), 2)

        # Current metrics
        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Form Score: {metrics.get('avg_form_score', 0):.1f}/1.0", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Shooting %: {metrics.get('shooting_percentage', 0):.1f}%", (250, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Balance: {metrics.get('balance_consistency', 0):.1f}/1.0", (450, 55), font, 0.5, (255, 255, 255), 1)

        # Real-time feedback
        current_feedback = None
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 2.0:  # Show for 2 seconds
                if moment['type'] == 'shot_attempt':
                    result_color = (0, 255, 0) if moment['result'] == 'made' else (0, 0, 255)
                    cv2.putText(frame, f"SHOT: {moment['result'].upper()}", (20, 90), font, 0.6, result_color, 2)

                    if moment.get('feedback'):
                        current_feedback = moment['feedback']

        # Show improvement suggestions
        if current_feedback:
            self._draw_feedback_text(frame, current_feedback, width, height)
        elif recommendations:
            # Cycle through recommendations every 3 seconds
            rec_index = int(timestamp / 3) % len(recommendations)
            self._draw_feedback_text(frame, recommendations[rec_index], width, height)

        return frame

    def _add_basic_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add basic overlays without corrections"""
        height, width = frame.shape[:2]

        # Simple stats overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "BASKETBALL ANALYSIS", (20, 35), font, 0.6, (255, 215, 0), 2)

        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Shots: {metrics.get('total_shots', 0)}", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Made: {metrics.get('shots_made', 0)}", (120, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Missed: {metrics.get('shots_missed', 0)}", (200, 55), font, 0.5, (255, 255, 255), 1)

        return frame

    def _draw_feedback_text(self, frame: np.ndarray, text: str, width: int, height: int):
        """Draw feedback text at bottom of frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Wrap text to fit screen
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

        # Draw background for text
        total_height = len(lines) * 30 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - total_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Draw text lines
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, 0.6, 2)[0]
            x = (width - text_size[0]) // 2
            y = height - total_height + 30 + (i * 30)

            # Draw text with border
            cv2.putText(frame, line, (x, y), font, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, line, (x, y), font, 0.6, (255, 255, 255), 2)

    def create_processed_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str) -> str:
        """Create processed video with standard analysis overlays"""
        return self.create_ar_video(video_path, analysis_data, output_dir, show_corrections=False)