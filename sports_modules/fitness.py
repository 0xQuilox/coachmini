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

class FitnessAnalyzer:
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
        """Analyze fitness video with running form and cardio analysis"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if gemini_analysis is None:
            gemini_analysis = self._analyze_with_gemini(video_path)

        analysis_data = {
            "sport": "fitness",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }

        frame_count = 0
        running_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 3 == 0:
                results = self._analyze_frame(frame)

                if results:
                    timestamp = frame_count / fps

                    # Analyze running form
                    running_analysis = self._analyze_running_form(results, timestamp)
                    if running_analysis:
                        running_data.append(running_analysis)
                        analysis_data["performance_data"].append(running_analysis)

                    # Detect key moments
                    key_moment = self._detect_fitness_moments(results, timestamp)
                    if key_moment:
                        analysis_data["key_moments"].append(key_moment)

        cap.release()

        analysis_data["technical_metrics"] = self._calculate_fitness_metrics(
            running_data, analysis_data["key_moments"]
        )

        analysis_data["recommendations"] = self._generate_fitness_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )

        return analysis_data

    def _analyze_with_gemini(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with Gemini AI at 1 FPS"""
        try:
            frames = self._extract_frames_1fps(video_path, max_frames=30)

            prompt = """
            Analyze this fitness training video and provide detailed feedback on:
            1. Running form and efficiency
            2. Cardio intensity and pacing
            3. Movement patterns and biomechanics
            4. Endurance and stamina indicators
            5. Overall fitness level assessment
            6. Performance rating (1-10)

            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed form analysis",
                "suggestions": "Specific improvement recommendations",
                "common_mistakes": "List of observed mistakes",
                "statistics": {
                    "performance_rating": "X/10",
                    "running_form": "X/10",
                    "endurance": "X/10",
                    "efficiency": "X/10"
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
            return self._mock_fitness_analysis()

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

        return self._mock_fitness_analysis()

    def _mock_fitness_analysis(self) -> Dict[str, Any]:
        """Generate mock analysis when Gemini is unavailable"""
        return {
            "summary": "Good running form with efficient stride. Cardio endurance shows steady improvement.",
            "technique": "Running form is efficient with good posture. Pacing is consistent throughout the session.",
            "suggestions": "Focus on breathing techniques, add interval training, and work on core strength.",
            "common_mistakes": ["Slight overstriding", "Inconsistent breathing", "Poor arm swing"],
            "statistics": {
                "performance_rating": "7/10",
                "running_form": "8/10",
                "endurance": "7/10",
                "efficiency": "8/10"
            }
        }

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for fitness-specific pose data"""
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

    def _analyze_running_form(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze running form from pose data"""
        pose_data = results.get('pose_data')
        if not pose_data:
            return None

        posture_score = self._calculate_running_posture(pose_data)
        stride_efficiency = self._calculate_stride_efficiency(pose_data)
        arm_swing = self._calculate_arm_swing(pose_data)

        return {
            'type': 'running_form',
            'timestamp': timestamp,
            'posture_score': posture_score,
            'stride_efficiency': stride_efficiency,
            'arm_swing': arm_swing,
            'overall_form': (posture_score + stride_efficiency + arm_swing) / 3
        }

    def _calculate_running_posture(self, pose_data: Dict[str, Any]) -> float:
        """Calculate running posture score"""
        head = pose_data['head']
        left_shoulder = pose_data['shoulders']['left']
        right_shoulder = pose_data['shoulders']['right']
        left_hip = pose_data['hips']['left']

        # Good posture when head is aligned with shoulders and hips
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        alignment_score = 1.0 - abs(head[0] - shoulder_center_x)

        return max(0, min(1, alignment_score))

    def _calculate_stride_efficiency(self, pose_data: Dict[str, Any]) -> float:
        """Calculate stride efficiency score"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        # Good stride when there's appropriate foot separation
        stride_length = abs(left_foot[0] - right_foot[0])
        optimal_stride = 0.3  # Normalized optimal stride length

        efficiency = 1.0 - abs(stride_length - optimal_stride)
        return max(0, min(1, efficiency))

    def _calculate_arm_swing(self, pose_data: Dict[str, Any]) -> float:
        """Calculate arm swing score"""
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']
        left_shoulder = pose_data['shoulders']['left']
        right_shoulder = pose_data['shoulders']['right']

        # Good arm swing when hands move opposite to each other
        hand_separation = abs(left_hand[0] - right_hand[0])
        if hand_separation > 0.2:
            return 1.0
        return 0.5

    def _detect_fitness_moments(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Detect key fitness moments"""
        pose_data = results.get('pose_data')
        if not pose_data:
            return None

        # Detect high intensity moments (simplified)
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        if abs(left_foot[1] - right_foot[1]) > 0.15:  # One foot significantly higher
            return {
                'type': 'high_intensity',
                'timestamp': timestamp,
                'intensity_level': 'high'
            }

        return None

    def _calculate_fitness_metrics(self, running_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate fitness-specific metrics"""
        metrics = {
            'total_analysis_points': len(running_data),
            'avg_form_score': 0,
            'posture_consistency': 0,
            'high_intensity_moments': 0,
            'endurance_estimate': 0
        }

        if running_data:
            form_scores = [r['overall_form'] for r in running_data]
            posture_scores = [r['posture_score'] for r in running_data]

            metrics['avg_form_score'] = np.mean(form_scores)
            metrics['posture_consistency'] = 1.0 - np.std(posture_scores)

        high_intensity = [m for m in key_moments if m['type'] == 'high_intensity']
        metrics['high_intensity_moments'] = len(high_intensity)

        # Estimate endurance based on consistency over time
        if running_data:
            metrics['endurance_estimate'] = metrics['posture_consistency'] * 0.8 + metrics['avg_form_score'] * 0.2

        return metrics

    def _generate_fitness_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate fitness-specific recommendations"""
        recommendations = []

        if metrics['avg_form_score'] < 0.6:
            recommendations.append("Focus on running form fundamentals - practice proper posture and stride")

        if metrics['posture_consistency'] < 0.7:
            recommendations.append("Work on maintaining consistent posture throughout your workout")

        if metrics['high_intensity_moments'] < 5:
            recommendations.append("Add more high-intensity intervals to improve cardiovascular fitness")

        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")

        return recommendations

    def create_ar_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str, show_corrections: bool = True) -> str:
        """Create AR version of video with improvement suggestions overlaid"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"fitness_ar_{int(time.time())}.mp4"
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

        cv2.putText(frame, "FITNESS ANALYSIS - AR MODE", (20, 30), font, 0.8, (255, 215, 0), 2)

        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Form: {metrics.get('avg_form_score', 0):.1f}/1.0", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Posture: {metrics.get('posture_consistency', 0):.1f}/1.0", (200, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Intensity: {metrics.get('high_intensity_moments', 0)}", (400, 55), font, 0.5, (255, 255, 255), 1)

        # Real-time feedback
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 2.0:
                if moment['type'] == 'high_intensity':
                    cv2.putText(frame, "HIGH INTENSITY DETECTED", (20, 90), font, 0.6, (0, 255, 255), 2)

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
        cv2.putText(frame, "FITNESS ANALYSIS", (20, 35), font, 0.6, (255, 215, 0), 2)

        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Form: {metrics.get('avg_form_score', 0):.1f}", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Intensity: {metrics.get('high_intensity_moments', 0)}", (120, 55), font, 0.5, (255, 255, 255), 1)

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
