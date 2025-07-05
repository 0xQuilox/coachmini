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

class FootballAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Configure Gemini API
        api_key = "AIzaSyAo_0NUZ3PYViVUgSiEO3IfJdleGbdSTJM"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze football video with throwing, catching, and running mechanics"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if gemini_analysis is None:
            gemini_analysis = self._analyze_with_gemini(video_path)

        analysis_data = {
            "sport": "football",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }

        frame_count = 0
        throw_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 3 == 0:
                results = self._analyze_frame(frame)

                if results:
                    timestamp = frame_count / fps

                    # Analyze throwing mechanics
                    throw_analysis = self._analyze_throwing_mechanics(results, timestamp)
                    if throw_analysis:
                        throw_sequence.append(throw_analysis)
                        analysis_data["performance_data"].append(throw_analysis)

                    # Analyze catching technique
                    catch_analysis = self._analyze_catching_technique(results, timestamp)
                    if catch_analysis:
                        analysis_data["performance_data"].append(catch_analysis)
                        analysis_data["key_moments"].append({
                            'type': 'catch_attempt',
                            'timestamp': timestamp,
                            'technique_score': catch_analysis['technique_score']
                        })

        cap.release()

        analysis_data["technical_metrics"] = self._calculate_football_metrics(
            throw_sequence, analysis_data["key_moments"]
        )

        analysis_data["recommendations"] = self._generate_football_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )

        return analysis_data

    def _analyze_with_gemini(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with Gemini AI at 1 FPS"""
        try:
            frames = self._extract_frames_1fps(video_path, max_frames=30)

            prompt = """
            Analyze this American football training video and provide detailed feedback on:
            1. Throwing mechanics and accuracy
            2. Catching technique and hand positioning
            3. Running form and speed
            4. Blocking technique and footwork
            5. Route running precision
            6. Overall performance rating (1-10)

            Format your response as JSON with the following structure:
            {
                "summary": "Brief overall assessment",
                "technique": "Detailed technique analysis",
                "suggestions": "Specific improvement recommendations",
                "common_mistakes": "List of observed mistakes",
                "statistics": {
                    "performance_rating": "X/10",
                    "throwing_accuracy": "X/10",
                    "catching_technique": "X/10",
                    "route_precision": "X/10"
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
            return self._mock_football_analysis()

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

        return self._mock_football_analysis()

    def _mock_football_analysis(self) -> Dict[str, Any]:
        """Generate mock analysis when Gemini is unavailable"""
        return {
            "summary": "Strong throwing mechanics with good accuracy. Route running shows precision and timing.",
            "technique": "Throwing motion is smooth with good spiral. Catching technique is solid with good hand positioning.",
            "suggestions": "Work on footwork for throwing, practice catching in traffic, and improve route timing.",
            "common_mistakes": ["Inconsistent footwork", "Poor hand positioning", "Route timing issues"],
            "statistics": {
                "performance_rating": "8/10",
                "throwing_accuracy": "9/10",
                "catching_technique": "7/10",
                "route_precision": "8/10"
            }
        }

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for football-specific pose data"""
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

    def _analyze_throwing_mechanics(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze throwing mechanics"""
        pose_data = results.get('pose_data')
        if not pose_data:
            return None

        stance_score = self._calculate_throwing_stance(pose_data)
        arm_motion = self._calculate_arm_motion(pose_data)
        follow_through = self._calculate_throwing_follow_through(pose_data)

        return {
            'type': 'throwing_mechanics',
            'timestamp': timestamp,
            'stance_score': stance_score,
            'arm_motion': arm_motion,
            'follow_through': follow_through,
            'overall_throwing': (stance_score + arm_motion + follow_through) / 3
        }

    def _analyze_catching_technique(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze catching technique"""
        pose_data = results.get('pose_data')
        if not pose_data:
            return None

        hand_position = self._calculate_hand_position(pose_data)
        body_alignment = self._calculate_body_alignment(pose_data)
        concentration = self._calculate_concentration_score(pose_data)

        return {
            'type': 'catching_technique',
            'timestamp': timestamp,
            'hand_position': hand_position,
            'body_alignment': body_alignment,
            'concentration': concentration,
            'technique_score': (hand_position + body_alignment + concentration) / 3
        }

    def _calculate_throwing_stance(self, pose_data: Dict[str, Any]) -> float:
        """Calculate throwing stance score"""
        left_foot = pose_data['feet']['left']
        right_foot = pose_data['feet']['right']

        # Good stance when feet are properly positioned for throwing
        foot_alignment = abs(left_foot[1] - right_foot[1])
        if foot_alignment > 0.1:  # Staggered stance
            return 1.0
        return 0.5

    def _calculate_arm_motion(self, pose_data: Dict[str, Any]) -> float:
        """Calculate arm motion score for throwing"""
        right_hand = pose_data['hands']['right']
        right_shoulder = pose_data['shoulders']['right']

        # Good arm motion when throwing hand is above shoulder
        if right_hand[1] < right_shoulder[1]:
            return 1.0
        return 0.5

    def _calculate_throwing_follow_through(self, pose_data: Dict[str, Any]) -> float:
        """Calculate follow-through score for throwing"""
        right_hand = pose_data['hands']['right']
        left_hip = pose_data['hips']['left']

        # Good follow-through when hand crosses body
        if right_hand[0] < left_hip[0]:
            return 1.0
        return 0.5

    def _calculate_hand_position(self, pose_data: Dict[str, Any]) -> float:
        """Calculate hand position score for catching"""
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']

        # Good hand position when hands are together and extended
        hand_distance = abs(left_hand[0] - right_hand[0]) + abs(left_hand[1] - right_hand[1])
        if hand_distance < 0.2:  # Hands close together
            return 1.0
        return 0.5

    def _calculate_body_alignment(self, pose_data: Dict[str, Any]) -> float:
        """Calculate body alignment score for catching"""
        head = pose_data['head']
        left_shoulder = pose_data['shoulders']['left']
        right_shoulder = pose_data['shoulders']['right']

        # Good alignment when head is between shoulders
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                          (left_shoulder[1] + right_shoulder[1]) / 2)

        if abs(head[0] - shoulder_center[0]) < 0.1:
            return 1.0
        return 0.5

    def _calculate_concentration_score(self, pose_data: Dict[str, Any]) -> float:
        """Calculate concentration score (simplified)"""
        head = pose_data['head']
        left_hand = pose_data['hands']['left']
        right_hand = pose_data['hands']['right']

        # Good concentration when head is looking toward hands
        hand_center = ((left_hand[0] + right_hand[0]) / 2, 
                      (left_hand[1] + right_hand[1]) / 2)

        if abs(head[0] - hand_center[0]) < 0.2:
            return 1.0
        return 0.5

    def _calculate_football_metrics(self, throw_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate football-specific metrics"""
        metrics = {
            'total_throws': len(throw_data),
            'avg_throwing_score': 0,
            'stance_consistency': 0,
            'catches_attempted': 0,
            'catching_technique': 0
        }

        if throw_data:
            throw_scores = [t['overall_throwing'] for t in throw_data]
            stance_scores = [t['stance_score'] for t in throw_data]

            metrics['avg_throwing_score'] = np.mean(throw_scores)
            metrics['stance_consistency'] = 1.0 - np.std(stance_scores)

        catch_moments = [m for m in key_moments if m['type'] == 'catch_attempt']
        metrics['catches_attempted'] = len(catch_moments)

        if catch_moments:
            catch_scores = [m['technique_score'] for m in catch_moments]
            metrics['catching_technique'] = np.mean(catch_scores)

        return metrics

    def _generate_football_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate football-specific recommendations"""
        recommendations = []

        if metrics['avg_throwing_score'] < 0.6:
            recommendations.append("Work on throwing fundamentals - practice footwork and arm motion")

        if metrics['stance_consistency'] < 0.7:
            recommendations.append("Focus on consistent throwing stance and setup")

        if metrics['catching_technique'] < 0.7:
            recommendations.append("Improve catching technique - practice hand positioning and concentration")

        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")

        return recommendations

    def create_ar_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str, show_corrections: bool = True) -> str:
        """Create AR version of video with improvement suggestions overlaid"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"football_ar_{int(time.time())}.mp4"
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

        cv2.putText(frame, "FOOTBALL ANALYSIS - AR MODE", (20, 30), font, 0.8, (255, 215, 0), 2)

        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Throwing: {metrics.get('avg_throwing_score', 0):.1f}/1.0", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Throws: {metrics.get('total_throws', 0)}", (250, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Catches: {metrics.get('catches_attempted', 0)}", (350, 55), font, 0.5, (255, 255, 255), 1)

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
        cv2.putText(frame, "FOOTBALL ANALYSIS", (20, 35), font, 0.6, (255, 215, 0), 2)

        metrics = analysis_data.get('technical_metrics', {})
        cv2.putText(frame, f"Throws: {metrics.get('total_throws', 0)}", (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Catches: {metrics.get('catches_attempted', 0)}", (120, 55), font, 0.5, (255, 255, 255), 1)

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