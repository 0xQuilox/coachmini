
import cv2
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple
import mediapipe as mp

class SoccerAnalyzer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Soccer-specific metrics
        self.ball_touches = []
        self.shooting_positions = []
        self.running_distances = []
        self.body_positions = []

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze soccer video with pose detection and ball tracking"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        analysis_data = {
            "sport": "soccer",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }
        
        frame_count = 0
        last_ball_position = None
        player_positions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                results = self._analyze_frame(frame)
                
                if results:
                    timestamp = frame_count / fps
                    
                    # Track player movement
                    if results.get('player_pose'):
                        player_positions.append({
                            'timestamp': timestamp,
                            'position': results['player_pose']['center'],
                            'pose_landmarks': results['player_pose']['landmarks']
                        })
                    
                    # Detect ball interactions
                    ball_interaction = self._detect_ball_interaction(results, timestamp)
                    if ball_interaction:
                        analysis_data["key_moments"].append(ball_interaction)
                    
                    # Analyze shooting technique
                    shooting_analysis = self._analyze_shooting_technique(results, timestamp)
                    if shooting_analysis:
                        analysis_data["performance_data"].append(shooting_analysis)
        
        cap.release()
        
        # Calculate technical metrics
        analysis_data["technical_metrics"] = self._calculate_soccer_metrics(
            player_positions, analysis_data["key_moments"]
        )
        
        # Generate recommendations
        analysis_data["recommendations"] = self._generate_soccer_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )
        
        return analysis_data

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze single frame for soccer-specific features"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        
        # Pose detection
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Calculate player center
            center_x = (landmarks[11].x + landmarks[12].x) / 2  # Shoulder center
            center_y = (landmarks[11].y + landmarks[12].y) / 2
            
            results['player_pose'] = {
                'center': (center_x, center_y),
                'landmarks': landmarks,
                'left_foot': (landmarks[31].x, landmarks[31].y),
                'right_foot': (landmarks[32].x, landmarks[32].y),
                'left_knee': (landmarks[25].x, landmarks[25].y),
                'right_knee': (landmarks[26].x, landmarks[26].y)
            }
        
        # Ball detection (simplified - using color detection)
        ball_position = self._detect_ball(frame)
        if ball_position:
            results['ball_position'] = ball_position
        
        return results

    def _detect_ball(self, frame: np.ndarray) -> Tuple[int, int]:
        """Detect soccer ball using color and shape detection"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for white/black soccer ball colors
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Filter by size
                # Check if contour is roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Reasonably circular
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            return (cx, cy)
        
        return None

    def _detect_ball_interaction(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Detect when player interacts with ball"""
        if not results.get('player_pose') or not results.get('ball_position'):
            return None
        
        player_feet = [
            results['player_pose']['left_foot'],
            results['player_pose']['right_foot']
        ]
        
        ball_pos = results['ball_position']
        
        # Check distance between feet and ball
        for i, foot in enumerate(['left', 'right']):
            foot_pos = player_feet[i]
            distance = np.sqrt((foot_pos[0] - ball_pos[0])**2 + (foot_pos[1] - ball_pos[1])**2)
            
            if distance < 0.1:  # Threshold for interaction
                return {
                    'type': 'ball_touch',
                    'timestamp': timestamp,
                    'foot': foot,
                    'position': ball_pos,
                    'technique_score': self._evaluate_touch_technique(results['player_pose'])
                }
        
        return None

    def _analyze_shooting_technique(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze shooting technique based on body position"""
        if not results.get('player_pose'):
            return None
        
        pose = results['player_pose']
        landmarks = pose['landmarks']
        
        # Calculate body angles
        left_knee_angle = self._calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]  # Hip, knee, ankle
        )
        right_knee_angle = self._calculate_angle(
            landmarks[24], landmarks[26], landmarks[28]
        )
        
        # Analyze shooting stance
        shooting_indicators = {
            'balanced_stance': abs(left_knee_angle - right_knee_angle) < 20,
            'proper_plant_foot': min(left_knee_angle, right_knee_angle) > 90,
            'body_over_ball': pose['center'][1] < 0.6  # Upper body forward
        }
        
        technique_score = sum(shooting_indicators.values()) / len(shooting_indicators)
        
        return {
            'type': 'shooting_technique',
            'timestamp': timestamp,
            'technique_score': technique_score,
            'indicators': shooting_indicators,
            'knee_angles': {'left': left_knee_angle, 'right': right_knee_angle}
        }

    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            p1 = np.array([point1.x, point1.y])
            p2 = np.array([point2.x, point2.y])
            p3 = np.array([point3.x, point3.y])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            return np.degrees(angle)
        except:
            return 0

    def _evaluate_touch_technique(self, pose: Dict[str, Any]) -> float:
        """Evaluate ball touch technique based on body position"""
        # Simple scoring based on body balance and positioning
        landmarks = pose['landmarks']
        
        # Check if player is balanced
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        balance_score = 1.0 - abs(left_ankle.y - right_ankle.y)
        
        # Check body posture
        head = landmarks[0]
        center = pose['center']
        posture_score = 1.0 - abs(head.x - center[0])
        
        return (balance_score + posture_score) / 2

    def _calculate_soccer_metrics(self, player_positions: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate soccer-specific performance metrics"""
        metrics = {
            'total_touches': len([m for m in key_moments if m['type'] == 'ball_touch']),
            'shooting_attempts': len([m for m in key_moments if m['type'] == 'shooting_technique']),
            'average_technique_score': 0,
            'movement_distance': 0,
            'activity_level': 'moderate'
        }
        
        # Calculate average technique score
        technique_scores = [m.get('technique_score', 0) for m in key_moments if 'technique_score' in m]
        if technique_scores:
            metrics['average_technique_score'] = sum(technique_scores) / len(technique_scores)
        
        # Calculate movement distance
        if len(player_positions) > 1:
            total_distance = 0
            for i in range(1, len(player_positions)):
                prev_pos = player_positions[i-1]['position']
                curr_pos = player_positions[i]['position']
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                total_distance += distance
            
            metrics['movement_distance'] = total_distance
            
            # Determine activity level
            if total_distance > 0.5:
                metrics['activity_level'] = 'high'
            elif total_distance > 0.2:
                metrics['activity_level'] = 'moderate'
            else:
                metrics['activity_level'] = 'low'
        
        return metrics

    def _generate_soccer_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized soccer training recommendations"""
        recommendations = []
        
        # Based on technique score
        if metrics['average_technique_score'] < 0.6:
            recommendations.append("Focus on ball control drills - practice juggling and first touches daily")
            recommendations.append("Work on body positioning when receiving the ball")
        
        # Based on shooting attempts
        if metrics['shooting_attempts'] < 3:
            recommendations.append("Practice shooting more frequently - aim for 20-30 shots per session")
            recommendations.append("Work on shooting from different angles and distances")
        
        # Based on activity level
        if metrics['activity_level'] == 'low':
            recommendations.append("Increase movement and running during training")
            recommendations.append("Practice quick changes of direction and acceleration")
        
        # Based on Gemini analysis
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")
        
        return recommendations

    def create_processed_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str) -> str:
        """Create processed video with soccer analysis overlays"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output path
        output_filename = f"soccer_analysis_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        key_moments = analysis_data.get('key_moments', [])
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Add soccer-specific overlays
            frame = self._add_soccer_overlays(frame, timestamp, key_moments, analysis_data)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        return output_path

    def _add_soccer_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add soccer-specific overlays to frame"""
        # Add performance metrics
        metrics = analysis_data.get('technical_metrics', {})
        
        # Overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "SOCCER ANALYSIS", (20, 35), font, 0.7, (255, 215, 0), 2)
        cv2.putText(frame, f"Ball Touches: {metrics.get('total_touches', 0)}", (20, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Technique Score: {metrics.get('average_technique_score', 0):.1f}/1.0", (20, 80), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Activity: {metrics.get('activity_level', 'N/A').upper()}", (20, 100), font, 0.5, (255, 255, 255), 1)
        
        # Highlight key moments
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 0.5:  # Within 0.5 seconds
                if moment['type'] == 'ball_touch':
                    cv2.putText(frame, "BALL TOUCH!", (20, 130), font, 0.6, (0, 255, 0), 2)
                elif moment['type'] == 'shooting_technique':
                    cv2.putText(frame, "SHOOTING!", (20, 130), font, 0.6, (0, 0, 255), 2)
        
        return frame

import time
