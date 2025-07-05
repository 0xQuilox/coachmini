
import cv2
import numpy as np
import json
import os
import time
from typing import Dict, Any, List, Tuple
import mediapipe as mp

class BaseballAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Baseball-specific tracking
        self.swing_phases = []
        self.pitch_mechanics = []
        self.fielding_positions = []

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze baseball video with batting and pitching mechanics"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
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
                        analysis_data["key_moments"].append(pitch_analysis)
        
        cap.release()
        
        # Process swing sequences
        complete_swings = self._process_swing_sequences(swing_sequence)
        analysis_data["key_moments"].extend(complete_swings)
        
        # Calculate metrics
        analysis_data["technical_metrics"] = self._calculate_baseball_metrics(
            analysis_data["performance_data"], analysis_data["key_moments"]
        )
        
        # Generate recommendations
        analysis_data["recommendations"] = self._generate_baseball_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )
        
        return analysis_data

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for baseball-specific pose data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            results['pose_data'] = {
                'shoulders': {
                    'left': (landmarks[11].x, landmarks[11].y),
                    'right': (landmarks[12].x, landmarks[12].y)
                },
                'hips': {
                    'left': (landmarks[23].x, landmarks[23].y),
                    'right': (landmarks[24].x, landmarks[24].y)
                },
                'hands': {
                    'left': (landmarks[15].x, landmarks[15].y),
                    'right': (landmarks[16].x, landmarks[16].y)
                },
                'feet': {
                    'left': (landmarks[31].x, landmarks[31].y),
                    'right': (landmarks[32].x, landmarks[32].y)
                }
            }
        
        return results

    def _analyze_swing_mechanics(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze batting swing mechanics"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Calculate swing metrics
        shoulder_rotation = self._calculate_shoulder_rotation(pose['shoulders'])
        hip_rotation = self._calculate_hip_rotation(pose['hips'])
        hand_speed = self._calculate_hand_speed(pose['hands'])
        
        # Determine swing phase
        swing_phase = self._determine_swing_phase(shoulder_rotation, hip_rotation)
        
        return {
            'type': 'swing_mechanics',
            'timestamp': timestamp,
            'shoulder_rotation': shoulder_rotation,
            'hip_rotation': hip_rotation,
            'hand_speed': hand_speed,
            'swing_phase': swing_phase,
            'technique_score': self._evaluate_swing_technique(pose)
        }

    def _analyze_pitching_mechanics(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze pitching mechanics"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Check for pitching stance
        if self._is_pitching_stance(pose):
            leg_kick_height = self._calculate_leg_kick(pose)
            arm_angle = self._calculate_arm_angle(pose)
            
            return {
                'type': 'pitching_mechanics',
                'timestamp': timestamp,
                'leg_kick_height': leg_kick_height,
                'arm_angle': arm_angle,
                'balance_score': self._evaluate_pitching_balance(pose),
                'technique_score': self._evaluate_pitching_technique(pose)
            }
        
        return None

    def _calculate_shoulder_rotation(self, shoulders: Dict[str, Tuple]) -> float:
        """Calculate shoulder rotation angle"""
        left_shoulder = shoulders['left']
        right_shoulder = shoulders['right']
        
        # Calculate angle from horizontal
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return abs(angle)

    def _calculate_hip_rotation(self, hips: Dict[str, Tuple]) -> float:
        """Calculate hip rotation angle"""
        left_hip = hips['left']
        right_hip = hips['right']
        
        dx = right_hip[0] - left_hip[0]
        dy = right_hip[1] - left_hip[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return abs(angle)

    def _calculate_hand_speed(self, hands: Dict[str, Tuple]) -> float:
        """Estimate hand speed based on position"""
        # Simplified hand speed calculation
        left_hand = hands['left']
        right_hand = hands['right']
        
        # Distance between hands as proxy for swing speed
        distance = np.sqrt((right_hand[0] - left_hand[0])**2 + (right_hand[1] - left_hand[1])**2)
        return distance

    def _determine_swing_phase(self, shoulder_rotation: float, hip_rotation: float) -> str:
        """Determine current phase of swing"""
        if shoulder_rotation < 10 and hip_rotation < 10:
            return "stance"
        elif shoulder_rotation < 30:
            return "load"
        elif shoulder_rotation < 60:
            return "stride"
        elif shoulder_rotation < 90:
            return "contact"
        else:
            return "follow_through"

    def _is_pitching_stance(self, pose: Dict[str, Any]) -> bool:
        """Detect if player is in pitching stance"""
        # Check if one foot is significantly higher (leg kick)
        left_foot = pose['feet']['left']
        right_foot = pose['feet']['right']
        
        height_diff = abs(left_foot[1] - right_foot[1])
        return height_diff > 0.1

    def _calculate_leg_kick(self, pose: Dict[str, Any]) -> float:
        """Calculate leg kick height"""
        left_foot = pose['feet']['left']
        right_foot = pose['feet']['right']
        
        return abs(left_foot[1] - right_foot[1])

    def _calculate_arm_angle(self, pose: Dict[str, Any]) -> float:
        """Calculate pitching arm angle"""
        # Simplified arm angle calculation
        left_hand = pose['hands']['left']
        right_hand = pose['hands']['right']
        
        # Use hand height as proxy for arm angle
        return abs(left_hand[1] - right_hand[1])

    def _evaluate_swing_technique(self, pose: Dict[str, Any]) -> float:
        """Evaluate swing technique quality"""
        # Balance check
        left_foot = pose['feet']['left']
        right_foot = pose['feet']['right']
        balance_score = 1.0 - abs(left_foot[1] - right_foot[1])
        
        # Posture check
        shoulders = pose['shoulders']
        shoulder_level = 1.0 - abs(shoulders['left'][1] - shoulders['right'][1])
        
        return (balance_score + shoulder_level) / 2

    def _evaluate_pitching_balance(self, pose: Dict[str, Any]) -> float:
        """Evaluate pitching balance"""
        # Check if body is balanced during leg kick
        hips = pose['hips']
        hip_level = 1.0 - abs(hips['left'][1] - hips['right'][1])
        
        return hip_level

    def _evaluate_pitching_technique(self, pose: Dict[str, Any]) -> float:
        """Evaluate overall pitching technique"""
        balance = self._evaluate_pitching_balance(pose)
        
        # Check arm position
        hands = pose['hands']
        arm_position = 1.0 - abs(hands['left'][0] - hands['right'][0])
        
        return (balance + arm_position) / 2

    def _process_swing_sequences(self, swing_sequence: List[Dict]) -> List[Dict]:
        """Process swing sequences to identify complete swings"""
        complete_swings = []
        current_swing = []
        
        for swing_data in swing_sequence:
            current_swing.append(swing_data)
            
            # Check if swing is complete
            if swing_data['swing_phase'] == 'follow_through':
                if len(current_swing) >= 3:  # Minimum phases for complete swing
                    complete_swing = {
                        'type': 'complete_swing',
                        'start_time': current_swing[0]['timestamp'],
                        'end_time': current_swing[-1]['timestamp'],
                        'phases': [s['swing_phase'] for s in current_swing],
                        'avg_technique_score': np.mean([s['technique_score'] for s in current_swing]),
                        'max_hand_speed': max([s['hand_speed'] for s in current_swing])
                    }
                    complete_swings.append(complete_swing)
                
                current_swing = []
        
        return complete_swings

    def _calculate_baseball_metrics(self, performance_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate baseball-specific metrics"""
        swing_data = [p for p in performance_data if p['type'] == 'swing_mechanics']
        complete_swings = [k for k in key_moments if k['type'] == 'complete_swing']
        pitching_data = [k for k in key_moments if k['type'] == 'pitching_mechanics']
        
        metrics = {
            'total_swings': len(complete_swings),
            'total_pitches': len(pitching_data),
            'avg_swing_technique': 0,
            'avg_pitching_technique': 0,
            'swing_consistency': 0,
            'dominant_swing_phase': 'stance'
        }
        
        # Calculate swing metrics
        if swing_data:
            technique_scores = [s['technique_score'] for s in swing_data]
            metrics['avg_swing_technique'] = np.mean(technique_scores)
            metrics['swing_consistency'] = 1.0 - np.std(technique_scores)
            
            # Find dominant swing phase
            phases = [s['swing_phase'] for s in swing_data]
            if phases:
                metrics['dominant_swing_phase'] = max(set(phases), key=phases.count)
        
        # Calculate pitching metrics
        if pitching_data:
            technique_scores = [p['technique_score'] for p in pitching_data]
            metrics['avg_pitching_technique'] = np.mean(technique_scores)
        
        return metrics

    def _generate_baseball_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate baseball-specific recommendations"""
        recommendations = []
        
        # Swing recommendations
        if metrics['avg_swing_technique'] < 0.6:
            recommendations.append("Focus on swing mechanics - practice proper stance and balance")
            recommendations.append("Work on keeping your head still and eyes on the ball")
        
        if metrics['swing_consistency'] < 0.7:
            recommendations.append("Practice consistent swing timing with tee work")
            recommendations.append("Focus on repeating the same swing mechanics each time")
        
        # Pitching recommendations
        if metrics['avg_pitching_technique'] < 0.6:
            recommendations.append("Work on pitching balance and leg kick consistency")
            recommendations.append("Practice proper arm angle and follow-through")
        
        # General recommendations
        if metrics['total_swings'] < 5:
            recommendations.append("Increase batting practice frequency")
        
        # Add Gemini insights
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")
        
        return recommendations

    def create_processed_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str) -> str:
        """Create processed video with baseball analysis overlays"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f"baseball_analysis_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
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
            
            frame = self._add_baseball_overlays(frame, timestamp, key_moments, analysis_data)
            out.write(frame)
        
        cap.release()
        out.release()
        
        return output_path

    def _add_baseball_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add baseball-specific overlays"""
        metrics = analysis_data.get('technical_metrics', {})
        
        # Overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "BASEBALL ANALYSIS", (20, 35), font, 0.7, (255, 215, 0), 2)
        cv2.putText(frame, f"Swings: {metrics.get('total_swings', 0)}", (20, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitches: {metrics.get('total_pitches', 0)}", (20, 80), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Swing Technique: {metrics.get('avg_swing_technique', 0):.1f}/1.0", (20, 100), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Consistency: {metrics.get('swing_consistency', 0):.1f}/1.0", (20, 120), font, 0.5, (255, 255, 255), 1)
        
        # Highlight key moments
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 0.5:
                if moment['type'] == 'complete_swing':
                    cv2.putText(frame, "SWING COMPLETE!", (20, 150), font, 0.6, (0, 255, 0), 2)
                elif moment['type'] == 'pitching_mechanics':
                    cv2.putText(frame, "PITCHING!", (20, 150), font, 0.6, (0, 0, 255), 2)
        
        return frame
