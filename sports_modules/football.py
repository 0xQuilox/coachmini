
import cv2
import numpy as np
import json
import os
import time
from typing import Dict, Any, List, Tuple
import mediapipe as mp

class FootballAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Football-specific tracking
        self.throwing_mechanics = []
        self.catching_attempts = []
        self.running_form = []

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze football video with throwing, catching, and running mechanics"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
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
                        analysis_data["key_moments"].append(catch_analysis)
                    
                    # Analyze running form
                    running_analysis = self._analyze_running_form(results, timestamp)
                    if running_analysis:
                        analysis_data["performance_data"].append(running_analysis)
        
        cap.release()
        
        # Process throwing sequences
        complete_throws = self._process_throwing_sequences(throw_sequence)
        analysis_data["key_moments"].extend(complete_throws)
        
        # Calculate metrics
        analysis_data["technical_metrics"] = self._calculate_football_metrics(
            analysis_data["performance_data"], analysis_data["key_moments"]
        )
        
        # Generate recommendations
        analysis_data["recommendations"] = self._generate_football_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )
        
        return analysis_data

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for football-specific pose data"""
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
                'elbows': {
                    'left': (landmarks[13].x, landmarks[13].y),
                    'right': (landmarks[14].x, landmarks[14].y)
                },
                'wrists': {
                    'left': (landmarks[15].x, landmarks[15].y),
                    'right': (landmarks[16].x, landmarks[16].y)
                },
                'hips': {
                    'left': (landmarks[23].x, landmarks[23].y),
                    'right': (landmarks[24].x, landmarks[24].y)
                },
                'knees': {
                    'left': (landmarks[25].x, landmarks[25].y),
                    'right': (landmarks[26].x, landmarks[26].y)
                },
                'ankles': {
                    'left': (landmarks[27].x, landmarks[27].y),
                    'right': (landmarks[28].x, landmarks[28].y)
                }
            }
        
        return results

    def _analyze_throwing_mechanics(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze quarterback throwing mechanics"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Check if in throwing position
        if self._is_throwing_position(pose):
            # Calculate throwing metrics
            arm_angle = self._calculate_throwing_arm_angle(pose)
            shoulder_rotation = self._calculate_shoulder_rotation(pose)
            hip_rotation = self._calculate_hip_rotation(pose)
            foot_position = self._analyze_foot_position(pose)
            
            # Determine throwing phase
            throw_phase = self._determine_throw_phase(arm_angle, shoulder_rotation)
            
            return {
                'type': 'throwing_mechanics',
                'timestamp': timestamp,
                'arm_angle': arm_angle,
                'shoulder_rotation': shoulder_rotation,
                'hip_rotation': hip_rotation,
                'foot_position': foot_position,
                'throw_phase': throw_phase,
                'technique_score': self._evaluate_throwing_technique(pose)
            }
        
        return None

    def _analyze_catching_technique(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze receiver catching technique"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Check if in catching position
        if self._is_catching_position(pose):
            hand_position = self._analyze_hand_position(pose)
            body_position = self._analyze_body_position(pose)
            
            return {
                'type': 'catching_technique',
                'timestamp': timestamp,
                'hand_position': hand_position,
                'body_position': body_position,
                'technique_score': self._evaluate_catching_technique(pose)
            }
        
        return None

    def _analyze_running_form(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze running form and technique"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Check if running
        if self._is_running(pose):
            stride_length = self._calculate_stride_length(pose)
            knee_drive = self._calculate_knee_drive(pose)
            arm_swing = self._calculate_arm_swing(pose)
            posture = self._analyze_running_posture(pose)
            
            return {
                'type': 'running_form',
                'timestamp': timestamp,
                'stride_length': stride_length,
                'knee_drive': knee_drive,
                'arm_swing': arm_swing,
                'posture': posture,
                'technique_score': self._evaluate_running_form(pose)
            }
        
        return None

    def _is_throwing_position(self, pose: Dict[str, Any]) -> bool:
        """Detect if player is in throwing position"""
        # Check arm position - throwing arm should be raised
        right_shoulder = pose['shoulders']['right']
        right_elbow = pose['elbows']['right']
        right_wrist = pose['wrists']['right']
        
        # Arm should be raised and extended
        arm_raised = right_elbow[1] < right_shoulder[1]  # Elbow above shoulder
        arm_extended = right_wrist[0] > right_elbow[0]   # Wrist behind elbow
        
        return arm_raised and arm_extended

    def _is_catching_position(self, pose: Dict[str, Any]) -> bool:
        """Detect if player is in catching position"""
        # Check if arms are extended upward
        left_wrist = pose['wrists']['left']
        right_wrist = pose['wrists']['right']
        left_shoulder = pose['shoulders']['left']
        right_shoulder = pose['shoulders']['right']
        
        # Arms should be raised
        left_arm_raised = left_wrist[1] < left_shoulder[1]
        right_arm_raised = right_wrist[1] < right_shoulder[1]
        
        return left_arm_raised and right_arm_raised

    def _is_running(self, pose: Dict[str, Any]) -> bool:
        """Detect if player is running"""
        # Check for running motion indicators
        left_knee = pose['knees']['left']
        right_knee = pose['knees']['right']
        left_ankle = pose['ankles']['left']
        right_ankle = pose['ankles']['right']
        
        # Check for knee drive and foot position differences
        knee_height_diff = abs(left_knee[1] - right_knee[1])
        foot_height_diff = abs(left_ankle[1] - right_ankle[1])
        
        return knee_height_diff > 0.05 or foot_height_diff > 0.05

    def _calculate_throwing_arm_angle(self, pose: Dict[str, Any]) -> float:
        """Calculate throwing arm angle"""
        shoulder = pose['shoulders']['right']
        elbow = pose['elbows']['right']
        wrist = pose['wrists']['right']
        
        # Calculate angle between shoulder-elbow and elbow-wrist
        v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
        v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def _calculate_shoulder_rotation(self, pose: Dict[str, Any]) -> float:
        """Calculate shoulder rotation"""
        left_shoulder = pose['shoulders']['left']
        right_shoulder = pose['shoulders']['right']
        
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return abs(angle)

    def _calculate_hip_rotation(self, pose: Dict[str, Any]) -> float:
        """Calculate hip rotation"""
        left_hip = pose['hips']['left']
        right_hip = pose['hips']['right']
        
        dx = right_hip[0] - left_hip[0]
        dy = right_hip[1] - left_hip[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return abs(angle)

    def _analyze_foot_position(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze foot positioning for throwing"""
        left_ankle = pose['ankles']['left']
        right_ankle = pose['ankles']['right']
        
        # Calculate foot spacing and positioning
        foot_spacing = abs(left_ankle[0] - right_ankle[0])
        foot_alignment = abs(left_ankle[1] - right_ankle[1])
        
        return {
            'spacing': foot_spacing,
            'alignment': foot_alignment
        }

    def _determine_throw_phase(self, arm_angle: float, shoulder_rotation: float) -> str:
        """Determine current phase of throw"""
        if arm_angle > 150 and shoulder_rotation < 20:
            return "wind_up"
        elif arm_angle > 120 and shoulder_rotation < 40:
            return "cocking"
        elif arm_angle > 90 and shoulder_rotation < 60:
            return "acceleration"
        elif arm_angle < 90:
            return "release"
        else:
            return "follow_through"

    def _analyze_hand_position(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze hand positioning for catching"""
        left_wrist = pose['wrists']['left']
        right_wrist = pose['wrists']['right']
        
        # Calculate hand separation and position
        hand_separation = np.sqrt((left_wrist[0] - right_wrist[0])**2 + 
                                 (left_wrist[1] - right_wrist[1])**2)
        
        avg_hand_height = (left_wrist[1] + right_wrist[1]) / 2
        
        return {
            'separation': hand_separation,
            'height': avg_hand_height
        }

    def _analyze_body_position(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze body positioning for catching"""
        shoulders = pose['shoulders']
        hips = pose['hips']
        
        # Calculate body alignment
        shoulder_level = abs(shoulders['left'][1] - shoulders['right'][1])
        hip_level = abs(hips['left'][1] - hips['right'][1])
        
        return {
            'shoulder_level': shoulder_level,
            'hip_level': hip_level
        }

    def _calculate_stride_length(self, pose: Dict[str, Any]) -> float:
        """Calculate running stride length"""
        left_ankle = pose['ankles']['left']
        right_ankle = pose['ankles']['right']
        
        return abs(left_ankle[0] - right_ankle[0])

    def _calculate_knee_drive(self, pose: Dict[str, Any]) -> float:
        """Calculate knee drive height"""
        left_knee = pose['knees']['left']
        right_knee = pose['knees']['right']
        left_hip = pose['hips']['left']
        right_hip = pose['hips']['right']
        
        # Calculate how high knees are driven relative to hips
        left_knee_drive = abs(left_knee[1] - left_hip[1])
        right_knee_drive = abs(right_knee[1] - right_hip[1])
        
        return max(left_knee_drive, right_knee_drive)

    def _calculate_arm_swing(self, pose: Dict[str, Any]) -> float:
        """Calculate arm swing efficiency"""
        left_wrist = pose['wrists']['left']
        right_wrist = pose['wrists']['right']
        
        # Calculate arm swing range
        arm_swing_range = abs(left_wrist[0] - right_wrist[0])
        
        return arm_swing_range

    def _analyze_running_posture(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze running posture"""
        shoulders = pose['shoulders']
        hips = pose['hips']
        
        # Calculate forward lean
        avg_shoulder_x = (shoulders['left'][0] + shoulders['right'][0]) / 2
        avg_hip_x = (hips['left'][0] + hips['right'][0]) / 2
        
        forward_lean = abs(avg_shoulder_x - avg_hip_x)
        
        return {
            'forward_lean': forward_lean
        }

    def _evaluate_throwing_technique(self, pose: Dict[str, Any]) -> float:
        """Evaluate throwing technique quality"""
        # Check foot position
        foot_pos = self._analyze_foot_position(pose)
        foot_score = 1.0 - min(foot_pos['spacing'], 0.2) / 0.2
        
        # Check shoulder alignment
        shoulder_score = 1.0 - min(abs(pose['shoulders']['left'][1] - pose['shoulders']['right'][1]), 0.1) / 0.1
        
        return (foot_score + shoulder_score) / 2

    def _evaluate_catching_technique(self, pose: Dict[str, Any]) -> float:
        """Evaluate catching technique quality"""
        hand_pos = self._analyze_hand_position(pose)
        body_pos = self._analyze_body_position(pose)
        
        # Good hand separation
        hand_score = min(hand_pos['separation'], 0.3) / 0.3
        
        # Good body alignment
        body_score = 1.0 - min(body_pos['shoulder_level'], 0.1) / 0.1
        
        return (hand_score + body_score) / 2

    def _evaluate_running_form(self, pose: Dict[str, Any]) -> float:
        """Evaluate running form quality"""
        # Check knee drive
        knee_drive = self._calculate_knee_drive(pose)
        knee_score = min(knee_drive, 0.15) / 0.15
        
        # Check posture
        posture = self._analyze_running_posture(pose)
        posture_score = min(posture['forward_lean'], 0.1) / 0.1
        
        return (knee_score + posture_score) / 2

    def _process_throwing_sequences(self, throw_sequence: List[Dict]) -> List[Dict]:
        """Process throwing sequences to identify complete throws"""
        complete_throws = []
        current_throw = []
        
        for throw_data in throw_sequence:
            current_throw.append(throw_data)
            
            if throw_data['throw_phase'] == 'follow_through':
                if len(current_throw) >= 3:
                    complete_throw = {
                        'type': 'complete_throw',
                        'start_time': current_throw[0]['timestamp'],
                        'end_time': current_throw[-1]['timestamp'],
                        'phases': [t['throw_phase'] for t in current_throw],
                        'avg_technique_score': np.mean([t['technique_score'] for t in current_throw]),
                        'peak_arm_angle': max([t['arm_angle'] for t in current_throw])
                    }
                    complete_throws.append(complete_throw)
                
                current_throw = []
        
        return complete_throws

    def _calculate_football_metrics(self, performance_data: List[Dict], key_moments: List[Dict]) -> Dict[str, Any]:
        """Calculate football-specific metrics"""
        throwing_data = [p for p in performance_data if p['type'] == 'throwing_mechanics']
        catching_data = [k for k in key_moments if k['type'] == 'catching_technique']
        running_data = [p for p in performance_data if p['type'] == 'running_form']
        complete_throws = [k for k in key_moments if k['type'] == 'complete_throw']
        
        metrics = {
            'total_throws': len(complete_throws),
            'total_catches': len(catching_data),
            'avg_throwing_technique': 0,
            'avg_catching_technique': 0,
            'avg_running_form': 0,
            'throwing_consistency': 0
        }
        
        # Calculate throwing metrics
        if throwing_data:
            technique_scores = [t['technique_score'] for t in throwing_data]
            metrics['avg_throwing_technique'] = np.mean(technique_scores)
            metrics['throwing_consistency'] = 1.0 - np.std(technique_scores)
        
        # Calculate catching metrics
        if catching_data:
            technique_scores = [c['technique_score'] for c in catching_data]
            metrics['avg_catching_technique'] = np.mean(technique_scores)
        
        # Calculate running metrics
        if running_data:
            technique_scores = [r['technique_score'] for r in running_data]
            metrics['avg_running_form'] = np.mean(technique_scores)
        
        return metrics

    def _generate_football_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate football-specific recommendations"""
        recommendations = []
        
        # Throwing recommendations
        if metrics['avg_throwing_technique'] < 0.6:
            recommendations.append("Focus on throwing mechanics - work on footwork and follow-through")
            recommendations.append("Practice proper arm angle and shoulder rotation")
        
        if metrics['throwing_consistency'] < 0.7:
            recommendations.append("Work on consistent throwing motion and timing")
        
        # Catching recommendations
        if metrics['avg_catching_technique'] < 0.6:
            recommendations.append("Improve catching technique - focus on hand positioning")
            recommendations.append("Practice catching with proper body alignment")
        
        # Running recommendations
        if metrics['avg_running_form'] < 0.6:
            recommendations.append("Work on running form - focus on knee drive and posture")
            recommendations.append("Practice arm swing coordination")
        
        # Add Gemini insights
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")
        
        return recommendations

    def create_processed_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str) -> str:
        """Create processed video with football analysis overlays"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f"football_analysis_{int(time.time())}.mp4"
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
            
            frame = self._add_football_overlays(frame, timestamp, key_moments, analysis_data)
            out.write(frame)
        
        cap.release()
        out.release()
        
        return output_path

    def _add_football_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add football-specific overlays"""
        metrics = analysis_data.get('technical_metrics', {})
        
        # Overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "FOOTBALL ANALYSIS", (20, 35), font, 0.7, (255, 215, 0), 2)
        cv2.putText(frame, f"Throws: {metrics.get('total_throws', 0)}", (20, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Catches: {metrics.get('total_catches', 0)}", (20, 80), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Throwing Technique: {metrics.get('avg_throwing_technique', 0):.1f}/1.0", (20, 100), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Catching Technique: {metrics.get('avg_catching_technique', 0):.1f}/1.0", (20, 120), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Running Form: {metrics.get('avg_running_form', 0):.1f}/1.0", (20, 140), font, 0.5, (255, 255, 255), 1)
        
        # Highlight key moments
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 0.5:
                if moment['type'] == 'complete_throw':
                    cv2.putText(frame, "THROW COMPLETE!", (20, 170), font, 0.6, (0, 255, 0), 2)
                elif moment['type'] == 'catching_technique':
                    cv2.putText(frame, "CATCHING!", (20, 170), font, 0.6, (0, 0, 255), 2)
        
        return frame
