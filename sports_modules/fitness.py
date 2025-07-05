
import cv2
import numpy as np
import json
import os
import time
from typing import Dict, Any, List, Tuple
import mediapipe as mp

class FitnessAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Fitness-specific tracking
        self.running_metrics = []
        self.cardio_intervals = []
        self.movement_patterns = []

    def analyze_video(self, video_path: str, gemini_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fitness video with cardio and movement analysis"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        analysis_data = {
            "sport": "fitness",
            "gemini_analysis": gemini_analysis,
            "technical_metrics": {},
            "performance_data": [],
            "key_moments": [],
            "recommendations": []
        }
        
        frame_count = 0
        position_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % 5 == 0:
                results = self._analyze_frame(frame)
                
                if results:
                    timestamp = frame_count / fps
                    position_history.append((timestamp, results))
                    
                    # Analyze running form
                    running_analysis = self._analyze_running_form(results, timestamp)
                    if running_analysis:
                        analysis_data["performance_data"].append(running_analysis)
                    
                    # Analyze cardio intensity
                    cardio_analysis = self._analyze_cardio_intensity(results, timestamp, position_history)
                    if cardio_analysis:
                        analysis_data["performance_data"].append(cardio_analysis)
                    
                    # Detect movement patterns
                    movement_analysis = self._analyze_movement_patterns(results, timestamp)
                    if movement_analysis:
                        analysis_data["key_moments"].append(movement_analysis)
        
        cap.release()
        
        # Calculate fitness metrics
        analysis_data["technical_metrics"] = self._calculate_fitness_metrics(
            analysis_data["performance_data"], position_history
        )
        
        # Generate recommendations
        analysis_data["recommendations"] = self._generate_fitness_recommendations(
            analysis_data["technical_metrics"], gemini_analysis
        )
        
        return analysis_data

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for fitness-specific pose data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Calculate center of mass
            center_x = (landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) / 4
            center_y = (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) / 4
            
            results['pose_data'] = {
                'center_of_mass': (center_x, center_y),
                'head': (landmarks[0].x, landmarks[0].y),
                'shoulders': {
                    'left': (landmarks[11].x, landmarks[11].y),
                    'right': (landmarks[12].x, landmarks[12].y)
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
                },
                'wrists': {
                    'left': (landmarks[15].x, landmarks[15].y),
                    'right': (landmarks[16].x, landmarks[16].y)
                }
            }
        
        return results

    def _analyze_running_form(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze running form and technique"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Check if running (based on movement patterns)
        if self._is_running_motion(pose):
            # Calculate running metrics
            stride_analysis = self._calculate_stride_metrics(pose)
            posture_analysis = self._analyze_running_posture(pose)
            arm_swing_analysis = self._analyze_arm_swing(pose)
            foot_strike_analysis = self._analyze_foot_strike(pose)
            
            return {
                'type': 'running_form',
                'timestamp': timestamp,
                'stride_metrics': stride_analysis,
                'posture': posture_analysis,
                'arm_swing': arm_swing_analysis,
                'foot_strike': foot_strike_analysis,
                'efficiency_score': self._calculate_running_efficiency(pose)
            }
        
        return None

    def _analyze_cardio_intensity(self, results: Dict[str, Any], timestamp: float, position_history: List) -> Dict[str, Any]:
        """Analyze cardio intensity based on movement"""
        if not results.get('pose_data') or len(position_history) < 10:
            return None
        
        # Calculate movement velocity over recent frames
        recent_positions = position_history[-10:]
        velocities = []
        
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1][1]['pose_data']['center_of_mass']
            curr_pos = recent_positions[i][1]['pose_data']['center_of_mass']
            time_diff = recent_positions[i][0] - recent_positions[i-1][0]
            
            if time_diff > 0:
                velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2) / time_diff
                velocities.append(velocity)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            intensity = self._classify_intensity(avg_velocity)
            
            return {
                'type': 'cardio_intensity',
                'timestamp': timestamp,
                'velocity': avg_velocity,
                'intensity': intensity,
                'heart_rate_estimate': self._estimate_heart_rate(intensity)
            }
        
        return None

    def _analyze_movement_patterns(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Analyze movement patterns and exercises"""
        if not results.get('pose_data'):
            return None
        
        pose = results['pose_data']
        
        # Detect specific movement patterns
        movement_type = self._detect_movement_type(pose)
        
        if movement_type:
            return {
                'type': 'movement_pattern',
                'timestamp': timestamp,
                'movement_type': movement_type,
                'form_score': self._evaluate_movement_form(pose, movement_type)
            }
        
        return None

    def _is_running_motion(self, pose: Dict[str, Any]) -> bool:
        """Detect if current pose indicates running motion"""
        # Check for alternating leg positions
        left_knee = pose['knees']['left']
        right_knee = pose['knees']['right']
        left_ankle = pose['ankles']['left']
        right_ankle = pose['ankles']['right']
        
        # Check for significant height differences indicating running gait
        knee_height_diff = abs(left_knee[1] - right_knee[1])
        ankle_height_diff = abs(left_ankle[1] - right_ankle[1])
        
        return knee_height_diff > 0.08 or ankle_height_diff > 0.06

    def _calculate_stride_metrics(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stride length and cadence metrics"""
        left_ankle = pose['ankles']['left']
        right_ankle = pose['ankles']['right']
        
        # Calculate stride length (distance between feet)
        stride_length = np.sqrt((left_ankle[0] - right_ankle[0])**2 + (left_ankle[1] - right_ankle[1])**2)
        
        # Calculate knee drive
        left_knee = pose['knees']['left']
        right_knee = pose['knees']['right']
        left_hip = pose['hips']['left']
        right_hip = pose['hips']['right']
        
        left_knee_drive = abs(left_knee[1] - left_hip[1])
        right_knee_drive = abs(right_knee[1] - right_hip[1])
        avg_knee_drive = (left_knee_drive + right_knee_drive) / 2
        
        return {
            'stride_length': stride_length,
            'knee_drive': avg_knee_drive
        }

    def _analyze_running_posture(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze running posture"""
        head = pose['head']
        shoulders = pose['shoulders']
        hips = pose['hips']
        
        # Calculate forward lean
        avg_shoulder_x = (shoulders['left'][0] + shoulders['right'][0]) / 2
        avg_hip_x = (hips['left'][0] + hips['right'][0]) / 2
        forward_lean = abs(avg_shoulder_x - avg_hip_x)
        
        # Calculate head position
        head_forward = abs(head[0] - avg_shoulder_x)
        
        # Calculate shoulder level
        shoulder_level = abs(shoulders['left'][1] - shoulders['right'][1])
        
        return {
            'forward_lean': forward_lean,
            'head_position': head_forward,
            'shoulder_level': shoulder_level
        }

    def _analyze_arm_swing(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze arm swing mechanics"""
        left_wrist = pose['wrists']['left']
        right_wrist = pose['wrists']['right']
        shoulders = pose['shoulders']
        
        # Calculate arm swing range
        arm_swing_range = abs(left_wrist[0] - right_wrist[0])
        
        # Calculate arm swing height
        avg_shoulder_y = (shoulders['left'][1] + shoulders['right'][1]) / 2
        left_arm_height = abs(left_wrist[1] - avg_shoulder_y)
        right_arm_height = abs(right_wrist[1] - avg_shoulder_y)
        avg_arm_height = (left_arm_height + right_arm_height) / 2
        
        return {
            'swing_range': arm_swing_range,
            'swing_height': avg_arm_height
        }

    def _analyze_foot_strike(self, pose: Dict[str, Any]) -> Dict[str, float]:
        """Analyze foot strike pattern"""
        left_ankle = pose['ankles']['left']
        right_ankle = pose['ankles']['right']
        left_knee = pose['knees']['left']
        right_knee = pose['knees']['right']
        
        # Calculate foot strike angle (simplified)
        left_leg_angle = np.arctan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0])
        right_leg_angle = np.arctan2(right_ankle[1] - right_knee[1], right_ankle[0] - right_knee[0])
        
        return {
            'left_strike_angle': abs(np.degrees(left_leg_angle)),
            'right_strike_angle': abs(np.degrees(right_leg_angle))
        }

    def _calculate_running_efficiency(self, pose: Dict[str, Any]) -> float:
        """Calculate overall running efficiency score"""
        posture = self._analyze_running_posture(pose)
        arm_swing = self._analyze_arm_swing(pose)
        
        # Score based on optimal running form
        posture_score = 1.0 - min(posture['forward_lean'], 0.1) / 0.1
        arm_score = min(arm_swing['swing_range'], 0.3) / 0.3
        balance_score = 1.0 - min(posture['shoulder_level'], 0.05) / 0.05
        
        return (posture_score + arm_score + balance_score) / 3

    def _classify_intensity(self, velocity: float) -> str:
        """Classify workout intensity based on velocity"""
        if velocity > 0.1:
            return "high"
        elif velocity > 0.05:
            return "moderate"
        else:
            return "low"

    def _estimate_heart_rate(self, intensity: str) -> int:
        """Estimate heart rate based on intensity"""
        intensity_mapping = {
            "low": 100,
            "moderate": 140,
            "high": 170
        }
        return intensity_mapping.get(intensity, 120)

    def _detect_movement_type(self, pose: Dict[str, Any]) -> str:
        """Detect specific movement/exercise type"""
        # Check for squatting motion
        if self._is_squatting(pose):
            return "squat"
        
        # Check for jumping motion
        if self._is_jumping(pose):
            return "jump"
        
        # Check for lunging motion
        if self._is_lunging(pose):
            return "lunge"
        
        # Check for stretching
        if self._is_stretching(pose):
            return "stretch"
        
        return None

    def _is_squatting(self, pose: Dict[str, Any]) -> bool:
        """Detect squatting motion"""
        knees = pose['knees']
        hips = pose['hips']
        ankles = pose['ankles']
        
        # Check if knees are bent and hips are low
        left_knee_angle = self._calculate_knee_angle(hips['left'], knees['left'], ankles['left'])
        right_knee_angle = self._calculate_knee_angle(hips['right'], knees['right'], ankles['right'])
        
        return left_knee_angle < 120 and right_knee_angle < 120

    def _is_jumping(self, pose: Dict[str, Any]) -> bool:
        """Detect jumping motion"""
        ankles = pose['ankles']
        
        # Check if both feet are off ground (high ankle position)
        avg_ankle_y = (ankles['left'][1] + ankles['right'][1]) / 2
        
        return avg_ankle_y < 0.7  # High ankle position indicates jumping

    def _is_lunging(self, pose: Dict[str, Any]) -> bool:
        """Detect lunging motion"""
        knees = pose['knees']
        ankles = pose['ankles']
        
        # Check for asymmetric leg position
        left_knee_y = knees['left'][1]
        right_knee_y = knees['right'][1]
        
        knee_height_diff = abs(left_knee_y - right_knee_y)
        
        return knee_height_diff > 0.1

    def _is_stretching(self, pose: Dict[str, Any]) -> bool:
        """Detect stretching motion"""
        wrists = pose['wrists']
        head = pose['head']
        
        # Check if arms are extended (reaching)
        left_wrist_high = wrists['left'][1] < head[1]
        right_wrist_high = wrists['right'][1] < head[1]
        
        return left_wrist_high or right_wrist_high

    def _calculate_knee_angle(self, hip: Tuple[float, float], knee: Tuple[float, float], ankle: Tuple[float, float]) -> float:
        """Calculate knee angle"""
        # Calculate vectors
        v1 = np.array([hip[0] - knee[0], hip[1] - knee[1]])
        v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def _evaluate_movement_form(self, pose: Dict[str, Any], movement_type: str) -> float:
        """Evaluate form quality for specific movement"""
        if movement_type == "squat":
            return self._evaluate_squat_form(pose)
        elif movement_type == "jump":
            return self._evaluate_jump_form(pose)
        elif movement_type == "lunge":
            return self._evaluate_lunge_form(pose)
        elif movement_type == "stretch":
            return self._evaluate_stretch_form(pose)
        
        return 0.5

    def _evaluate_squat_form(self, pose: Dict[str, Any]) -> float:
        """Evaluate squat form quality"""
        # Check knee alignment
        knees = pose['knees']
        ankles = pose['ankles']
        
        left_knee_alignment = abs(knees['left'][0] - ankles['left'][0])
        right_knee_alignment = abs(knees['right'][0] - ankles['right'][0])
        
        alignment_score = 1.0 - min((left_knee_alignment + right_knee_alignment) / 2, 0.1) / 0.1
        
        # Check posture
        shoulders = pose['shoulders']
        shoulder_level = abs(shoulders['left'][1] - shoulders['right'][1])
        posture_score = 1.0 - min(shoulder_level, 0.05) / 0.05
        
        return (alignment_score + posture_score) / 2

    def _evaluate_jump_form(self, pose: Dict[str, Any]) -> float:
        """Evaluate jump form quality"""
        # Check body alignment during jump
        center_of_mass = pose['center_of_mass']
        head = pose['head']
        
        body_alignment = abs(head[0] - center_of_mass[0])
        alignment_score = 1.0 - min(body_alignment, 0.05) / 0.05
        
        return alignment_score

    def _evaluate_lunge_form(self, pose: Dict[str, Any]) -> float:
        """Evaluate lunge form quality"""
        # Check knee alignment and posture
        knees = pose['knees']
        ankles = pose['ankles']
        
        # Front knee should be over ankle
        front_knee_alignment = min(abs(knees['left'][0] - ankles['left'][0]), 
                                  abs(knees['right'][0] - ankles['right'][0]))
        
        alignment_score = 1.0 - min(front_knee_alignment, 0.05) / 0.05
        
        return alignment_score

    def _evaluate_stretch_form(self, pose: Dict[str, Any]) -> float:
        """Evaluate stretching form quality"""
        # Check if stretch is controlled and balanced
        shoulders = pose['shoulders']
        shoulder_level = abs(shoulders['left'][1] - shoulders['right'][1])
        
        balance_score = 1.0 - min(shoulder_level, 0.05) / 0.05
        
        return balance_score

    def _calculate_fitness_metrics(self, performance_data: List[Dict], position_history: List) -> Dict[str, Any]:
        """Calculate fitness-specific metrics"""
        running_data = [p for p in performance_data if p['type'] == 'running_form']
        cardio_data = [p for p in performance_data if p['type'] == 'cardio_intensity']
        
        metrics = {
            'total_distance': 0,
            'avg_intensity': 'moderate',
            'avg_running_efficiency': 0,
            'workout_duration': 0,
            'calories_estimate': 0,
            'movement_variety': 0
        }
        
        # Calculate workout duration
        if position_history:
            metrics['workout_duration'] = position_history[-1][0] - position_history[0][0]
        
        # Calculate average intensity
        if cardio_data:
            intensities = [c['intensity'] for c in cardio_data]
            intensity_counts = {i: intensities.count(i) for i in set(intensities)}
            metrics['avg_intensity'] = max(intensity_counts, key=intensity_counts.get)
        
        # Calculate running efficiency
        if running_data:
            efficiency_scores = [r['efficiency_score'] for r in running_data]
            metrics['avg_running_efficiency'] = np.mean(efficiency_scores)
        
        # Calculate total distance (simplified)
        if position_history:
            total_movement = 0
            for i in range(1, len(position_history)):
                prev_pos = position_history[i-1][1]['pose_data']['center_of_mass']
                curr_pos = position_history[i][1]['pose_data']['center_of_mass']
                movement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                total_movement += movement
            
            metrics['total_distance'] = total_movement
        
        # Estimate calories burned
        duration_minutes = metrics['workout_duration'] / 60
        intensity_multiplier = {'low': 5, 'moderate': 8, 'high': 12}
        metrics['calories_estimate'] = int(duration_minutes * intensity_multiplier.get(metrics['avg_intensity'], 8))
        
        return metrics

    def _generate_fitness_recommendations(self, metrics: Dict[str, Any], gemini_analysis: Dict[str, Any]) -> List[str]:
        """Generate fitness-specific recommendations"""
        recommendations = []
        
        # Running form recommendations
        if metrics['avg_running_efficiency'] < 0.6:
            recommendations.append("Focus on running form - work on posture and arm swing")
            recommendations.append("Practice proper foot strike and cadence")
        
        # Intensity recommendations
        if metrics['avg_intensity'] == 'low':
            recommendations.append("Increase workout intensity for better cardiovascular benefits")
            recommendations.append("Add high-intensity intervals to your routine")
        elif metrics['avg_intensity'] == 'high':
            recommendations.append("Consider adding recovery periods to prevent overtraining")
        
        # Duration recommendations
        if metrics['workout_duration'] < 1200:  # Less than 20 minutes
            recommendations.append("Aim for longer workout sessions (30+ minutes)")
        
        # Movement variety
        recommendations.append("Add variety to your workouts with different movement patterns")
        recommendations.append("Include both cardio and strength exercises")
        
        # Add Gemini insights
        if gemini_analysis and 'suggestions' in gemini_analysis:
            recommendations.append(f"AI Analysis: {gemini_analysis['suggestions']}")
        
        return recommendations

    def create_processed_video(self, video_path: str, analysis_data: Dict[str, Any], output_dir: str) -> str:
        """Create processed video with fitness analysis overlays"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f"fitness_analysis_{int(time.time())}.mp4"
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
            
            frame = self._add_fitness_overlays(frame, timestamp, key_moments, analysis_data)
            out.write(frame)
        
        cap.release()
        out.release()
        
        return output_path

    def _add_fitness_overlays(self, frame: np.ndarray, timestamp: float, key_moments: List[Dict], analysis_data: Dict[str, Any]) -> np.ndarray:
        """Add fitness-specific overlays"""
        metrics = analysis_data.get('technical_metrics', {})
        
        # Overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "FITNESS ANALYSIS", (20, 35), font, 0.7, (255, 215, 0), 2)
        cv2.putText(frame, f"Duration: {metrics.get('workout_duration', 0):.0f}s", (20, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Intensity: {metrics.get('avg_intensity', 'N/A').upper()}", (20, 80), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Efficiency: {metrics.get('avg_running_efficiency', 0):.1f}/1.0", (20, 100), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Calories: ~{metrics.get('calories_estimate', 0)}", (20, 120), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Distance: {metrics.get('total_distance', 0):.1f}m", (20, 140), font, 0.5, (255, 255, 255), 1)
        
        # Highlight key moments
        for moment in key_moments:
            if abs(moment['timestamp'] - timestamp) < 0.5:
                if moment['type'] == 'movement_pattern':
                    movement_type = moment.get('movement_type', 'EXERCISE')
                    cv2.putText(frame, f"{movement_type.upper()}!", (20, 170), font, 0.6, (0, 255, 0), 2)
        
        return frame
