import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
import os

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수: 세 점(A, B, C)의 각도(∠ABC)를 계산
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# ─────────────────────────────────────────────────────────────────────────────
# 랜드마크 신뢰도 검증 함수
# ─────────────────────────────────────────────────────────────────────────────
def validate_landmarks(lm, required_landmarks: List[int], min_confidence: float = 0.7) -> bool:
    """
    필요한 랜드마크들이 모두 충분한 신뢰도로 검출되었는지 확인
    """
    for landmark_idx in required_landmarks:
        if lm[landmark_idx].visibility < min_confidence:
            return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 고급 필터링 시스템 (논문 기반)
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedFilteringSystem:
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size-1)
        
    def update(self, value: float) -> Dict[str, float]:
        """값 업데이트 및 속도 계산"""
        self.values.append(value)
        
        # 이동평균 계산
        smoothed_value = np.mean(self.values) if len(self.values) > 0 else value
        
        # 속도 계산 (이전 값과의 차이)
        if len(self.values) >= 2:
            velocity = self.values[-1] - self.values[-2]
            self.velocities.append(velocity)
            avg_velocity = np.mean(self.velocities) if len(self.velocities) > 0 else 0
        else:
            avg_velocity = 0
            
        return {
            'smoothed': smoothed_value,
            'velocity': avg_velocity,
            'is_ready': len(self.values) >= self.window_size
        }

# ─────────────────────────────────────────────────────────────────────────────
# 스쿼트 단계 감지 클래스 (논문 기반)
# ─────────────────────────────────────────────────────────────────────────────
class SquatPhaseDetector:
    def __init__(self):
        self.phases = ['preparation', 'descent', 'bottom', 'ascent', 'completion']
        self.current_phase = 'preparation'
        self.phase_history = []
        self.bottom_detected = False
        self.velocity_threshold = 2.0  # 각도 변화 속도 임계값
        
        # 사이클 완료 감지를 위한 변수들
        self.cycle_completed = False
        self.best_score_in_cycle = 0.0
        self.best_grade_in_cycle = 'F'
        self.cycle_start_frame = None
        self.should_display_score = False  # 점수 표시 여부
        
        # 각 사이클별 최고 점수 기록
        self.cycle_scores = []  # [(cycle_num, score, grade), ...]
        self.current_cycle_num = 0
        self.current_cycle_best_score = 0.0
        self.current_cycle_best_grade = 'F'
        
    def detect_phase(self, knee_angle: float, velocity: float, frame_count: int) -> str:
        """스쿼트 단계 감지"""
        phase_info = {
            'frame': frame_count,
            'knee_angle': knee_angle,
            'velocity': velocity,
            'phase': self.current_phase
        }
        
        # 이전 단계 저장 (사이클 완료 감지용)
        previous_phase = self.current_phase
        
        # 새로운 사이클 시작 감지
        # 1. preparation에서 descent로 전환 (첫 번째 사이클)
        # 2. completion에서 descent로 전환 (두 번째 사이클 이후)
        if ((self.current_phase == 'preparation' and velocity < -self.velocity_threshold) or
            (self.current_phase == 'completion' and velocity < -self.velocity_threshold)):
            
            # 이전 사이클이 완료된 경우, 해당 사이클의 최고 점수 저장
            if self.current_phase == 'completion' and self.current_cycle_best_score > 0:
                self.cycle_scores.append((
                    self.current_cycle_num,
                    self.current_cycle_best_score,
                    self.current_cycle_best_grade
                ))
                self.cycle_completed = True
                self.should_display_score = True
            
            # 새로운 사이클 시작
            self.current_phase = 'descent'
            self.current_cycle_num += 1
            self.cycle_start_frame = frame_count
            
            # 현재 사이클 점수 초기화
            self.current_cycle_best_score = 0.0
            self.current_cycle_best_grade = 'F'
            
            # 첫 번째 사이클이 아닌 경우에만 표시 활성화
            if previous_phase == 'completion':
                pass  # 이전 사이클 점수는 유지
            else:
                # 첫 번째 사이클 시작 시 표시 비활성화
                self.should_display_score = False
        
        # 속도 기반 단계 감지
        elif abs(velocity) < self.velocity_threshold:
            if self.current_phase == 'descent' and knee_angle < 100:
                self.current_phase = 'bottom'
                self.bottom_detected = True
            elif self.current_phase == 'ascent' and knee_angle > 150:
                self.current_phase = 'completion'
        elif velocity < -self.velocity_threshold:
            if self.current_phase == 'preparation':
                self.current_phase = 'descent'
        elif velocity > self.velocity_threshold:
            if self.current_phase == 'bottom':
                self.current_phase = 'ascent'
        
        phase_info['phase'] = self.current_phase
        self.phase_history.append(phase_info)
        
        return self.current_phase
    
    def update_best_score(self, score: float, grade: str):
        """현재 사이클 내 최고 점수 업데이트 (descent -> bottom 구간에서만)"""
        # descent -> bottom 구간에서만 점수 업데이트
        if self.current_phase in ['descent', 'bottom']:
            if score > self.current_cycle_best_score:
                self.current_cycle_best_score = score
                self.current_cycle_best_grade = grade
    
    def get_display_score_and_grade(self) -> tuple:
        """표시할 점수와 등급 반환 (가장 최근 완료된 사이클의 점수)"""
        if self.should_display_score and self.cycle_completed and len(self.cycle_scores) > 0:
            # 가장 최근 완료된 사이클의 점수 반환
            latest_cycle = self.cycle_scores[-1]
            return latest_cycle[1], latest_cycle[2]  # score, grade
        else:
            return None, None
    
    def get_all_cycle_scores(self) -> list:
        """모든 사이클의 점수 반환"""
        return self.cycle_scores.copy()
    
    def get_current_cycle_info(self) -> tuple:
        """현재 사이클 정보 반환"""
        return self.current_cycle_num, self.current_cycle_best_score, self.current_cycle_best_grade
    
    def get_bottom_frame(self) -> Optional[int]:
        """하단점 프레임 찾기"""
        for info in self.phase_history:
            if info['phase'] == 'bottom':
                return info['frame']
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 생체역학적 검증 클래스 (논문 기반)
# ─────────────────────────────────────────────────────────────────────────────
class BiomechanicalValidator:
    def __init__(self):
        self.biomechanical_rules = {
            'knee_tracking': self.validate_knee_tracking,
            'hip_hinge': self.validate_hip_hinge,
            'torso_stability': self.validate_torso_stability,
            'ankle_mobility': self.validate_ankle_mobility
        }
    
    def validate_knee_tracking(self, knee_pos: Tuple[int, int], ankle_pos: Tuple[int, int]) -> float:
        """무릎이 발끝을 넘지 않는지 검증"""
        # 무릎과 발목의 수직 정렬 검사
        knee_x, knee_y = knee_pos
        ankle_x, ankle_y = ankle_pos
        
        # 무릎이 발목보다 앞으로 나가지 않도록
        if knee_x > ankle_x:
            return 0.0  # 무릎이 발끝을 넘음
        else:
            return 1.0  # 정상
    
    def validate_hip_hinge(self, hip_angle: float, torso_angle: float) -> float:
        """엉덩이 힌지 동작 검증 - 더 관대한 기준"""
        # 엉덩이와 상체의 상관관계 분석
        # 엉덩이 각도가 적절한 범위에 있어야 함
        if 25 <= hip_angle <= 75:  # 30-60에서 25-75로 확대
            return 1.0
        elif 15 <= hip_angle <= 85:  # 20-70에서 15-85로 확대
            return 0.8  # 0.7에서 0.8로 증가
        else:
            return 0.5  # 0.3에서 0.5로 증가
    
    def validate_torso_stability(self, torso_angle: float) -> float:
        """상체 안정성 검증 - 더 관대한 기준"""
        # 상체가 너무 기울어지지 않도록
        if 25 <= torso_angle <= 75:  # 30-60에서 25-75로 확대
            return 1.0
        elif 15 <= torso_angle <= 85:  # 20-70에서 15-85로 확대
            return 0.8  # 0.7에서 0.8로 증가
        else:
            return 0.5  # 0.3에서 0.5로 증가
    
    def validate_ankle_mobility(self, ankle_angle: float) -> float:
        """발목 가동성 검증 - 더 관대한 기준"""
        # 발목 각도가 적절한 범위에 있어야 함
        if 50 <= ankle_angle <= 90:  # 60-80에서 50-90으로 확대
            return 1.0
        elif 40 <= ankle_angle <= 100:  # 50-90에서 40-100으로 확대
            return 0.8  # 0.7에서 0.8로 증가
        else:
            return 0.5  # 0.3에서 0.5로 증가
    
    def calculate_biomechanical_score(self, angles_data: Dict, positions: Dict) -> Dict[str, float]:
        """생체역학적 점수 계산"""
        scores = {}
        
        # 각 생체역학적 규칙 적용
        scores['knee_tracking'] = self.validate_knee_tracking(
            positions.get('knee', (0, 0)), 
            positions.get('ankle', (0, 0))
        )
        scores['hip_hinge'] = self.validate_hip_hinge(
            angles_data.get('hip', 0), 
            angles_data.get('torso', 0)
        )
        scores['torso_stability'] = self.validate_torso_stability(
            angles_data.get('torso', 0)
        )
        scores['ankle_mobility'] = self.validate_ankle_mobility(
            angles_data.get('ankle', 0)
        )
        
        # 전체 생체역학적 점수
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores

# ─────────────────────────────────────────────────────────────────────────────
# 스쿼트 자세 검증 클래스 (개선된 버전)
# ─────────────────────────────────────────────────────────────────────────────
class SquatPoseValidator:
    def __init__(self):
        # 기준 각도 (동적으로 조정 가능) - 더 현실적인 값으로 조정
        self.reference_angles = {
            'knee': 90.0,    # 75도에서 90도로 조정 (더 현실적인 스쿼트 각도)
            'hip': 45.0,
            'ankle': 70.0,
            'torso': 45.0  # 상체 기울기
        }
        
        # 각도 허용 범위 - 더 관대하게 조정
        self.angle_tolerance = {
            'knee': 30.0,    # 20도에서 30도로 조정
            'hip': 30.0,     # 20도에서 30도로 조정
            'ankle': 35.0,   # 25도에서 35도로 조정
            'torso': 35.0    # 25도에서 35도로 조정
        }
    
    def calculate_dynamic_threshold(self, user_height: float, squat_depth: str = 'medium') -> dict:
        """
        사용자 키와 스쿼트 깊이에 따른 동적 임계값 계산
        """
        # 키에 따른 기본 조정 (예시)
        height_factor = user_height / 170.0  # 170cm 기준
        
        # 스쿼트 깊이에 따른 조정
        depth_factors = {
            'shallow': 1.1,  # 더 큰 각도 (얕은 스쿼트)
            'medium': 1.0,   # 기본 각도
            'deep': 0.85     # 더 작은 각도 (깊은 스쿼트) - 0.9에서 0.85로 조정
        }
        
        factor = height_factor * depth_factors.get(squat_depth, 1.0)
        
        return {
            'knee': self.reference_angles['knee'] * factor,
            'hip': self.reference_angles['hip'] * factor,
            'ankle': self.reference_angles['ankle'] * factor,
            'torso': self.reference_angles['torso'] * factor
        }
    
    def validate_squat_pose(self, angles: dict, thresholds: dict) -> dict:
        """
        스쿼트 자세의 각 관절별 유효성 검증
        """
        results = {}
        
        for joint, angle in angles.items():
            if joint in thresholds:
                threshold = thresholds[joint]
                tolerance = self.angle_tolerance[joint]
                
                # 각도가 허용 범위 내에 있는지 확인
                is_valid = abs(angle - threshold) <= tolerance
                deviation = abs(angle - threshold)
                
                results[joint] = {
                    'angle': angle,
                    'threshold': threshold,
                    'deviation': deviation,
                    'is_valid': is_valid,
                    'score': max(0, 1 - (deviation / tolerance))
                }
        
        return results

# ─────────────────────────────────────────────────────────────────────────────
# 코사인 유사도 계산 함수들
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity_scalar(angle1: float, angle2: float) -> float:
    diff_rad = (angle1 - angle2) * np.pi / 180.0
    return np.cos(diff_rad)

def assign_grade_from_similarity(sim: float) -> Optional[str]:
    # 더 현실적인 기준으로 조정
    if sim >= 0.95:      # 0.98에서 0.95로 조정
        return "A"
    elif sim >= 0.90:    # 0.95에서 0.90으로 조정
        return "B"
    elif sim >= 0.80:    # 0.90에서 0.80으로 조정
        return "C"
    elif sim >= 0.70:    # 0.80에서 0.70으로 조정
        return "D"
    else:
        return "F"

# ─────────────────────────────────────────────────────────────────────────────
# 고급 스쿼트 등급 평가 함수 (논문 기반) - 더 후한 평가로 개선
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_advanced_squat_grade(
    knee_angle: float, 
    hip_angle: float, 
    threshold_knee: float, 
    threshold_hip: float,
    biomechanical_score: float,
    phase: str
) -> str:
    """
    무릎, 엉덩이 각도, 생체역학적 점수, 스쿼트 단계를 종합하여 등급 평가
    """
    # 기본 각도 점수 계산 - 더 관대한 허용 범위
    knee_diff = abs(knee_angle - threshold_knee)
    hip_diff = abs(hip_angle - threshold_hip)
    
    # 허용 범위를 30도에서 45도로 확대
    knee_score = max(0, 1 - (knee_diff / 45.0))
    hip_score = max(0, 1 - (hip_diff / 45.0))
    
    # 단계별 가중치 적용 - 더 관대하게 조정
    phase_weights = {
        'preparation': 0.6,  # 0.3에서 0.6으로 증가
        'descent': 0.8,      # 0.7에서 0.8로 증가
        'bottom': 1.0,       # 하단점에서 가장 중요 (유지)
        'ascent': 0.9,       # 0.8에서 0.9로 증가
        'completion': 0.7    # 0.5에서 0.7로 증가
    }
    
    phase_weight = phase_weights.get(phase, 0.7)  # 기본값도 0.5에서 0.7로 증가
    
    # 종합 점수 계산 - 생체역학적 점수 비중 감소
    angle_score = (knee_score * 0.7) + (hip_score * 0.3)
    biomechanical_weight = 0.2  # 0.3에서 0.2로 감소
    angle_weight = 0.8          # 0.7에서 0.8로 증가
    
    total_score = (angle_score * angle_weight + biomechanical_score * biomechanical_weight) * phase_weight
    
    # 등급 할당 - 더욱 관대한 기준으로 조정
    if phase == 'bottom':
        if total_score >= 0.60:  # 0.75에서 0.60으로 감소
            return "A"
        elif total_score >= 0.50:  # 0.65에서 0.50으로 감소
            return "B"
        elif total_score >= 0.40:  # 0.55에서 0.40으로 감소
            return "C"
        elif total_score >= 0.30:  # 0.45에서 0.30으로 감소
            return "D"
        else:
            return "F"
    else:
        if total_score >= 0.55:  # 0.70에서 0.55로 감소
            return "A"
        elif total_score >= 0.45:  # 0.60에서 0.45로 감소
            return "B"
        elif total_score >= 0.35:  # 0.50에서 0.35로 감소
            return "C"
        elif total_score >= 0.25:  # 0.40에서 0.25로 감소
            return "D"
        else:
            return "F"

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose 초기화 함수
# ─────────────────────────────────────────────────────────────────────────────
def initialize_mediapipe_pose():
    """MediaPipe Pose 객체를 초기화하고 반환"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 더 정확한 모델 사용
        enable_segmentation=False,
        min_detection_confidence=0.7,  # 더 높은 검출 신뢰도
        min_tracking_confidence=0.7,   # 추적 신뢰도 추가
        smooth_landmarks=True
    )
    return mp_pose, pose

# ─────────────────────────────────────────────────────────────────────────────
# 영상 정보 가져오기 함수
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(input_path: str):
    """영상 파일의 정보를 가져오는 함수"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"비디오를 열 수 없습니다: {input_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    
    cap.release()
    
    return {
        'frame_count': frame_count,
        'frame_width': frame_width,
        'frame_height': frame_height,
        'fps': fps
    }

# ─────────────────────────────────────────────────────────────────────────────
# 필터 및 검증기 초기화 함수
# ─────────────────────────────────────────────────────────────────────────────
def initialize_filters_and_validator(user_height: float = 170.0, squat_depth: str = 'medium'):
    """필터와 검증기를 초기화하는 함수"""
    # 고급 필터 초기화
    left_knee_filter = AdvancedFilteringSystem(window_size=7)
    right_knee_filter = AdvancedFilteringSystem(window_size=7)
    left_hip_filter = AdvancedFilteringSystem(window_size=7)
    right_hip_filter = AdvancedFilteringSystem(window_size=7)
    
    # 검증기 초기화
    pose_validator = SquatPoseValidator()
    dynamic_thresholds = pose_validator.calculate_dynamic_threshold(user_height, squat_depth)
    
    # 스쿼트 단계 감지기 초기화
    phase_detector = SquatPhaseDetector()
    
    # 생체역학적 검증기 초기화
    biomechanical_validator = BiomechanicalValidator()
    
    return {
        'left_knee_filter': left_knee_filter,
        'right_knee_filter': right_knee_filter,
        'left_hip_filter': left_hip_filter,
        'right_hip_filter': right_hip_filter,
        'pose_validator': pose_validator,
        'dynamic_thresholds': dynamic_thresholds,
        'phase_detector': phase_detector,
        'biomechanical_validator': biomechanical_validator
    }

# ─────────────────────────────────────────────────────────────────────────────
# 필요한 랜드마크 인덱스 정의 함수
# ─────────────────────────────────────────────────────────────────────────────
def get_required_landmarks(mp_pose):
    """필요한 랜드마크 인덱스들을 반환하는 함수"""
    return [
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 메인 스쿼트 분석 함수
# ─────────────────────────────────────────────────────────────────────────────
def analyze_squat_video(
    input_path: str,
    output_path: str,
    user_height: float = 170.0,
    squat_depth: str = 'medium',
    display_duration_seconds: int = 3
):
    """
    스쿼트 영상을 분석하는 메인 함수
    
    Args:
        input_path: 입력 영상 파일 경로
        output_path: 출력 영상 파일 경로
        user_height: 사용자 키 (cm)
        squat_depth: 스쿼트 깊이 ('shallow', 'medium', 'deep')
        display_duration_seconds: 결과 표시 지속 시간 (초)
    """
    
    # 1. MediaPipe Pose 초기화
    mp_pose, pose = initialize_mediapipe_pose()
    
    # 2. 영상 정보 가져오기
    video_info = get_video_info(input_path)
    frame_count = video_info['frame_count']
    frame_width = video_info['frame_width']
    frame_height = video_info['frame_height']
    fps = video_info['fps']
    
    display_duration = int(fps * display_duration_seconds)
    
    # 3. 필터 및 검증기 초기화
    filters_and_validator = initialize_filters_and_validator(user_height, squat_depth)
    left_knee_filter = filters_and_validator['left_knee_filter']
    right_knee_filter = filters_and_validator['right_knee_filter']
    left_hip_filter = filters_and_validator['left_hip_filter']
    right_hip_filter = filters_and_validator['right_hip_filter']
    pose_validator = filters_and_validator['pose_validator']
    dynamic_thresholds = filters_and_validator['dynamic_thresholds']
    phase_detector = filters_and_validator['phase_detector']
    biomechanical_validator = filters_and_validator['biomechanical_validator']
    
    # 4. 상태 변수들 초기화
    display_countdown = 0
    squat_data = {
        "left_knee": None,
        "right_knee": None,
        "left_hip": None,
        "right_hip": None,
        "avg_knee": None,
        "avg_hip": None,
        "overall_grade": None,
        "pose_validation": None,
        "overall_score": None
    }
    
    # 5. 필요한 랜드마크 인덱스들
    required_landmarks = get_required_landmarks(mp_pose)
    
    # 6. 영상 읽기/쓰기 준비
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 7. 메인 처리 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark

            # 랜드마크 신뢰도 검증
            if validate_landmarks(lm, required_landmarks, min_confidence=0.7):
                def get_point(landmark):
                    return (int(landmark.x * w), int(landmark.y * h))

                # 주요 관절 좌표 추출
                left_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
                right_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                left_knee = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                right_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                left_ankle = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                right_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                left_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
                right_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])

                # 각도 계산
                angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
                angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
                angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
                angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)

                # 필터 적용 (고급 필터링 시스템 사용)
                left_knee_result = left_knee_filter.update(angle_left_knee)
                right_knee_result = right_knee_filter.update(angle_right_knee)
                left_hip_result = left_hip_filter.update(angle_left_hip)
                right_hip_result = right_hip_filter.update(angle_right_hip)

                # 스쿼트 하단점 감지 (필터가 준비된 후)
                if (left_knee_result['is_ready'] and right_knee_result['is_ready'] and
                    left_hip_result['is_ready'] and right_hip_result['is_ready']):
                    
                    # 평균 각도 계산 (좌우 균형 고려)
                    avg_knee_angle = (left_knee_result['smoothed'] + right_knee_result['smoothed']) / 2
                    avg_hip_angle = (left_hip_result['smoothed'] + right_hip_result['smoothed']) / 2
                    
                    # 스쿼트 단계 감지
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    avg_velocity = (left_knee_result['velocity'] + right_knee_result['velocity']) / 2
                    current_phase = phase_detector.detect_phase(avg_knee_angle, avg_velocity, current_frame)
                    
                    # 발목과 상체 각도 계산
                    angle_left_ankle = calculate_angle(left_knee, left_ankle, (left_ankle[0], left_ankle[1] - 50))
                    angle_right_ankle = calculate_angle(right_knee, right_ankle, (right_ankle[0], right_ankle[1] - 50))
                    angle_left_torso = calculate_angle(left_hip, left_shoulder, (left_shoulder[0], left_shoulder[1] - 50))
                    angle_right_torso = calculate_angle(right_hip, right_shoulder, (right_shoulder[0], right_shoulder[1] - 50))
                    
                    avg_ankle_angle = (angle_left_ankle + angle_right_ankle) / 2
                    avg_torso_angle = (angle_left_torso + angle_right_torso) / 2
                    
                    # 스쿼트 자세 검증
                    current_angles = {
                        'knee': avg_knee_angle,
                        'hip': avg_hip_angle,
                        'ankle': avg_ankle_angle,
                        'torso': avg_torso_angle
                    }
                    
                    pose_validation = pose_validator.validate_squat_pose(current_angles, dynamic_thresholds)
                    
                    # 생체역학적 검증
                    positions = {
                        'knee': left_knee,  # 왼쪽 무릎 위치 사용
                        'ankle': left_ankle  # 왼쪽 발목 위치 사용
                    }
                    biomechanical_scores = biomechanical_validator.calculate_biomechanical_score(
                        current_angles, positions
                    )
                    
                    # 고급 등급 평가
                    grade = evaluate_advanced_squat_grade(
                        avg_knee_angle, avg_hip_angle, 
                        dynamic_thresholds['knee'], dynamic_thresholds['hip'],
                        biomechanical_scores['overall'], current_phase
                    )
                    
                    # 전체 점수 계산 (개선된 버전)
                    valid_scores = [v['score'] for v in pose_validation.values() if v['is_valid']]
                    pose_score = np.mean(valid_scores) if valid_scores else 0.0
                    overall_score = (pose_score * 0.6) + (biomechanical_scores['overall'] * 0.4)
                    
                    # 사이클 내 최고 점수 업데이트 (descent -> bottom 구간에서만)
                    phase_detector.update_best_score(overall_score, grade)
                    
                    # 표시할 점수와 등급 가져오기
                    display_score, display_grade = phase_detector.get_display_score_and_grade()
                    
                    # 스쿼트 데이터 업데이트
                    squat_data = {
                        "left_knee": left_knee_result['smoothed'],
                        "right_knee": right_knee_result['smoothed'],
                        "left_hip": left_hip_result['smoothed'],
                        "right_hip": right_hip_result['smoothed'],
                        "avg_knee": avg_knee_angle,
                        "avg_hip": avg_hip_angle,
                        "avg_ankle": avg_ankle_angle,
                        "avg_torso": avg_torso_angle,
                        "current_phase": current_phase,
                        "overall_grade": grade,
                        "pose_validation": pose_validation,
                        "biomechanical_scores": biomechanical_scores,
                        "overall_score": overall_score,
                        "display_score": display_score,
                        "display_grade": display_grade
                    }
                    
                    # 디버깅용 - 항상 카운트다운 시작
                    display_countdown = display_duration

        # ─────────────────────────────────────────────────────────────────────────
        # 결과 표시 (디버깅용 - 모든 정보 표시)
        # ─────────────────────────────────────────────────────────────────────────
        if display_countdown > 0:
            # 스켈레톤 그리기
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )

            # 상세 정보 표시 (논문 기반 개선된 버전)
            phase_color = {
                'preparation': (255, 255, 0),  # 노란색
                'descent': (0, 255, 255),      # 청록색
                'bottom': (0, 255, 0),         # 초록색
                'ascent': (255, 165, 0),       # 주황색
                'completion': (255, 255, 255)  # 흰색
            }
            
            current_phase = squat_data.get('current_phase', 'preparation')
            phase_col = phase_color.get(current_phase, (255, 255, 255))
            
            # 표시할 점수와 등급
            display_score = squat_data.get('display_score')
            display_grade = squat_data.get('display_grade')
            current_score = squat_data.get('overall_score', 0)
            current_grade = squat_data.get('overall_grade', 'F')
            
            # 사이클 상태 정보
            cycle_status = "Completed" if phase_detector.cycle_completed else "In Progress"
            should_display = "Yes" if phase_detector.should_display_score else "No"
            
            # 현재 사이클 정보
            current_cycle_num, current_cycle_best_score, current_cycle_best_grade = phase_detector.get_current_cycle_info()
            
            # None 값 처리
            display_score_str = f"{display_score:.2f}" if display_score is not None else "N/A"
            display_grade_str = display_grade if display_grade is not None else "N/A"
            
            # 모든 사이클 점수 정보
            all_scores = phase_detector.get_all_cycle_scores()
            scores_summary = ""
            if all_scores:
                scores_summary = " ".join([f"C{i+1}:{grade}({score:.1f})" for i, (_, score, grade) in enumerate(all_scores)])
            
            overlay_text = [
                f"Phase: {current_phase.upper()}",
                f"L Knee: {squat_data['left_knee']:.1f}°  R Knee: {squat_data['right_knee']:.1f}°",
                f"L Hip: {squat_data['left_hip']:.1f}°  R Hip: {squat_data['right_hip']:.1f}°",
                f"Ankle: {squat_data.get('avg_ankle', 0):.1f}°  Torso: {squat_data.get('avg_torso', 0):.1f}°",
                f"Best: {display_grade_str} ({display_score_str})",
                f"All Cycles: {scores_summary}" if scores_summary else "No completed cycles"
            ]

            # 텍스트 표시
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # 등급별 색상 정의
            def get_grade_color(grade):
                if grade == 'A':
                    return (0, 255, 0)    # 연두색
                elif grade == 'B':
                    return (0, 128, 0)    # 초록색
                elif grade == 'C':
                    return (0, 255, 255)  # 노란색
                elif grade == 'D':
                    return (0, 165, 255)  # 주황색
                elif grade == 'F':
                    return (0, 0, 255)    # 빨간색
                else:
                    return (255, 255, 255) # 흰색 (기본값)
            
            # 사이클 best 점수에 따른 전체 글씨 색상 결정
            if current_cycle_best_score > 0:
                overall_color = get_grade_color(current_cycle_best_grade)
            else:
                overall_color = (255, 255, 255)  # 흰색
            
            y_offset = 25
            for i, text in enumerate(overlay_text):
                y_pos = frame_height - 180 + (i * y_offset)
                
                # 각 라인별 색상 결정
                if i == 0:  # Phase 라인
                    text_color = phase_col
                    font_size = font_scale + 0.2
                    line_thickness = thickness + 1
                elif i == 4:  # Best 등급 라인
                    best_grade_color = get_grade_color(display_grade) if display_grade else (255, 255, 255)
                    text_color = best_grade_color
                    font_size = font_scale
                    line_thickness = thickness
                else:  # 기타 라인들 - 사이클 best 점수에 따른 색상
                    text_color = overall_color
                    font_size = font_scale
                    line_thickness = thickness
                
                cv2.putText(
                    frame, text, (20, y_pos),
                    font_face, font_size, text_color, line_thickness, lineType=cv2.LINE_AA
                )

            display_countdown -= 1

        # 스켈레톤은 항상 그리기 (점수 표시 여부와 관계없이)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        out.write(frame)

    # 8. 정리
    cap.release()
    out.release()
    pose.close()

    print(f"[완료] 개선된 스쿼트 분석 결과 저장: {output_path}")
    print(f"동적 임계값: {dynamic_thresholds}")

# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행 부분
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 기본 파일명으로 실행 (기존과 호환)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "squat.mp4")
    output_path = os.path.join(script_dir, "squat_output.mp4")
    
    # 함수 호출
    analyze_squat_video(
        input_path=input_path,
        output_path=output_path,
        user_height=170.0,
        squat_depth='medium',
        display_duration_seconds=3
    ) 