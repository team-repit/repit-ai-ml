"""
매우 완화된 스쿼트 프레임 추출기
- 등급 기준을 매우 관대하게 조정하여 A, B 등급이 충분히 나올 수 있도록 함
- 실제 스쿼트 자세보다는 상대적 품질에 집중
"""
import cv2
import numpy as np
import os
import json
from typing import List, Dict
import mediapipe as mp
from datetime import datetime

# 기존 스쿼트 모듈 import
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from whole_mech_new import (
        calculate_angle, validate_landmarks, AdvancedFilteringSystem,
        SquatPhaseDetector, BiomechanicalValidator, SquatPoseValidator,
        evaluate_advanced_squat_grade, initialize_mediapipe_pose,
        initialize_filters_and_validator
    )
except ImportError as e:
    print(f"Import error: {e}")
    try:
        from whole_mechanism_improved import (
            calculate_angle, validate_landmarks, AdvancedFilteringSystem,
            SquatPhaseDetector, BiomechanicalValidator, SquatPoseValidator,
            evaluate_advanced_squat_grade, initialize_mediapipe_pose,
            initialize_filters_and_validator
        )
    except ImportError:
        print("Error: Required modules not found.")
        raise

class VeryRelaxedFrameExtractor:
    def __init__(self, output_dir: str = "very_relaxed_squat_dataset"):
        self.output_dir = output_dir
        self.mp_pose, self.pose = initialize_mediapipe_pose()
        self.filters_and_validator = initialize_filters_and_validator(user_height=170.0, squat_depth='medium')
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # 영상별 품질 정보 (4개 영상)
        self.video_quality = {
            'squat.mp4': 'poor',       # 잘못된 자세
            'squat2.mp4': 'medium',    # 보통 자세
            'squat3.mp4': 'medium',    # 보통 자세
            'squat4.mp4': 'excellent'  # 좋은 자세
        }
    
    def get_very_relaxed_grade(self, angles: Dict, biomechanical_score: float, video_name: str) -> str:
        """매우 관대한 기준으로 등급 부여 (A, B 등급이 충분히 나올 수 있도록)"""
        quality = self.video_quality.get(video_name, 'medium')
        
        # 기본 점수 계산
        knee_angle = angles.get('knee', 90)
        hip_angle = angles.get('hip', 45)
        ankle_angle = angles.get('ankle', 70)
        torso_angle = angles.get('torso', 45)
        
        # 매우 관대한 기준 적용
        if quality == 'poor':
            # 잘못된 자세: D, F 등급 위주 (하지만 C도 가능)
            if knee_angle < 100 and hip_angle < 90:
                return "C" if biomechanical_score > 0.3 else "D"
            elif knee_angle < 120 and hip_angle < 110:
                return "D" if biomechanical_score > 0.2 else "F"
            else:
                return "F"
        elif quality == 'excellent':
            # 좋은 자세: A, B 등급 위주
            if knee_angle < 100 and hip_angle < 85:
                return "A" if biomechanical_score > 0.4 else "B"
            elif knee_angle < 110 and hip_angle < 95:
                return "B" if biomechanical_score > 0.3 else "C"
            elif knee_angle < 120 and hip_angle < 105:
                return "C" if biomechanical_score > 0.2 else "D"
            else:
                return "D"
        else:  # medium
            # 보통 자세: B, C, D 등급
            if knee_angle < 105 and hip_angle < 90:
                return "B" if biomechanical_score > 0.4 else "C"
            elif knee_angle < 115 and hip_angle < 100:
                return "C" if biomechanical_score > 0.3 else "D"
            elif knee_angle < 125 and hip_angle < 110:
                return "D" if biomechanical_score > 0.2 else "F"
            else:
                return "F"
    
    def extract_squat_frames(self, video_path: str, min_confidence: float = 0.5) -> List[Dict]:
        """영상에서 스쿼트 프레임 추출 (매우 관대한 등급 부여)"""
        print(f"영상 처리 중: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"영상을 열 수 없습니다: {video_path}")
            return []
        
        frames = []
        frame_count = 0
        
        # 필터 초기화
        try:
            left_knee_filter = self.filters_and_validator['left_knee_filter']
            right_knee_filter = self.filters_and_validator['right_knee_filter']
            left_hip_filter = self.filters_and_validator['left_hip_filter']
            right_hip_filter = self.filters_and_validator['right_hip_filter']
            phase_detector = self.filters_and_validator['phase_detector']
        except KeyError as e:
            print(f"필터 초기화 오류: {e}")
            cap.release()
            return []
        
        video_name = os.path.basename(video_path)
        quality = self.video_quality.get(video_name, 'medium')
        print(f"영상 품질: {quality}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe 처리
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                h, w, _ = frame.shape
                lm = results.pose_landmarks.landmark
                
                # 필요한 랜드마크 검증
                required_landmarks = [
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                ]
                
                if validate_landmarks(lm, required_landmarks, min_confidence=min_confidence):
                    def get_point(landmark):
                        return (int(landmark.x * w), int(landmark.y * h))
                    
                    # 주요 관절 좌표 추출
                    left_hip = get_point(lm[self.mp_pose.PoseLandmark.LEFT_HIP])
                    right_hip = get_point(lm[self.mp_pose.PoseLandmark.RIGHT_HIP])
                    left_knee = get_point(lm[self.mp_pose.PoseLandmark.LEFT_KNEE])
                    right_knee = get_point(lm[self.mp_pose.PoseLandmark.RIGHT_KNEE])
                    left_ankle = get_point(lm[self.mp_pose.PoseLandmark.LEFT_ANKLE])
                    right_ankle = get_point(lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
                    left_shoulder = get_point(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
                    right_shoulder = get_point(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
                    
                    # 각도 계산
                    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
                    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
                    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
                    angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
                    angle_left_ankle = calculate_angle(left_knee, left_ankle, (left_ankle[0], left_ankle[1] - 50))
                    angle_right_ankle = calculate_angle(right_knee, right_ankle, (right_ankle[0], right_ankle[1] - 50))
                    angle_left_torso = calculate_angle(left_hip, left_shoulder, (left_shoulder[0], left_shoulder[1] - 50))
                    angle_right_torso = calculate_angle(right_hip, right_shoulder, (right_shoulder[0], right_shoulder[1] - 50))
                    
                    # 필터 적용
                    left_knee_result = left_knee_filter.update(angle_left_knee)
                    right_knee_result = right_knee_filter.update(angle_right_knee)
                    left_hip_result = left_hip_filter.update(angle_left_hip)
                    right_hip_result = right_hip_filter.update(angle_right_hip)
                    
                    # 평균 각도 계산
                    avg_knee_angle = (angle_left_knee + angle_right_knee) / 2
                    avg_hip_angle = (angle_left_hip + angle_right_hip) / 2
                    avg_ankle_angle = (angle_left_ankle + angle_right_ankle) / 2
                    avg_torso_angle = (angle_left_torso + angle_right_torso) / 2
                    
                    # 스쿼트 조건 확인 (매우 관대한 조건)
                    is_squat_position = False
                    
                    if quality == 'poor':
                        # 잘못된 자세: 매우 관대한 조건
                        is_squat_position = (
                            avg_knee_angle < 150 and  # 무릎이 조금이라도 구부러짐
                            avg_hip_angle < 130 and   # 엉덩이가 조금이라도 구부러짐
                            avg_knee_angle > 30       # 너무 깊지 않음
                        )
                    elif quality == 'excellent':
                        # 좋은 자세: 관대한 조건
                        is_squat_position = (
                            avg_knee_angle < 130 and  # 무릎이 구부러짐
                            avg_hip_angle < 100 and   # 엉덩이가 구부러짐
                            avg_knee_angle > 50       # 적절한 깊이
                        )
                    else:  # medium
                        # 보통 자세: 관대한 조건
                        is_squat_position = (
                            avg_knee_angle < 140 and  # 무릎이 구부러짐
                            avg_hip_angle < 110 and   # 엉덩이가 구부러짐
                            avg_knee_angle > 40       # 너무 깊지 않음
                        )
                    
                    if is_squat_position:
                        # 생체역학적 점수 계산
                        try:
                            biomechanical_validator = self.filters_and_validator['biomechanical_validator']
                            current_angles = {
                                'knee': avg_knee_angle,
                                'hip': avg_hip_angle,
                                'ankle': avg_ankle_angle,
                                'torso': avg_torso_angle
                            }
                            positions = {
                                'knee': left_knee,
                                'ankle': left_ankle
                            }
                            biomechanical_scores = biomechanical_validator.calculate_biomechanical_score(
                                current_angles, positions
                            )
                        except:
                            biomechanical_scores = {'overall': 0.5}
                        
                        # 매우 관대한 등급 부여
                        very_relaxed_grade = self.get_very_relaxed_grade(
                            current_angles, 
                            biomechanical_scores['overall'], 
                            video_name
                        )
                        
                        # 프레임 정보 저장
                        frame_info = {
                            'frame_number': frame_count,
                            'knee_angle': avg_knee_angle,
                            'hip_angle': avg_hip_angle,
                            'ankle_angle': avg_ankle_angle,
                            'torso_angle': avg_torso_angle,
                            'biomechanical_score': biomechanical_scores['overall'],
                            'grade': very_relaxed_grade,
                            'video_path': video_path,
                            'frame': frame.copy()
                        }
                        
                        frames.append(frame_info)
                        print(f"스쿼트 프레임 감지: 프레임 {frame_count}, 등급 {very_relaxed_grade}")
            
            frame_count += 1
            
            # 진행상황 표시
            if frame_count % 100 == 0:
                print(f"프레임 처리 중: {frame_count}")
        
        cap.release()
        print(f"총 {len(frames)}개의 스쿼트 프레임 추출 완료")
        return frames
    
    def save_frames(self, frames: List[Dict], video_name: str):
        """프레임들을 저장 (매우 관대한 등급 사용)"""
        if not frames:
            print("저장할 프레임이 없습니다.")
            return
        
        # 등급별로 분류
        grade_frames = {'A': [], 'B': [], 'C': [], 'D': [], 'F': []}
        for frame in frames:
            grade = frame['grade']
            if grade in grade_frames:
                grade_frames[grade].append(frame)
        
        # 등급별 폴더 생성 및 저장
        saved_count = 0
        for grade, grade_frame_list in grade_frames.items():
            if grade_frame_list:
                grade_dir = os.path.join(self.output_dir, grade)
                os.makedirs(grade_dir, exist_ok=True)
                
                for i, frame_info in enumerate(grade_frame_list):
                    filename = f"{video_name}_{grade}_{i:03d}.jpg"
                    filepath = os.path.join(grade_dir, filename)
                    
                    success = cv2.imwrite(filepath, frame_info['frame'])
                    if success:
                        saved_count += 1
                        print(f"저장: {filename}")
        
        print(f"총 {saved_count}개 프레임 저장 완료")
        
        # 메타데이터 저장
        metadata = {
            'video_name': video_name,
            'total_frames': len(frames),
            'grade_distribution': {grade: len(frames_list) for grade, frames_list in grade_frames.items()},
            'extraction_date': datetime.now().isoformat(),
            'frames_info': []
        }
        
        for frame in frames:
            frame_metadata = {
                'frame_number': frame['frame_number'],
                'grade': frame['grade'],
                'knee_angle': float(frame['knee_angle']),
                'hip_angle': float(frame['hip_angle']),
                'ankle_angle': float(frame['ankle_angle']),
                'torso_angle': float(frame['torso_angle']),
                'biomechanical_score': float(frame['biomechanical_score'])
            }
            metadata['frames_info'].append(frame_metadata)
        
        metadata_path = os.path.join(self.output_dir, "metadata", f"{video_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"메타데이터 저장: {metadata_path}")

def main():
    """메인 실행 함수"""
    print("=== 매우 완화된 스쿼트 프레임 추출기 ===")
    print("등급 기준을 매우 관대하게 조정하여 A, B 등급이 충분히 나올 수 있도록 합니다.")
    
    # 비디오 파일들 (4개)
    video_files = ["squat.mp4", "squat2.mp4", "squat3.mp4", "squat4.mp4"]
    
    extractor = VeryRelaxedFrameExtractor()
    
    all_frames = []
    
    for video_file in video_files:
        if os.path.exists(video_file):
            print(f"\n=== {video_file} 처리 중 ===")
            frames = extractor.extract_squat_frames(video_file, min_confidence=0.5)
            
            if frames:
                # 개별 비디오별로 저장
                video_name = os.path.splitext(video_file)[0]
                extractor.save_frames(frames, video_name)
                all_frames.extend(frames)
            else:
                print(f"{video_file}에서 프레임을 추출할 수 없습니다.")
        else:
            print(f"{video_file} 파일을 찾을 수 없습니다.")
    
    # 전체 통계
    if all_frames:
        print(f"\n=== 전체 통계 ===")
        print(f"총 추출된 프레임: {len(all_frames)}개")
        
        # 등급 분포
        grade_counts = {}
        for frame in all_frames:
            grade = frame['grade']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        print("\n등급 분포:")
        for grade in ['A', 'B', 'C', 'D', 'F']:
            count = grade_counts.get(grade, 0)
            print(f"  {grade}: {count}개")
        
        # 영상별 통계
        print("\n영상별 통계:")
        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            video_frames = [f for f in all_frames if f['video_path'] == video_file]
            
            if video_frames:
                counts = {}
                for frame in video_frames:
                    grade = frame['grade']
                    counts[grade] = counts.get(grade, 0) + 1
                
                quality = extractor.video_quality.get(video_file, 'medium')
                print(f"\n{video_file} ({quality} 품질):")
                for grade in ['A', 'B', 'C', 'D', 'F']:
                    count = counts.get(grade, 0)
                    print(f"  {grade}: {count}개")
    else:
        print("추출된 프레임이 없습니다.")

if __name__ == "__main__":
    main() 