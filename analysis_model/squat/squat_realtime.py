import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import List, Dict, Tuple
from collections import Counter as GradeCounter

# MediaPipe Pose 모델 초기화
print("🔍 MediaPipe 초기화 중...")
import time
start_time = time.time()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=1,  # 0:Lite, 1:Full, 2:Heavy (기본값: 1)
    enable_segmentation=False,  # 세그멘테이션 비활성화로 속도 향상
    smooth_landmarks=True
)
mp_drawing = mp.solutions.drawing_utils

load_time = time.time() - start_time
print(f"✅ MediaPipe 로드 완료! (소요시간: {load_time:.2f}초)")

def calculate_angle(a: list, b: list, c: list) -> float:
    """세 점 사이의 각도를 계산하는 함수 (결과값: 0-180)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class ComprehensiveSquatGrader:
    """
    'AI 자세 교정을 위한 종합 평가 기준'을 기반으로 한 새로운 평가 클래스.
    계층적 피드백 구조(안전성 > 효과성 > 최적화)를 따릅니다.
    """
    def __init__(self):
        pass

    def evaluate_errors(self, landmarks: dict, angles: dict, phase: str, rep_start_hip_y: float) -> List[str]:
        """
        자세를 평가하고 발생한 모든 오류 목록을 계층적으로 반환합니다.
        """
        errors = []
        
        # 레벨 1: 안전성 (Safety) - 즉시 교정 대상
        if phase in ["DESCEND", "BOTTOM", "ASCEND"]:
            # 1-1. 허리 말림 (Butt Wink)
            if 'hip' in angles and angles['hip'] < 65:
                errors.append("허리 말림")
            
            # 1-2. 무릎 모임 (Knee Valgus)
            lk_pos, rk_pos = landmarks.get('left_knee'), landmarks.get('right_knee')
            la_pos, ra_pos = landmarks.get('left_ankle'), landmarks.get('right_ankle')
            if all([lk_pos, rk_pos, la_pos, ra_pos]):
                knee_dist = abs(lk_pos[0] - rk_pos[0])
                ankle_dist = abs(la_pos[0] - ra_pos[0])
                if ankle_dist > 0 and knee_dist < ankle_dist * 0.85:
                    errors.append("무릎 모임")

            # 1-3. "굿모닝" 스쿼트
            if phase == "ASCEND":
                hip_y = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
                shoulder_y = (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
                # 엉덩이가 어깨보다 유의미하게 먼저 올라가는지 확인
                if hip_y < (rep_start_hip_y * 0.9) and shoulder_y > (rep_start_hip_y * 0.95):
                     errors.append("굿모닝 스쿼트")

        # 레벨 2: 효과성 (Effectiveness) - 주요 교정 대상
        if phase in ["DESCEND", "BOTTOM"]:
            # 2-1. 과도한 상체 숙임 (Chest Drop)
            if 'torso' in angles and angles['torso'] < 45 and "허리 말림" not in errors:
                errors.append("상체 숙임")
            
            # 2-2. 뒤꿈치 들림 (Heel Lift)
            left_heel_vis = landmarks.get('left_heel_visibility', 1.0)
            right_heel_vis = landmarks.get('right_heel_visibility', 1.0)
            if left_heel_vis < 0.7 or right_heel_vis < 0.7:
                 errors.append("뒤꿈치 들림")

            # 2-3. 골반 치우침 (Pelvic Shift)
            hip_center_x = (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2
            ankle_center_x = (landmarks['left_ankle'][0] + landmarks['right_ankle'][0]) / 2
            shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
            if shoulder_width > 0 and abs(hip_center_x - ankle_center_x) > shoulder_width * 0.15:
                errors.append("골반 치우침")

        # 레벨 3: 최적화 (Optimization) - 미세 조정
        if phase == "BOTTOM":
            # 3-1. 깊이 부족 (Insufficient Depth)
            if 'knee' in angles and angles['knee'] > 120:
                errors.append("깊이 부족")
            
            # 3-2. 발목 가동성 부족 (Ankle Mobility)
            if 'ankle' in angles and angles['ankle'] > 80: # 배굴곡 각도가 충분하지 않음
                errors.append("발목 가동성 부족")

        return errors

    def get_grade_from_errors(self, errors: List[str]) -> str:
        """오류 개수에 따라 등급을 반환합니다."""
        num_errors = len(set(errors))
        if num_errors == 0: return "A"
        elif num_errors == 1: return "B"
        elif num_errors == 2: return "C"
        elif num_errors == 3: return "D"
        else: return "F"

# 오류 키와 상세 설명을 매핑하는 딕셔너리
ERROR_CRITERIA_MAP = {
    "허리 말림": "허리 말림 (Butt Wink): 하강 최저점에서 엉덩이가 안으로 말리며 허리의 중립이 무너지는 현상.",
    "무릎 모임": "무릎 모임 (Knee Valgus): 하강 또는 상승 시 무릎이 발보다 안쪽으로 무너지는 현상.",
    "굿모닝 스쿼트": '"굿모닝" 스쿼트: 상승 시 엉덩이가 상체보다 현저히 빠르게 올라와 허리에 과부하가 걸리는 현상.',
    "상체 숙임": "과도한 상체 숙임 (Chest Drop): 힙 힌지 범위를 넘어 상체가 과도하게 앞으로 쏠리는 자세.",
    "뒤꿈치 들림": "뒤꿈치 들림 (Heel Lift): 무게 중심이 앞으로 쏠려 뒤꿈치가 바닥에서 뜨는 현상.",
    "골반 치우침": "골반 치우침 (Pelvic Shift): 하강 또는 상승 시 골반이 좌우 한쪽으로 쏠리는 현상.",
    "깊이 부족": "깊이 부족 (Insufficient Depth): 허벅지가 지면과 평행이 되는 지점까지 충분히 하강하지 못하는 경우.",
    "발목 가동성 부족": "발목 가동성 부족 (Ankle Mobility): 스쿼트 최저점에서 발목 각도(배굴곡)가 충분하지 않은 경우."
}

def save_report(report_path: str, total_reps: int, results: List[Dict]):
    """분석 결과와 전체 평가 기준을 텍스트 파일로 저장합니다."""
    grades = [res['grade'] for res in results]
    grade_counts = GradeCounter(grades)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("실시간 스쿼트 자세 분석 리포트\n")
        f.write("="*30 + "\n")
        f.write(f"총 스쿼트 횟수: {total_reps}회\n\n")
        
        f.write("등급별 요약:\n")
        for grade in ["A", "B", "C", "D", "F"]:
            count = grade_counts.get(grade, 0)
            f.write(f"- 등급 {grade}: {count}회\n")
        
        f.write("\n" + "="*30 + "\n")
        f.write("반복별 상세 결과:\n")
        for res in results:
            f.write(f"\n--- {res['rep']}회차: 등급 {res['grade']} ---\n")
            if res['errors']:
                f.write("  [수행하지 못한 기준]\n")
                for error_key in sorted(res['errors']): # 오류를 가나다 순으로 정렬하여 출력
                    error_description = ERROR_CRITERIA_MAP.get(error_key, "알 수 없는 오류")
                    f.write(f"  - {error_description}\n")
            else:
                f.write("  - 모든 기준을 만족했습니다.\n")

        # --- 전체 평가 기준 추가 ---
        f.write("\n\n" + "="*40 + "\n")
        f.write("          자세 평가 기준 (참고)\n")
        f.write("="*40 + "\n\n")

        # 스쿼트 기준
        f.write("1. 스쿼트 (Squat) 종합 기준\n")
        f.write("-------------------------\n")
        f.write("레벨 1: 안전성 (Safety) - 즉시 교정 대상\n")
        f.write("- 허리 말림 (Butt Wink): 하강 최저점에서 엉덩이가 안으로 말리며 허리의 중립이 무너지는 현상.\n")
        f.write("- 무릎 모임 (Knee Valgus): 하강 또는 상승 시 무릎이 발보다 안쪽으로 무너지는 현상.\n")
        f.write("- \"굿모닝\" 스쿼트: 상승 시 엉덩이가 상체보다 현저히 빠르게 올라와 허리에 과부하가 걸리는 현상.\n\n")
        f.write("레벨 2: 효과성 (Effectiveness) - 주요 교정 대상\n")
        f.write("- 과도한 상체 숙임 (Chest Drop): 힙 힌지 범위를 넘어 상체가 과도하게 앞으로 쏠리는 자세.\n")
        f.write("- 뒤꿈치 들림 (Heel Lift): 무게 중심이 앞으로 쏠려 뒤꿈치가 바닥에서 뜨는 현상.\n")
        f.write("- 골반 치우침 (Pelvic Shift): 하강 또는 상승 시 골반이 좌우 한쪽으로 쏠리는 현상.\n\n")
        f.write("레벨 3: 최적화 (Optimization) - 미세 조정\n")
        f.write("- 깊이 부족 (Insufficient Depth): 허벅지가 지면과 평행이 되는 지점(무릎 각도 약 110~120도)까지 충분히 하강하지 못하는 경우.\n")
        f.write("- 발목 가동성 부족 (Ankle Mobility): 스쿼트 최저점에서 발목 각도(배굴곡)가 약 20도 미만으로, 가동 범위가 제한되는 경우.\n\n")

    print(f"리포트가 '{report_path}'에 저장되었습니다.")

def main():
    """실시간 카메라를 통한 스쿼트 분석 메인 함수"""
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)  # 기본 카메라 (보통 내장 웹캠)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 영상 저장을 위한 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 타임스탬프를 포함한 파일명 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_video_path = f"squat_realtime_analysis_{timestamp}.mp4"
    output_report_path = f"squat_realtime_report_{timestamp}.txt"
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 변수 초기화
    counter = 0 
    stage = None
    grader = ComprehensiveSquatGrader()
    all_rep_results = []
    current_rep_errors = set()
    last_rep_grade = "N/A"
    rep_start_hip_y = 0
    
    # 타이머 설정
    start_time = time.time()
    recording_duration = 15  # 15초
    
    print("스쿼트 분석을 시작합니다. 15초간 카메라가 켜집니다.")
    print("스쿼트 동작을 시작하세요!")
    print("종료하려면 'q'를 누르세요.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 현재 시간 계산
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, recording_duration - elapsed_time)
        
        # 15초 경과 시 종료
        if elapsed_time >= recording_duration:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            lm_data = {
                'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h],
                'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h],
                'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h],
                'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h],
                'left_foot_index': [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h],
                'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h],
                'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h],
                'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h],
                'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h],
                'right_foot_index': [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h],
                'left_heel_visibility': landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].visibility,
                'right_heel_visibility': landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].visibility,
            }
            
            angles = {}
            use_left_side = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            if use_left_side:
                angles['hip'] = calculate_angle(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_knee'])
                angles['knee'] = calculate_angle(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle'])
                angles['ankle'] = calculate_angle(lm_data['left_knee'], lm_data['left_ankle'], lm_data['left_foot_index'])
                angles['torso'] = calculate_angle(lm_data['left_hip'], lm_data['left_shoulder'], [lm_data['left_shoulder'][0], lm_data['left_shoulder'][1] - 1])
            else:
                angles['hip'] = calculate_angle(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_knee'])
                angles['knee'] = calculate_angle(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle'])
                angles['ankle'] = calculate_angle(lm_data['right_knee'], lm_data['right_ankle'], lm_data['right_foot_index'])
                angles['torso'] = calculate_angle(lm_data['right_hip'], lm_data['right_shoulder'], [lm_data['right_shoulder'][0], lm_data['right_shoulder'][1] - 1])
            
            if 'knee' in angles:
                knee_angle = angles['knee']
                
                if knee_angle > 160:
                    if stage == 'down': 
                        final_grade = grader.get_grade_from_errors(list(current_rep_errors))
                        all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})
                        last_rep_grade = final_grade
                        current_rep_errors.clear()
                    stage = "up"

                if knee_angle < 100 and stage == 'up':
                    stage = "down"
                    counter += 1
                    rep_start_hip_y = (lm_data['left_hip'][1] + lm_data['right_hip'][1]) / 2

                current_phase = ""
                if stage == "up": current_phase = "ASCEND" if knee_angle < 170 else "READY"
                elif stage == "down": current_phase = "BOTTOM" if knee_angle < 90 else "DESCEND"
                
                if stage == "down" or stage == "up":
                    errors_in_frame = grader.evaluate_errors(lm_data, angles, current_phase, rep_start_hip_y)
                    current_rep_errors.update(errors_in_frame)

        except Exception as e:
            pass
        
        # ------------------ 화면 표시 정보 수정 ------------------
        # 상단 정보 박스
        cv2.rectangle(image, (0,0), (frame_width, 120), (245,117,16), -1)
        
        # 타이머 표시
        cv2.putText(image, f'TIME: {remaining_time:.1f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # REPS
        cv2.putText(image, 'REPS', (int(frame_width * 0.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter), (int(frame_width * 0.3), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # PHASE
        cv2.putText(image, 'PHASE', (int(frame_width * 0.5), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, current_phase if 'current_phase' in locals() else "READY", (int(frame_width * 0.5), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

        # LAST REP GRADE
        cv2.putText(image, 'GRADE', (int(frame_width * 0.7), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, last_rep_grade, (int(frame_width * 0.7), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # 하단 안내 메시지
        cv2.rectangle(image, (0, frame_height-50), (frame_width, frame_height), (0,0,0), -1)
        cv2.putText(image, 'Press Q to quit early', (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        # ----------------------------------------------------
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
        
        out.write(image)
        cv2.imshow('Real-time Squat Analysis', image)

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    # 마지막 스쿼트가 완료되지 않았다면 처리
    if stage == 'down' and current_rep_errors:
        final_grade = grader.get_grade_from_errors(list(current_rep_errors))
        all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 결과 저장
    save_report(output_report_path, counter, all_rep_results)
    print(f"분석 영상이 '{output_video_path}'에 저장되었습니다.")
    print(f"분석 리포트가 '{output_report_path}'에 저장되었습니다.")
    print(f"총 {counter}회의 스쿼트를 분석했습니다.")

if __name__ == "__main__":
    main() 