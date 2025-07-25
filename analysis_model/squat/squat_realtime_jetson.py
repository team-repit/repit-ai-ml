import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import List, Dict, Tuple
from collections import Counter as GradeCounter

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

def initialize_camera():
    """젯슨 환경에서 카메라를 초기화하는 함수"""
    print("카메라 초기화 중...")
    
    # 사용 가능한 카메라 장치 확인
    available_cameras = []
    for i in range(10):  # 0-9까지 테스트
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"카메라 {i} 사용 가능")
            cap.release()
    
    if not available_cameras:
        print("사용 가능한 카메라가 없습니다.")
        return None
    
    # 첫 번째 사용 가능한 카메라 사용
    camera_index = available_cameras[0]
    print(f"카메라 {camera_index}를 사용합니다.")
    
    # 카메라 초기화
    cap = cv2.VideoCapture(camera_index)
    
    # 젯슨 최적화 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 젯슨에서는 낮은 해상도 권장
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # V4L2 백엔드 사용 (리눅스에서 더 안정적)
    cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"카메라 {camera_index}를 열 수 없습니다.")
        return None
    
    # 실제 설정된 값 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"카메라 설정: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    return cap

def main():
    """실시간 카메라를 통한 스쿼트 분석 메인 함수 (젯슨 최적화)"""
    
    print("=== 젯슨 실시간 스쿼트 분석 ===")
    print("카메라 초기화 중...")
    
    # 카메라 초기화
    cap = initialize_camera()
    
    if cap is None:
        print("카메라를 초기화할 수 없습니다.")
        print("\n문제 해결 방법:")
        print("1. 카메라가 연결되어 있는지 확인")
        print("2. 카메라 권한 확인: sudo usermod -a -G video $USER")
        print("3. 시스템 재부팅 후 다시 시도")
        print("4. 다른 카메라 장치 시도: v4l2-ctl --list-devices")
        return
    
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
    
    print("\n스쿼트 분석을 시작합니다. 15초간 카메라가 켜집니다.")
    print("스쿼트 동작을 시작하세요!")
    print("종료하려면 'q'를 누르세요.")
    print("="*50)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("프레임을 읽을 수 없습니다.")
            break
        
        frame_count += 1
        
        # 현재 시간 계산
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, recording_duration - elapsed_time)
        
        # 15초 경과 시 종료
        if elapsed_time >= recording_duration:
            print("\n15초 분석 완료!")
            break
        
        # 프레임 처리
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
        
        # 포즈 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
        
        out.write(image)
        cv2.imshow('Jetson Real-time Squat Analysis', image)

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            print("\n사용자가 조기 종료했습니다.")
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
    print(f"\n분석 완료!")
    print(f"분석 영상: '{output_video_path}'")
    print(f"분석 리포트: '{output_report_path}'")
    print(f"총 {counter}회의 스쿼트를 분석했습니다.")
    print(f"처리된 프레임 수: {frame_count}")

if __name__ == "__main__":
    main() 