import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import List, Dict, Tuple
from collections import Counter

# --- 경로 설정 ---
# 출력 파일 기본 이름
output_file_base_name = 'plank_output'

# --- 자동 넘버링으로 출력 경로 설정 ---
n = 1
while True:
    output_report_path = f"{output_file_base_name}_report_{n}.txt"
    output_video_path = f"{output_file_base_name}_video_{n}.mp4"
    if not os.path.exists(output_report_path) and not os.path.exists(output_video_path):
        break
    n += 1

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

class PlankGrader:
    """
    'AI 플랭크 자세 교정을 위한 종합 평가 기준'을 기반으로 한 평가 클래스. (대폭 완화된 기준 적용)
    """
    def __init__(self):
        pass

    def evaluate_errors(self, angles: dict, landmarks: dict) -> List[str]:
        """
        자세를 평가하고 발생한 모든 오류 목록을 계층적으로 반환합니다.
        """
        errors = []
        
        # 레벨 1: 안전성 (Safety) - 기준 대폭 완화
        if 'body' in angles and angles['body'] > 200: # 190 -> 200
            errors.append("엉덩이 처짐")

        # 레벨 2: 효과성 (Effectiveness) - 기준 대폭 완화
        if 'body' in angles and angles['body'] < 150: # 165 -> 150
            errors.append("엉덩이 솟음")
        if 'neck' in angles and not (150 <= angles['neck'] <= 210): # 165-195 -> 150-210
            errors.append("고개 정렬 불량")

        # 레벨 3: 최적화 (Optimization) - 기준 대폭 완화
        is_elbow_misaligned = 'arm' in angles and not (60 <= angles['arm'] <= 120) # 75-105 -> 60-120
        # 팔꿈치가 어깨보다 너무 앞이나 뒤에 있는지 확인
        shoulder_x = (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2
        elbow_x = (landmarks['left_elbow'][0] + landmarks['right_elbow'][0]) / 2
        shoulder_hip_dist = abs(landmarks['left_shoulder'][0] - landmarks['left_hip'][0]) # 기준 거리
        is_elbow_pos_off = abs(shoulder_x - elbow_x) > shoulder_hip_dist * 0.40 # 0.20 -> 0.40

        if is_elbow_misaligned or is_elbow_pos_off:
            errors.append("팔꿈치 정렬 불량")
            
        if 'leg' in angles and angles['leg'] < 150: # 165 -> 150
            errors.append("무릎 굽힘")

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
    "엉덩이 처짐": "엉덩이 처짐 (Hip Sag / 허리 꺾임): 코어와 둔근의 힘이 풀려 허리가 U자로 꺾이는 현상.",
    "엉덩이 솟음": "엉덩이 솟음 (Hip Pike): 코어의 부담을 줄이기 위해 엉덩이를 과도하게 높이 드는 자세.",
    "고개 정렬 불량": "고개 떨굼 / 젖힘 (Head/Neck Misalignment): 목이 척추의 중립선에서 벗어나는 자세.",
    "팔꿈치 정렬 불량": "팔꿈치/손목 정렬 불량 (Elbow/Wrist Misalignment): 팔꿈치가 어깨 바로 아래에 위치하지 않는 자세.",
    "무릎 굽힘": "무릎 굽힘 (Knee Bend): 다리의 긴장이 풀려 무릎이 굽혀지는 현상."
}

def save_report(report_path: str, hold_results: List[Dict]):
    """분석 결과와 전체 평가 기준을 텍스트 파일로 저장합니다."""
    total_hold_time = sum(res['duration'] for res in hold_results)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("플랭크 자세 분석 리포트\n")
        f.write("="*30 + "\n")
        f.write(f"총 플랭크 유지 시간: {total_hold_time:.2f}초\n\n")
        
        f.write("구간별 상세 결과:\n")
        if not hold_results:
            f.write("- 플랭크 자세가 감지되지 않았습니다.\n")
        
        for i, res in enumerate(hold_results):
            grade = res['grade']
            duration = res['duration']
            errors = res['errors']
            f.write(f"\n--- {i+1}번째 구간 (유지 시간: {duration:.2f}초): 등급 {grade} ---\n")
            if errors:
                f.write("  [주요 발생 오류]\n")
                # 가장 많이 발생한 오류 순으로 정렬
                sorted_errors = sorted(errors.items(), key=lambda item: item[1], reverse=True)
                for error_key, count in sorted_errors:
                    error_description = ERROR_CRITERIA_MAP.get(error_key, "알 수 없는 오류")
                    f.write(f"  - {error_description} ({count}회 감지)\n")
            else:
                f.write("  - 모든 기준을 만족했습니다.\n")

        # --- 전체 평가 기준 추가 (대폭 완화된 기준 반영) ---
        f.write("\n\n" + "="*40 + "\n")
        f.write("          자세 평가 기준 (참고)\n")
        f.write("="*40 + "\n\n")
        f.write("레벨 1: 안전성 (Safety) - 즉시 교정 대상\n")
        f.write("- 엉덩이 처짐 (Hip Sag / 허리 꺾임): 어깨-엉덩이-발목 각도 > 200도\n\n")
        f.write("레벨 2: 효과성 (Effectiveness) - 주요 교정 대상\n")
        f.write("- 엉덩이 솟음 (Hip Pike): 어깨-엉덩이-발목 각도 < 150도\n")
        f.write("- 고개 떨굼 / 젖힘 (Head/Neck Misalignment): 귀-어깨-엉덩이 각도가 150도~210도 범위를 벗어남\n\n")
        f.write("레벨 3: 최적화 (Optimization) - 미세 조정\n")
        f.write("- 팔꿈치/손목 정렬 불량 (Elbow/Wrist Misalignment): 어깨-팔꿈치-손목 각도가 60도~120도 범위를 벗어나거나, 팔꿈치가 어깨 수직선상에서 벗어남\n")
        f.write("- 무릎 굽힘 (Knee Bend): 고관절-무릎-발목 각도 < 150도\n")

    print(f"리포트가 '{report_path}'에 저장되었습니다.")


# --- 메인 실행 로직 ---
cap = cv2.VideoCapture(1)

# --- 사용자에게 분석 시간 입력받기 ---
try:
    minutes = int(input("플랭크 분석 시간 - 몇 분? (정수로 입력): "))
    seconds = int(input("플랭크 분석 시간 - 몇 초? (정수로 입력): "))
except ValueError:
    print("잘못된 입력입니다. 기본값 1분(60초)으로 진행합니다.")
    minutes, seconds = 1, 0

total_duration = minutes * 60 + seconds
print(f"\n총 분석 시간: {total_duration}초 동안 플랭크 자세를 분석합니다.\n")

# --- 시작 시간 기록 ---
start_time = time.time()


# 영상 저장을 위한 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 변수 초기화
grader = PlankGrader()
all_hold_results = []
current_hold_errors = Counter()
is_holding = False
hold_start_time = 0



while cap.isOpened():
    elapsed_time = time.time() - start_time
    if elapsed_time >= total_duration:
        print("분석 시간이 종료되었습니다.")
        break

    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    is_plank_pose = False
    feedback = ""
    grade = "N/A"
    
    try:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        
        lm_data = {
            'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h],
            'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h],
            'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h],
            'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h],
            'left_ear': [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h],
            'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h],
            'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h],
            'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h],
            'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h],
            'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h],
            'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h],
            'right_ear': [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h],
            'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h],
            'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h],
        }

        # 플랭크 자세 기본 조건 확인
        shoulder_y = (lm_data['left_shoulder'][1] + lm_data['right_shoulder'][1]) / 2
        hip_y = (lm_data['left_hip'][1] + lm_data['right_hip'][1]) / 2
        if shoulder_y < hip_y + 50: # 어깨가 엉덩이보다 너무 낮지 않은지 (엎드린 자세 확인)
            is_plank_pose = True

        if is_plank_pose:
            # 각도 계산
            angles = {}
            use_left_side = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            if use_left_side:
                angles['body'] = calculate_angle(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_ankle'])
                angles['neck'] = calculate_angle(lm_data['left_ear'], lm_data['left_shoulder'], lm_data['left_hip'])
                angles['arm'] = calculate_angle(lm_data['left_shoulder'], lm_data['left_elbow'], lm_data['left_wrist'])
                angles['leg'] = calculate_angle(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle'])
            else:
                angles['body'] = calculate_angle(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_ankle'])
                angles['neck'] = calculate_angle(lm_data['right_ear'], lm_data['right_shoulder'], lm_data['right_hip'])
                angles['arm'] = calculate_angle(lm_data['right_shoulder'], lm_data['right_elbow'], lm_data['right_wrist'])
                angles['leg'] = calculate_angle(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle'])
            
            # 플랭크 유지 상태 관리
            if not is_holding:
                is_holding = True
                hold_start_time = time.time()
                current_hold_errors.clear()

            # 오류 평가
            errors_in_frame = grader.evaluate_errors(angles, lm_data)
            current_hold_errors.update(errors_in_frame)
            grade = grader.get_grade_from_errors(list(current_hold_errors.keys()))
            feedback = ", ".join(errors_in_frame) if errors_in_frame else "자세 좋습니다!"

        elif is_holding: # 플랭크 자세가 깨졌을 때
            is_holding = False
            hold_duration = time.time() - hold_start_time
            if hold_duration > 1: # 1초 이상 유지했을 때만 기록
                final_grade = grader.get_grade_from_errors(list(current_hold_errors.keys()))
                all_hold_results.append({
                    'duration': hold_duration,
                    'grade': final_grade,
                    'errors': current_hold_errors
                })

    except Exception as e:
        pass

    # 화면에 정보 표시
    # 상태 박스
    status_text = "HOLDING" if is_holding else "READY"
    status_color = (0, 255, 0) if is_holding else (0, 0, 255)
    cv2.rectangle(image, (0,0), (450, 72), (245,117,16), -1)
    cv2.putText(image, 'STATUS', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, status_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 2, cv2.LINE_AA)
    
    # 시간 표시
    hold_time = (time.time() - hold_start_time) if is_holding else 0
    cv2.putText(image, 'TIME', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, f"{hold_time:.1f}s", (195,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    # 등급 표시
    cv2.putText(image, 'GRADE', (350,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, grade, (370,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    # 피드백 박스
    # feedback_color = (0, 0, 255) if feedback != "자세 좋습니다!" else (0, 255, 0)
    # cv2.rectangle(image, (0, 410), (640, 480), feedback_color, -1)
    # cv2.putText(image, 'FEEDBACK', (15, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    # cv2.putText(image, feedback, (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
    
    out.write(image)
    cv2.imshow('Comprehensive Plank Analysis', image)

    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

# 마지막 홀드 세션 저장
if is_holding:
    hold_duration = time.time() - hold_start_time
    if hold_duration > 1:
        final_grade = grader.get_grade_from_errors(list(current_hold_errors.keys()))
        all_hold_results.append({
            'duration': hold_duration,
            'grade': final_grade,
            'errors': current_hold_errors
        })

save_report(output_report_path, all_hold_results)
print(f"분석 영상이 '{output_video_path}'에 저장되었습니다.")
