import cv2
import mediapipe as mp
import numpy as np
import os
from typing import List, Dict, Tuple
from collections import Counter

# --- 경로 설정 ---
# 입력 영상 파일 경로
input_video_name = '화면 기록 2025-07-22 오후 3.35.37.mov'
# 출력 파일 기본 이름
output_file_base_name = '화면 기록 2025-07-22 오후 3.35.37'

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

class LungeGrader:
    """
    'AI 런지 자세 교정을 위한 종합 평가 기준'을 기반으로 한 평가 클래스.
    """
    def __init__(self):
        pass

    def evaluate_errors(self, angles: dict, landmarks: dict, front_leg: str) -> List[str]:
        """
        자세를 평가하고 발생한 모든 오류 목록을 계층적으로 반환합니다.
        (기준 완화 적용)
        """
        errors = []
        
        # 레벨 1: 안전성 (Safety)
        # 1-1. 측면 불안정성 (기존 ±10도 -> ±15도)
        shoulder_angle_with_horizontal = calculate_angle(landmarks['right_shoulder'], landmarks['left_shoulder'], [landmarks['left_shoulder'][0] + 100, landmarks['left_shoulder'][1]])
        hip_angle_with_horizontal = calculate_angle(landmarks['right_hip'], landmarks['left_hip'], [landmarks['left_hip'][0] + 100, landmarks['left_hip'][1]])
        if not (165 <= shoulder_angle_with_horizontal <= 195) or not (165 <= hip_angle_with_horizontal <= 195):
            errors.append("측면 불안정성")

        # 1-2. 무릎 모임 (기존 10px -> 25px)
        if front_leg == 'left':
            if landmarks['left_knee'][0] < landmarks['left_hip'][0] - 25:
                 errors.append("무릎 모임")
        else:
            if landmarks['right_knee'][0] > landmarks['right_hip'][0] + 25:
                 errors.append("무릎 모임")

        # 1-3. 과도한 무릎 전진 (기존 20px -> 35px)
        if front_leg == 'left':
            if landmarks['left_knee'][0] > landmarks['left_ankle'][0] + 35:
                errors.append("과도한 무릎 전진")
        else: # front_leg == 'right'
            if landmarks['right_knee'][0] < landmarks['right_ankle'][0] - 35:
                errors.append("과도한 무릎 전진")

        # 레벨 2: 효과성 (Effectiveness)
        # 2-1. 상체 숙여짐 (기존 15도 -> 25도 허용, 즉 각도 < 75 -> < 65)
        if 'torso' in angles and angles['torso'] < 65: 
            errors.append("상체 숙여짐")

        # 2-2. 부족한 깊이 (기존 100도 -> 115도)
        if 'front_knee' in angles and angles['front_knee'] > 115:
            errors.append("부족한 깊이")
        if 'back_knee' in angles and angles['back_knee'] > 115:
            errors.append("부족한 깊이")

        # 2-3. 좁은 스탠스 (기존 어깨너비 20% -> 15%)
        ankle_dist = abs(landmarks['left_ankle'][0] - landmarks['right_ankle'][0])
        shoulder_dist = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
        if shoulder_dist > 0 and ankle_dist < shoulder_dist * 0.15:
            errors.append("좁은 스탠스")

        # 레벨 3: 최적화 (Optimization)
        # 3-1. 앞발목 가동성 부족 (기존 80도 -> 90도)
        if 'front_ankle' in angles and angles['front_ankle'] > 90:
            errors.append("앞발목 가동성 부족")

        return errors

    # [수정] 평가 등급 기준을 "매우" 너그럽게 변경
    def get_grade_from_errors(self, errors: List[str]) -> str:
        """오류 개수에 따라 등급을 반환합니다."""
        num_errors = len(set(errors))
        if num_errors == 0: return "A"       # 완벽
        elif num_errors <= 3: return "B"     # 1-3개 오류: 좋음
        elif num_errors <= 5: return "C"     # 4-5개 오류: 보통
        elif num_errors <= 7: return "D"     # 6-7개 오류: 노력 필요
        else: return "F"                     # 8개 이상 오류

# 오류 키와 상세 설명을 매핑하는 딕셔너리
ERROR_CRITERIA_MAP = {
    "측면 불안정성": "측면 불안정성: 몸통이 옆으로 기울어지거나 골반이 떨어지는 불안정한 자세.",
    "무릎 모임": "무릎 모임 (Knee Valgus): 앞쪽 다리의 무릎이 발보다 안쪽으로 무너지는 현상.",
    "과도한 무릎 전진": "과도한 무릎 전진: 앞 무릎이 발끝보다 훨씬 앞으로 나아가는 현상.",
    "상체 숙여짐": "상체 숙여짐: 코어 안정성 부족으로 상체가 앞으로 굽혀지는 자세.",
    "부족한 깊이": "부족한 깊이: 근육을 충분히 활성화하지 못하는 얕은 런지 자세.",
    "좁은 스탠스": "좁은 스탠스 (\"외줄타기\"): 양발의 좌우 간격이 거의 없어 지지 기반이 불안정한 자세.",
    "앞발목 가동성 부족": "앞발목 가동성 부족: 최저점에서 앞발목의 배측 굴곡이 충분하지 않은 경우."
}

def save_report(report_path: str, total_reps: int, results: List[Dict]):
    """분석 결과와 전체 평가 기준을 텍스트 파일로 저장합니다."""
    grades = [res['grade'] for res in results]
    grade_counts = Counter(grades)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("런지 자세 분석 리포트\n")
        f.write("="*30 + "\n")
        f.write(f"총 런지 횟수: {total_reps}회\n\n")
        
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
                for error_key in sorted(res['errors']):
                    error_description = ERROR_CRITERIA_MAP.get(error_key, "알 수 없는 오류")
                    f.write(f"  - {error_description}\n")
            else:
                f.write("  - 모든 기준을 만족했습니다.\n")
        
        # 전체 평가 기준 추가
        f.write("\n\n" + "="*40 + "\n")
        f.write("          자세 평가 기준 (참고)\n")
        f.write("="*40 + "\n\n")
        f.write("레벨 1: 안전성 (Safety)\n")
        f.write("- 측면 불안정성: 어깨/엉덩이 선이 수평에서 ±10도 이상 벗어남\n")
        f.write("- 무릎 모임: 앞 무릎이 엉덩이-발목 선보다 안쪽으로 들어옴\n")
        f.write("- 과도한 무릎 전진: 앞 무릎이 발목보다 유의미하게 앞으로 나감\n\n")
        f.write("레벨 2: 효과성 (Effectiveness)\n")
        f.write("- 상체 숙여짐: 상체가 수직선 대비 15도 이상 기울어짐\n")
        f.write("- 부족한 깊이: 앞/뒤 무릎 각도가 100도를 넘음\n")
        f.write("- 좁은 스탠스: 발목 간격이 어깨너비의 20% 미만\n\n")
        f.write("레벨 3: 최적화 (Optimization)\n")
        f.write("- 앞발목 가동성 부족: 앞발목 각도가 80도를 넘음 (배측 굴곡 부족)\n")

    print(f"리포트가 '{report_path}'에 저장되었습니다.")


# --- 메인 실행 로직 ---
cap = cv2.VideoCapture(input_video_name)

# 영상 저장을 위한 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 변수 초기화
counter = 0 
stage = "up" # 시작 자세를 'up'으로 명확히 설정
grader = LungeGrader()
all_rep_results = []
current_rep_errors = set()
last_rep_grade = "N/A"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

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
        }
        
        # 각도 계산
        angles = {}
        left_knee_angle = calculate_angle(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle'])
        right_knee_angle = calculate_angle(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle'])
        
        # 앞다리 판단
        front_leg = 'left' if left_knee_angle < right_knee_angle else 'right'
        
        if front_leg == 'left':
            angles['front_knee'] = left_knee_angle
            angles['back_knee'] = right_knee_angle
            angles['torso'] = calculate_angle(lm_data['left_hip'], lm_data['left_shoulder'], [lm_data['left_shoulder'][0], lm_data['left_shoulder'][1] - 1])
            angles['front_ankle'] = calculate_angle(lm_data['left_knee'], lm_data['left_ankle'], lm_data['left_foot_index'])
        else:
            angles['front_knee'] = right_knee_angle
            angles['back_knee'] = left_knee_angle
            angles['torso'] = calculate_angle(lm_data['right_hip'], lm_data['right_shoulder'], [lm_data['right_shoulder'][0], lm_data['right_shoulder'][1] - 1])
            angles['front_ankle'] = calculate_angle(lm_data['right_knee'], lm_data['right_ankle'], lm_data['right_foot_index'])
        
        # [수정] 반복 횟수(카운트) 로직 개선
        # 런지 깊이가 충분할 때 (내려갔을 때)
        if (angles['front_knee'] < 100 or angles['back_knee'] < 100) and stage == 'up':
            stage = "down"
            # 새로운 랩이 시작될 때 이전 오류를 초기화
            current_rep_errors.clear()

        # 완전히 일어섰을 때
        if (angles['front_knee'] > 160 and angles['back_knee'] > 160) and stage == 'down':
            stage = "up"
            counter += 1
            # 1회 반복이 끝났으므로 최종 등급을 매기고 결과 저장
            final_grade = grader.get_grade_from_errors(list(current_rep_errors))
            all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})
            last_rep_grade = final_grade
            
        # 현재 단계(Phase) 결정
        current_phase = ""
        if stage == "up":
            current_phase = "UP"
        elif stage == "down":
            current_phase = "DOWN"
        
        # 오류 누적: 내려간 상태('down')일 때만 오류를 기록
        if stage == "down":
            errors_in_frame = grader.evaluate_errors(angles, lm_data, front_leg)
            current_rep_errors.update(errors_in_frame)

        # ------------------ 화면 표시 정보 수정 ------------------
        # 박스 높이를 150 -> 75로 변경
        cv2.rectangle(image, (0,0), (frame_width, 75), (245,117,16), -1)
        
        # REPS (Y 위치와 폰트 크기 조정)
        cv2.putText(image, 'REPS', (int(frame_width * 0.1), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter), (int(frame_width * 0.1), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # PHASE (Y 위치와 폰트 크기 조정)
        cv2.putText(image, 'PHASE', (int(frame_width * 0.4), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, current_phase, (int(frame_width * 0.35), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

        # LAST REP GRADE (Y 위치와 폰트 크기 조정)
        cv2.putText(image, 'GRADE', (int(frame_width * 0.75), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, last_rep_grade, (int(frame_width * 0.78), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        # ----------------------------------------------------

    except Exception as e:
        # print(f"오류 발생: {e}") # 디버깅 필요 시 주석 해제
        pass
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
    
    out.write(image)
    cv2.imshow('Comprehensive Lunge Analysis', image)

    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

# 영상이 끝나기 전에 마지막 랩이 'down' 상태에서 종료되었을 경우를 대비
if stage == 'down' and counter >= 0: # counter가 0일때도 1회로 포함시켜야 함
    # 랩 카운트를 하나 더해주고 저장. (영상이 중간에 끊겼을 경우)
    if not any(d['rep'] == counter + 1 for d in all_rep_results):
        counter += 1
        final_grade = grader.get_grade_from_errors(list(current_rep_errors))
        all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})


save_report(output_report_path, counter, all_rep_results)
print(f"분석 영상이 '{output_video_path}'에 저장되었습니다.")