import cv2
import mediapipe as mp
import numpy as np
import os
import time  # time 모듈 추가
from typing import List, Dict, Tuple
from collections import Counter

# --- 경로 설정 ---
# 실시간 카메라 입력이므로 input_video_name은 사용되지 않음.
# 출력 파일은 타임스탬프를 사용하여 자동 생성되므로 기본 이름 설정 필요 없음.

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
        f.write("- 측면 불안정성: 어깨/엉덩이 선이 수평에서 ±15도 이상 벗어남\n") # 수정된 기준 반영
        f.write("- 무릎 모임: 앞 무릎이 엉덩이-발목 선보다 안쪽으로 25px 이상 들어옴\n") # 수정된 기준 반영
        f.write("- 과도한 무릎 전진: 앞 무릎이 발목보다 35px 이상 앞으로 나감\n\n") # 수정된 기준 반영
        f.write("레벨 2: 효과성 (Effectiveness)\n")
        f.write("- 상체 숙여짐: 상체가 수직선 대비 25도 이상 기울어짐 (각도 65도 미만)\n") # 수정된 기준 반영
        f.write("- 부족한 깊이: 앞/뒤 무릎 각도가 115도를 넘음\n") # 수정된 기준 반영
        f.write("- 좁은 스탠스: 발목 간격이 어깨너비의 15% 미만\n\n") # 수정된 기준 반영
        f.write("레벨 3: 최적화 (Optimization)\n")
        f.write("- 앞발목 가동성 부족: 앞발목 각도가 90도를 넘음 (배측 굴곡 부족)\n") # 수정된 기준 반영

    print(f"리포트가 '{report_path}'에 저장되었습니다.")


# --- Main function for real-time lunge analysis ---
def main_lunge_analysis():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)  # 기본 카메라 (보통 내장 웹캠)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 카메라가 연결되어 있는지 확인하거나 다른 카메라 인덱스를 시도해보세요.")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 영상 저장을 위한 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 실제 카메라의 FPS를 가져옴
    if fps == 0: # 일부 카메라에서 FPS가 0으로 반환될 경우 기본값 설정
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 코덱
    
    # 타임스탬프를 포함한 파일명 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_video_path = f"lunge_realtime_analysis_{timestamp}.mp4"
    output_report_path = f"lunge_realtime_report_{timestamp}.txt"
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 변수 초기화
    counter = 0 
    stage = None # 'up' 또는 'down'
    grader = LungeGrader() # LungeGrader 인스턴스 사용
    all_rep_results = []
    current_rep_errors = set()
    last_rep_grade = "N/A"
    
    # 타이머 설정
    start_time = time.time()
    recording_duration = 60  # 예를 들어 60초 (1분)
    
    print("런지 분석을 시작합니다. 약 {}초간 카메라가 켜집니다.".format(recording_duration))
    print("런지 동작을 시작하세요!")
    print("종료하려면 'q'를 누르세요.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break
        
        # 현재 시간 계산
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, recording_duration - elapsed_time)
        
        # 설정된 시간 경과 시 종료
        if elapsed_time >= recording_duration:
            print(f"설정된 녹화 시간({recording_duration}초)이 경과하여 종료합니다.")
            break
        
        # 이미지 처리 (BGR to RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, c = image.shape
        lm_data = {}
        
        try:
            landmarks = results.pose_landmarks.landmark
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
            
            # 앞다리 판단 (더 작은 무릎 각도를 가진 다리가 앞다리로 가정)
            front_leg = 'left' if left_knee_angle < right_knee_angle else 'right'
            
            if front_leg == 'left':
                angles['front_knee'] = left_knee_angle
                angles['back_knee'] = right_knee_angle
                # 상체 각도는 수직선 대비
                angles['torso'] = calculate_angle(lm_data['left_hip'], lm_data['left_shoulder'], [lm_data['left_shoulder'][0], lm_data['left_shoulder'][1] - 100]) # Y-100으로 수직 위를 가리킴
                angles['front_ankle'] = calculate_angle(lm_data['left_knee'], lm_data['left_ankle'], lm_data['left_foot_index'])
            else:
                angles['front_knee'] = right_knee_angle
                angles['back_knee'] = left_knee_angle
                # 상체 각도는 수직선 대비
                angles['torso'] = calculate_angle(lm_data['right_hip'], lm_data['right_shoulder'], [lm_data['right_shoulder'][0], lm_data['right_shoulder'][1] - 100]) # Y-100으로 수직 위를 가리킴
                angles['front_ankle'] = calculate_angle(lm_data['right_knee'], lm_data['right_ankle'], lm_data['right_foot_index'])
            
            # 런지 반복 횟수(카운트) 로직 개선
            # 런지 깊이가 충분할 때 (내려갔을 때)
            # 앞무릎 또는 뒷무릎 중 하나가 100도 미만으로 굽혀지면 "down" 상태
            if (angles['front_knee'] < 100 or angles['back_knee'] < 100) and stage == 'up':
                stage = "down"
                current_rep_errors.clear() # 새로운 랩 시작 시 이전 오류 초기화

            # 완전히 일어섰을 때
            # 앞무릎과 뒷무릎 모두 160도 초과로 펴지면 "up" 상태
            if (angles['front_knee'] > 160 and angles['back_knee'] > 160) and stage == 'down':
                stage = "up"
                counter += 1
                # 1회 반복이 끝났으므로 최종 등급을 매기고 결과 저장
                final_grade = grader.get_grade_from_errors(list(current_rep_errors))
                all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})
                last_rep_grade = final_grade
                
            # 현재 단계(Phase) 결정
            current_phase = ""
            if stage is None: # 초기 상태
                current_phase = "READY"
            elif stage == "up":
                current_phase = "UP"
            elif stage == "down":
                current_phase = "DOWN"
            
            # 오류 누적: 내려간 상태('down')일 때만 오류를 기록
            if stage == "down":
                errors_in_frame = grader.evaluate_errors(angles, lm_data, front_leg)
                current_rep_errors.update(errors_in_frame)

            # ------------------ 화면 표시 정보 ------------------
            # 상단 정보 박스
            cv2.rectangle(image, (0,0), (frame_width, 75), (245,117,16), -1)
            
            # REPS
            cv2.putText(image, 'REPS', (int(frame_width * 0.1), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(counter), (int(frame_width * 0.1), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
            
            # PHASE
            cv2.putText(image, 'PHASE', (int(frame_width * 0.4), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, current_phase, (int(frame_width * 0.35), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

            # LAST REP GRADE
            cv2.putText(image, 'GRADE', (int(frame_width * 0.75), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, last_rep_grade, (int(frame_width * 0.78), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

            # 남은 시간 표시
            time_text = f"Time: {int(remaining_time)}s"
            cv2.putText(image, time_text, (frame_width - 200, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
            
        except Exception as e:
            # 랜드마크가 감지되지 않을 때 (처음 프레임 등) 오류를 무시하고 진행
            # print(f"랜드마크 처리 중 오류 발생: {e}") # 디버깅 필요 시 주석 해제
            cv2.putText(image, "No person detected. Stand clearly in frame.", (50, frame_height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(image) # 비디오 파일로 프레임 저장
        cv2.imshow('Comprehensive Lunge Analysis (Realtime)', image) # 화면에 표시

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            print("사용자 요청으로 분석을 종료합니다.")
            break

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 마지막 랩 처리 (영상이 중간에 끊겼을 경우)
    # 현재 `down` 상태이고, 아직 최종 결과에 포함되지 않은 랩이 있다면 추가
    if stage == 'down' and (not all_rep_results or all_rep_results[-1]['rep'] != counter + 1):
        if counter == 0 and len(current_rep_errors) > 0: # 1회도 완료 못했지만 오류가 있는 경우
             counter += 1 # 0회 -> 1회로 간주
        elif counter > 0 and len(current_rep_errors) > 0: # 이미 카운트가 올라갔지만 현재 랩의 오류가 남아있을 경우
            counter += 1 # 해당 랩을 한 번 더 더해줌
        
        # 마지막 랩이 'down' 상태에서 종료되었을 때만 처리
        if len(current_rep_errors) > 0: # 오류가 하나라도 있었다면 그 랩을 평가에 포함
            final_grade = grader.get_grade_from_errors(list(current_rep_errors))
            all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})
        elif counter > 0 and not any(d['rep'] == counter for d in all_rep_results): # 카운트는 있으나 오류가 없는 랩이 누락된 경우
            final_grade = grader.get_grade_from_errors([]) # 오류 없으므로 'A' 등급
            all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': []})
        
    save_report(output_report_path, counter, all_rep_results)
    print(f"분석 영상이 '{output_video_path}'에 저장되었습니다.")

# 스크립트 실행 시 메인 함수 호출
if __name__ == "__main__":
    main_lunge_analysis()