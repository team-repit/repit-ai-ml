import cv2
import mediapipe as mp
import numpy as np
import os
import shutil # 파일 이동을 위해 shutil 모듈 추가

# --- MediaPipe 초기 설정 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- 디렉토리 설정 ---
INPUT_DIR = '/Users/pado/Library/Mobile Documents/com~apple~CloudDocs/HIU/CS/4-1/졸프/github/final_pt/lunge/crawling/lunge_images/handmade'  # 원본 이미지 폴더 (절대 경로 권장)
OUTPUT_DIR = 'lunge_classified' # 분류된 이미지가 저장될 상위 폴더

# --- 등급별 폴더 생성 ---
grades = ['A', 'B', 'C', 'D', 'F', 'NA'] # NA: Not Available (감지 실패)
for grade_folder in grades:
    # exist_ok=True 옵션은 폴더가 이미 있어도 에러를 발생시키지 않음
    os.makedirs(os.path.join(OUTPUT_DIR, grade_folder), exist_ok=True)


def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산 (b가 꼭짓점)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def classify_lunge(landmarks):
    """스켈레톤 데이터를 기반으로 런지 자세를 평가하고 등급을 반환합니다."""
    # 주요 랜드마크 추출 (왼쪽 다리 기준)
    try:
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    except:
        return 'F' # 주요 랜드마크 감지 실패 시 F등급 처리

    scores = []

    # 1. 앞쪽 무릎 각도 평가
    front_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    if 85 <= front_knee_angle <= 100:
        scores.append("Good")
    elif 75 <= front_knee_angle < 85 or 100 < front_knee_angle <= 110:
        scores.append("Okay")
    else:
        scores.append("Bad")

    # 2. 상체 기울기 평가 (어깨-엉덩이-무릎 각도)
    torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    if torso_angle > 145:
        scores.append("Good")
    elif 130 <= torso_angle <= 145:
        scores.append("Okay")
    else:
        scores.append("Bad")

    # 3. 앞쪽 정강이 수직 평가 (수직선과 정강이의 각도)
    vertical_point = (left_ankle[0], left_ankle[1] - 1)
    shin_angle = calculate_angle(vertical_point, left_ankle, left_knee)
    if shin_angle <= 10:
        scores.append("Good")
    elif 10 < shin_angle <= 20:
        scores.append("Okay")
    else:
        scores.append("Bad")

    # 최종 등급 산정
    bad_count = scores.count("Bad")
    okay_count = scores.count("Okay")

    if bad_count >= 2:
        grade = 'F'
    elif bad_count == 1:
        grade = 'D'
    elif okay_count >= 2:
        grade = 'C'
    elif okay_count == 1:
        grade = 'B'
    else: # All "Good"
        grade = 'A'

    return grade


# --- 이미지 처리 및 분류 루프 ---
# INPUT_DIR에 파일이 없으면 오류가 나므로, 존재 여부 확인
if not os.path.isdir(INPUT_DIR):
    print(f"오류: 입력 폴더 '{INPUT_DIR}'를 찾을 수 없습니다.")
    print("스크립트를 종료합니다.")
else:
    for file_name in os.listdir(INPUT_DIR):
        # 이미지 파일만 대상으로 함
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        source_path = os.path.join(INPUT_DIR, file_name)
        
        # 이미지를 읽고 자세 추정
        image = cv2.imread(source_path)
        if image is None:
            print(f"'{file_name}' 파일을 읽을 수 없습니다.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        grade = 'NA' # 기본 등급은 NA(감지 실패)
        
        # 랜드마크가 감지된 경우 등급 분류 수행
        if results.pose_landmarks:
            grade = classify_lunge(results.pose_landmarks.landmark)

        # 결정된 등급의 폴더로 원본 이미지 이동
        destination_path = os.path.join(OUTPUT_DIR, grade, file_name)
        shutil.move(source_path, destination_path)
        
        print(f"'{file_name}' 파일을 '{grade}' 등급으로 분류했습니다.")

    print(f"\n모든 이미지 분류가 완료되었습니다.")
    print(f"결과는 '{OUTPUT_DIR}' 폴더에 등급별로 저장되었습니다.")