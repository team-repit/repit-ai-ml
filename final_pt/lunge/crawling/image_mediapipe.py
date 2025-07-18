import cv2
import mediapipe as mp
import numpy as np
import os

# --- 초기 설정 ---
# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 디렉토리 설정 ---
# 이미지가 있는 입력 폴더
INPUT_DIR = '/Users/pado/Library/Mobile Documents/com~apple~CloudDocs/HIU/CS/4-1/졸프/github/final_pt/lunge/crawling/lunge_images/handmade'
# 결과 이미지를 저장할 출력 폴더
OUTPUT_DIR = 'lunge_analysis_results'

# 출력 폴더가 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산하는 함수 (b가 꼭짓점)"""
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 꼭짓점
    c = np.array(c)  # 세 번째 점
    
    # 각 벡터 계산
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # 각도가 180도를 넘어가면 360에서 빼서 작은 각을 구함
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- 이미지 처리 루프 ---
# 입력 디렉토리에 있는 모든 파일 목록을 가져옴
file_list = os.listdir(INPUT_DIR)

for file_name in file_list:
    # 파일 확장자가 이미지 형식인지 확인 (jpg, jpeg, png)
    if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue # 이미지 파일이 아니면 건너뜀

    print(f"--- '{file_name}' 파일 처리 중... ---")
    
    # 이미지 파일 경로 생성
    image_path = os.path.join(INPUT_DIR, file_name)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"오류: '{file_name}' 파일을 로드할 수 없습니다.")
        continue # 다음 파일로 넘어감

    # BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 자세 추정 수행
    results = pose.process(image_rgb)
    
    # 결과 이미지 복사
    annotated_image = image.copy()

    # 관절점(랜드마크)이 감지되었는지 확인
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 런지 자세 평가를 위한 주요 관절점 추출 (왼쪽 다리 기준)
        # 이미지 속 인물이 왼쪽 다리를 앞으로 내밀었다고 가정합니다.
        # 만약 오른쪽 다리가 기준이라면 'LEFT'를 'RIGHT'로 변경하세요.
        try:
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # 무릎 각도 계산
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # --- 등급 분류 기준 ---
            lunge_grade = ''
            if 85 <= knee_angle <= 100:
                lunge_grade = 'Good'
                color = (0, 255, 0) # 초록색
            elif knee_angle < 85:
                lunge_grade = 'Bad: Knee too bent'
                color = (0, 0, 255) # 빨간색
            else: # knee_angle > 100
                lunge_grade = 'Bad: Knee not bent enough'
                color = (0, 165, 255) # 주황색

            # 결과 시각화
            cv2.putText(annotated_image, f"Knee Angle: {int(knee_angle)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Grade: {lunge_grade}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                        
            # 감지된 관절 그리기
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        except Exception as e:
            print(f"오류: 주요 관절점을 감지하지 못했습니다. - {e}")
            cv2.putText(annotated_image, "Pose detection failed", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        print("이미지에서 자세를 감지하지 못했습니다.")
        cv2.putText(annotated_image, "Could not detect pose", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 결과 이미지 저장
    # 원본 파일명에 '_result'를 붙여서 저장
    base_name = os.path.splitext(file_name)[0]
    output_filename = f"{base_name}_result.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, annotated_image)
    print(f"분석 완료! '{output_path}' 파일로 결과가 저장되었습니다.\n")

print("모든 이미지 처리가 완료되었습니다.")