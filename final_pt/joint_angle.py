import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수: 세 점(A, B, C)의 각도(∠ABC)를 계산
# A, B, C는 (x, y) 튜플 또는 리스트 (픽셀 좌표)
# 반환값은 도 단위 각도 (0° ~ 180°)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose 초기화
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,       # 정적 이미지 처리 모드
    model_complexity=1,           
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# ─────────────────────────────────────────────────────────────────────────────
# 이미지 불러오기 (경로에 맞게 수정)
# ─────────────────────────────────────────────────────────────────────────────
image_path = "squat/다운로드.jpeg"  # 분석할 이미지 파일 경로
image = cv2.imread(image_path)
if image is None:
    raise IOError(f"이미지를 불러올 수 없습니다: {image_path}")

# BGR → RGB 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ─────────────────────────────────────────────────────────────────────────────
# Pose 추론
# ─────────────────────────────────────────────────────────────────────────────
results = pose.process(image_rgb)
if not results.pose_landmarks:
    print("PoseLandmark를 검출하지 못했습니다.")
    pose.close()
    exit(0)

# 랜드마크 좌표를 픽셀로 변환
h, w, _ = image.shape
lm = results.pose_landmarks.landmark

# 주요 관절 좌표 추출 (왼쪽/오른쪽)
def get_point(landmark):
    return (int(landmark.x * w), int(landmark.y * h))

# 왼쪽 및 오른쪽 랜드마크
left_shoulder  = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
right_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
left_elbow     = get_point(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
right_elbow    = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
left_wrist     = get_point(lm[mp_pose.PoseLandmark.LEFT_WRIST])
right_wrist    = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
left_hip       = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
right_hip      = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
left_knee      = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE])
right_knee     = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
left_ankle     = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
right_ankle    = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

# ─────────────────────────────────────────────────────────────────────────────
# 관절 각도 계산 예시
#   - 왼쪽 팔꿈치(어깨-팔꿈치-손목)
#   - 오른쪽 팔꿈치(어깨-팔꿈치-손목)
#   - 왼쪽 무릎(엉덩이-무릎-발목)
#   - 오른쪽 무릎(엉덩이-무릎-발목)
#   - 왼쪽 어깨(팔꿈치-어깨-엉덩이)
#   - 오른쪽 어깨(팔꿈치-어깨-엉덩이)
#   - 왼쪽 엉덩이(어깨-엉덩이-무릎)
#   - 오른쪽 엉덩이(어깨-엉덩이-무릎)
# ─────────────────────────────────────────────────────────────────────────────
angles = {}

# 팔꿈치 각도
angles['left_elbow']  = calculate_angle(left_shoulder, left_elbow, left_wrist)
angles['right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)

# 무릎 각도
angles['left_knee']   = calculate_angle(left_hip, left_knee, left_ankle)
angles['right_knee']  = calculate_angle(right_hip, right_knee, right_ankle)

# 어깨 각도
angles['left_shoulder']  = calculate_angle(left_elbow, left_shoulder, left_hip)
angles['right_shoulder'] = calculate_angle(right_elbow, right_shoulder, right_hip)

# 엉덩이(힙) 각도
angles['left_hip']   = calculate_angle(left_shoulder, left_hip, left_knee)
angles['right_hip']  = calculate_angle(right_shoulder, right_hip, right_knee)

# ─────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────────────────────────
print("==== 관절 각도 ====")
for joint, angle in angles.items():
    print(f"{joint:15s}: {angle:6.2f}°")

# ─────────────────────────────────────────────────────────────────────────────
# (선택) 이미지 위에 랜드마크와 각도를 시각화하는 예시
# ─────────────────────────────────────────────────────────────────────────────
# 랜드마크 그리기
annotated_image = image.copy()
mp.solutions.drawing_utils.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
)

# 각 관절 점에 각도 텍스트 추가
def put_angle_text(img, point, text):
    cv2.putText(img, text, (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

put_angle_text(annotated_image, left_elbow,  f"{angles['left_elbow']:.1f}°")
put_angle_text(annotated_image, right_elbow, f"{angles['right_elbow']:.1f}°")
put_angle_text(annotated_image, left_knee,   f"{angles['left_knee']:.1f}°")
put_angle_text(annotated_image, right_knee,  f"{angles['right_knee']:.1f}°")
put_angle_text(annotated_image, left_shoulder,  f"{angles['left_shoulder']:.1f}°")
put_angle_text(annotated_image, right_shoulder, f"{angles['right_shoulder']:.1f}°")
put_angle_text(annotated_image, left_hip,    f"{angles['left_hip']:.1f}°")
put_angle_text(annotated_image, right_hip,   f"{angles['right_hip']:.1f}°")

# 결과 이미지 저장
output_path = "output_with_angles.jpg"
cv2.imwrite(output_path, annotated_image)
print(f"\n[Saved] 관절 각도 시각화 이미지: {output_path}")