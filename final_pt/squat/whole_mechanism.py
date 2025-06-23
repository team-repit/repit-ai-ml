import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

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
# 두 스칼라 각도(angle1, angle2)에 대해 코사인 유사도(θ 차이의 코사인) 계산
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity_scalar(angle1: float, angle2: float) -> float:
    diff_rad = (angle1 - angle2) * np.pi / 180.0
    return np.cos(diff_rad)

# ─────────────────────────────────────────────────────────────────────────────
# 코사인 유사도(sim)에 따라 A, B 등급 또는 None 반환 (C등급 이하 사용 안함)
# ─────────────────────────────────────────────────────────────────────────────
def assign_grade_from_similarity(sim: float) -> Optional[str]:
    if sim >= 0.98:
        return "A"
    elif sim >= 0.95:
        return "B"
    else:
        return None  # C, D, F 등급인 경우 None 반환

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose 초기화 (비디오 처리용)
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    smooth_landmarks=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 영상 읽기/쓰기 준비
# ─────────────────────────────────────────────────────────────────────────────
input_path  = "squat_input.mp4"
output_path = "squat_output_grade_adjusted.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"비디오를 열 수 없습니다: {input_path}")

frame_count   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps           = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0

display_duration = int(fps * 2)

# ─────────────────────────────────────────────────────────────────────────────
# 이전 두 프레임 무릎 각도(왼쪽/오른쪽) 저장용 변수
# ─────────────────────────────────────────────────────────────────────────────
left_prev2 = None
right_prev2 = None
left_prev1 = None
right_prev1 = None

# ─────────────────────────────────────────────────────────────────────────────
# Bottom 단계 감지 후 2초 동안 출력할 데이터 저장 변수
# ─────────────────────────────────────────────────────────────────────────────
display_countdown = 0
bottom_data = {
    "left": None,
    "right": None,
    "bottom": None,
    "grade": None
}

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

reference_knee_angle = 75.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark

        def get_point(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        left_hip    = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
        right_hip   = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
        left_knee   = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE])
        right_knee  = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
        left_ankle  = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
        right_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

        angle_left_curr  = calculate_angle(left_hip, left_knee, left_ankle)
        angle_right_curr = calculate_angle(right_hip, right_knee, right_ankle)
        bottom_curr = min(angle_left_curr, angle_right_curr)
    else:
        angle_left_curr = None
        angle_right_curr = None
        bottom_curr = None

    # ─────────────────────────────────────────────────────────────────────────
    # Local Bottom 단계 감지
    # ─────────────────────────────────────────────────────────────────────────
    if (
        left_prev2 is not None and right_prev2 is not None and
        left_prev1 is not None and right_prev1 is not None and
        angle_left_curr is not None and angle_right_curr is not None
    ):
        bottom_prev2 = min(left_prev2, right_prev2)
        bottom_prev1 = min(left_prev1, right_prev1)

        if bottom_prev2 > bottom_prev1 < bottom_curr:
            cos_sim = cosine_similarity_scalar(bottom_prev1, reference_knee_angle)
            grade = assign_grade_from_similarity(cos_sim)  # A, B 또는 None
            bottom_data = {
                "left": left_prev1,
                "right": right_prev1,
                "bottom": bottom_prev1,
                "grade": grade
            }
            display_countdown = display_duration

    # ─────────────────────────────────────────────────────────────────────────
    # 온전히 Local Bottom이면서 A/B 등급인 경우에만 오버레이 출력
    # ─────────────────────────────────────────────────────────────────────────
    if display_countdown > 0 and bottom_data["bottom"] is not None and bottom_data["grade"] is not None:
        overlay_text = (
            # f"L:{bottom_data['left']:.1f}°  R:{bottom_data['right']:.1f}°  "
            # f"Bottom:{bottom_data['bottom']:.1f}°  Grade:{bottom_data['grade']}"
            f"L:{bottom_data['left']:.1f}°  R:{bottom_data['right']:.1f}°  "
            f"Bottom:{bottom_data['bottom']:.1f}°  Grade:{bottom_data['grade']}"
        )

        # 스켈레톤 그리기
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        # 우측 하단, 빨간색, 폰트 크기, 두께 13
        font_face  = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.7
        font_scale = base_scale * 7
        thickness  = 13
        color      = (0, 0, 255)

        (text_width, text_height), baseline = cv2.getTextSize(
            overlay_text, font_face, font_scale, thickness
        )
        x_pos = frame_width  - text_width  - 250
        y_pos = frame_height - 20

        cv2.putText(
            frame, overlay_text, (x_pos, y_pos),
            font_face, font_scale, color, thickness, lineType=cv2.LINE_AA
        )

        display_countdown -= 1

    else:
        # 스켈레톤만 그리기 (오버레이 없을 때)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

    out.write(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # 이전 프레임 데이터 갱신
    # ─────────────────────────────────────────────────────────────────────────
    left_prev2   = left_prev1
    right_prev2  = right_prev1
    left_prev1   = angle_left_curr
    right_prev1  = angle_right_curr

cap.release()
out.release()
pose.close()

print(f"[완료] 결과 동영상 저장: {output_path}")
