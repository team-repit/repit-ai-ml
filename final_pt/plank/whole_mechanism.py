import os
import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    # A-B-C 세 점이 이루는 각도 계산, 꼭짓점은 B! -> BA 벡터와 BC 벡터 사이 각도

    BA = np.array(A) - np.array(B) # 두 벡터 기준으로! numpy로 각도! 구하기. BA = 벡터 B → A
    BC = np.array(C) - np.array(B) # BC = 벡터 B → C

    # 이때 두 벡터 사이의 각도를 구하는 공식은 코사인 유사도 공식을 사용!!
    # 벡터 내적 (dot product), 벡터의 크기 (norm = 유클리드 거리)

    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    # cosine = (BA ⋅ BC) / (|BA| * |BC|)
    cosine = np.clip(cosine, -1.0, 1.0)
    # 계산 오차로 인해 cosine 값이 -1.0001 같은 수가 되는 걸 방지

    return np.degrees(np.arccos(cosine))
    # 벡터 간의 코사인 유사도 -> np.arccos()로 라디안 값을 도로 변환

def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D" # 점수를 수정 해야 할까?


# 플랭크 자세에서 5가지 항목을 평가해 점수화. (몸통 직선성, 팔꿈치 각도, 엉덩이 높이, 팔 위치, 허리 과신전 여부)
def evaluate_posture(shoulder_center, hip_center, knee_center,
                     l_shoulder, l_elbow, l_wrist,
                     r_shoulder, r_elbow, r_wrist,
                     l_hip, l_knee, r_hip, r_knee, arms_below_body):
    scores = {}

    # 몸통 직선성 판단 (어깨 -> 엉덩이 -> 무릎)
    body_angle = calculate_angle(shoulder_center, hip_center, knee_center)
    scores["body_straight"] = max(0, 100 - abs(180 - body_angle) * 2) # 실제 각도(body_angle)가 180도에서 얼마나 벗어났는지. (예: 170도 → 오차 10 → 감점 20 → 점수 80)

    # 팔꿈치 각도 판단 (어깨 -> 팔꿈치 -> 손목)
    l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    if 85 <= l_elbow_angle <= 90 or 85 <= r_elbow_angle <= 95:
        scores["arm_angle"] = 100 # 기준 각도인 85~95도(+-5도) 사이라면 100점
    else:
        deviation = min(abs(l_elbow_angle - 90), abs(r_elbow_angle - 90))
        scores["arm_angle"] = max(0, 100 - deviation * 2)  # 90도 기준으로 얼마나 벗어났는지에 따라 감점

    # 엉덩이 높이
    hip_offset = hip_center[1] - shoulder_center[1]  # 아래로 내려갈수록 +값
    body_height = abs(knee_center[1] - shoulder_center[1])

    if body_height == 0:
        hip_score = 0
    else:
        ratio = hip_offset / body_height  # 음수면 엉덩이가 어깨보다 위
        # 0.33보다 작으면 너무 올라감, 0.5쯤이 이상적, 0.7보다 크면 너무 쳐짐
        if 0.3 <= ratio <= 0.7:
            hip_score = 100
        else:
            deviation = min(abs(ratio - 0.5), 0.5)  # 0.5를 기준으로 벗어난 정도
            hip_score = max(0, 100 - deviation * 200)  # 감점 비율 조절 가능
    scores["hip_level"] = hip_score


    # 팔 위치 (팔꿈치가 어깨보다 아래에 위치하면 정상적인 플랭크 → 100점)
    scores["arm_position"] = 100 if arms_below_body else 30

    # 허리 각도 (과신전 판단)
    l_waist_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    r_waist_angle = calculate_angle(r_shoulder, r_hip, r_knee)
    avg_waist_angle = (l_waist_angle + r_waist_angle) / 2

    if avg_waist_angle > 195:
        # 허리가 너무 꺾인 경우 감점 (과신전)
        scores["waist_bend"] = max(0, 100 - (avg_waist_angle - 180) * 3)
    else:
        scores["waist_bend"] = 100

    # 엉덩이 위치 평가 (힙업 or 쳐짐)
    hip_y_diff = abs(hip_center[1] - shoulder_center[1])
    body_height = abs(knee_center[1] - shoulder_center[1])
    hip_ratio = hip_y_diff / body_height if body_height > 0 else 1.0

    if avg_waist_angle <= 195:
        # 허리는 괜찮은데 엉덩이 위치가 비정상적인 경우
        if hip_ratio < 0.3:  # 엉덩이가 너무 올라간 경우 (힙업)
            scores["hip_level"] = max(0, 100 - (0.3 - hip_ratio) * 200)
        elif hip_ratio > 0.7:  # 엉덩이가 너무 내려간 경우 (처짐)
            scores["hip_level"] = max(0, 100 - (hip_ratio - 0.7) * 300)
        else:
            scores["hip_level"] = 100
    else:
        # 허리가 꺾인 경우는 이미 감점됐으니 hip_level은 따로 안 감점
        scores["hip_level"] = 100


    return scores # 각 항목에 대해 0~100점 점수를 담은 딕셔너리 반환

# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5, smooth_landmarks=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

input_path  = os.path.join(base_dir, "plank_perfact.mov")
output_path = os.path.join(base_dir, "plank_output_evaluated.mp4")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"비디오를 열 수 없습니다: {input_path}")

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0

display_duration = int(fps * 2)
display_countdown = 0
overlay_data = None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ─────────────────────────────────────────────────────────────────────────────
# Main Loop (영상 프레임을 하나씩 처리하며 MediaPipe로 사람 자세를 분석)
# ─────────────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read() # 프레임 가져오기
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark

        # 포즈 랜드마크 추출 후 어깨, 엉덩이, 무릎 등 주요 포인트 좌표 계산

        def get_point(l): return (int(l.x * w), int(l.y * h))

        l_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
        r_shoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        l_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP])
        r_hip = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
        l_knee = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE])
        r_knee = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
        l_elbow = get_point(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
        r_elbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
        l_wrist = get_point(lm[mp_pose.PoseLandmark.LEFT_WRIST])
        r_wrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST])

        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) // 2,
                           (l_shoulder[1] + r_shoulder[1]) // 2)
        hip_center = ((l_hip[0] + r_hip[0]) // 2,
                      (l_hip[1] + r_hip[1]) // 2)
        knee_center = ((l_knee[0] + r_knee[0]) // 2,
                       (l_knee[1] + r_knee[1]) // 2)

        arms_below = (l_elbow[1] > shoulder_center[1] and r_elbow[1] > shoulder_center[1])

        # 플랭크 조건이 만족되면 평가
        is_plank_pose = (
            arms_below and
            hip_center[1] > shoulder_center[1] and
            abs(knee_center[1] - shoulder_center[1]) > 50  # 몸통 높이 확보
        )

        if is_plank_pose:
            # 플랭크 자세일 경우만 평가 진행
            scores = evaluate_posture(shoulder_center, hip_center, knee_center,
                                    l_shoulder, l_elbow, l_wrist,
                                    r_shoulder, r_elbow, r_wrist,
                                    l_hip, l_knee, r_hip, r_knee, arms_below)

            avg_score = np.mean(list(scores.values()))
            grade = get_grade(avg_score)

            # A ~ F 어떤 등급이든
            overlay_data = {"grade": grade, "scores": scores}
            display_countdown = display_duration
        else:
            # 플랭크 자세가 아니면 아무것도 띄우지 않음
            overlay_data = None

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    # Overlay 출력
    if display_countdown > 0 and overlay_data:
        text = f"Grade: {overlay_data['grade']} | " + \
               " ".join([f"{k}:{v:.0f}" for k, v in overlay_data['scores'].items()])

        cv2.putText(frame, text, (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
        display_countdown -= 1

    out.write(frame)

cap.release()
out.release()
pose.close()
print(f"[완료] 결과 동영상 저장: {output_path}")
