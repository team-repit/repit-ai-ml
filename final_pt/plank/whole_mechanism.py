import os
import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    """A-B-C 세 점이 이루는 각도 계산, 꼭짓점은 B"""
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosine))

# 통일 필요?
def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    elif score >= 50:
        return "E"
    else:
        return "F"

def evaluate_posture(
                     l_shoulder, l_elbow, l_wrist,
                     r_shoulder, r_elbow, r_wrist,
                     l_hip, l_knee, r_hip, r_knee,
                     l_ear, r_ear, l_ankle, r_ankle):
    
    scores = {}

    # 1. 몸통 직선성 평가

    def segment_score(angle, target=180, weight=0.6):  
        return max(0, 100 - abs(target - angle) * weight)

    l_neck_angle = calculate_angle(l_ear, l_shoulder, l_hip)
    r_neck_angle = calculate_angle(r_ear, r_shoulder, r_hip)
    l_back_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    r_back_angle = calculate_angle(r_shoulder, r_hip, r_knee)
    l_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_leg_angle = calculate_angle(r_hip, r_knee, r_ankle)

    neck_angle = (l_neck_angle + r_neck_angle) / 2
    back_angle = (l_back_angle + r_back_angle) / 2
    leg_angle = (l_leg_angle + r_leg_angle) / 2

    neck_score = segment_score(neck_angle, target=175, weight=1.5)  
    back_score = segment_score(back_angle, target=180, weight=2.0)  
    leg_score = segment_score(leg_angle, target=180, weight=1.2)

    scores["body_straight"] = (
        neck_score * 0.3 +
        back_score * 0.5 +
        leg_score * 0.2
    )

    # 2. 팔꿈치 각도 판단

    l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2
    
    # 이상적인 팔꿈치 각도: 85-95도
    if 85 <= avg_elbow_angle <= 95:
        scores["arm_angle"] = 100
    else:
        deviation = min(abs(avg_elbow_angle - 90), 45)  
        scores["arm_angle"] = max(0, 100 - deviation * 2.5)

    # 3. 허리 과신전 판단
    l_waist_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    r_waist_angle = calculate_angle(r_shoulder, r_hip, r_knee)
    avg_waist_angle = (l_waist_angle + r_waist_angle) / 2

    if avg_waist_angle < 155:
        deviation = 155 - avg_waist_angle
        scores["waist_extension"] = max(0, 100 - deviation * 2.5) 
    elif avg_waist_angle > 195:
        deviation = avg_waist_angle - 195
        scores["waist_extension"] = max(0, 100 - deviation * 3.0)
    else:
        deviation = abs(avg_waist_angle - 175)
        scores["waist_extension"] = max(80, 100 - deviation * 1.0)  

    return scores

# mediaPipe Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5, smooth_landmarks=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

input_path  = os.path.join(base_dir, "plank_practice.mov")
output_path = os.path.join(base_dir, "plank_output_practice_improved.mp4")

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

# Main Loop

frame_count = 0
evaluation_interval = 15  # 15프레임마다 평가 (약 0.5초마다)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark

        def get_point(l): return (int(l.x * w), int(l.y * h))

        # 주요 포인트 추출
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
        l_ear = get_point(lm[mp_pose.PoseLandmark.LEFT_EAR])
        r_ear = get_point(lm[mp_pose.PoseLandmark.RIGHT_EAR])
        l_ankle = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
        r_ankle = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

        # 중심점 계산
        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) // 2,
                           (l_shoulder[1] + r_shoulder[1]) // 2)
        hip_center = ((l_hip[0] + r_hip[0]) // 2,
                      (l_hip[1] + r_hip[1]) // 2)
        knee_center = ((l_knee[0] + r_knee[0]) // 2,
                       (l_knee[1] + r_knee[1]) // 2)
        ear = ((l_ear[0] + r_ear[0]) // 2, (l_ear[1] + r_ear[1]) // 2)
        ankle = ((l_ankle[0] + r_ankle[0]) // 2, (l_ankle[1] + r_ankle[1]) // 2)


        # 거리 계산
        def euclidean(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        # 기본 조건
        arms_below = (l_elbow[1] > shoulder_center[1] and r_elbow[1] > shoulder_center[1])

        # 어깨~엉덩이 라인이 수평에 가까운가
        shoulder_hip_horizontal = abs(shoulder_center[1] - hip_center[1]) < abs(shoulder_center[0] - hip_center[0])

        # 어깨~무릎이 충분히 긴가 (앉아있는 자세 제외)
        shoulder_knee_length = euclidean(shoulder_center, knee_center)
        is_long_enough = shoulder_knee_length > 100

        # 무릎~발목이 수직에 가까운가 (서 있는 자세 제거)
        knee_ankle_diff_x = abs(knee_center[0] - ankle[0])
        knee_ankle_diff_y = abs(knee_center[1] - ankle[1])
        knee_ankle_vertical_ratio = knee_ankle_diff_y / (knee_ankle_diff_x + 1e-5)
        is_leg_horizontal = knee_ankle_vertical_ratio < 2.5  # 다리 뻗음

        # 전체 조건
        is_plank_pose = (
            arms_below and
            shoulder_hip_horizontal and
            is_long_enough and
            is_leg_horizontal
        )

        # 일정 간격으로만 평가 수행 (성능 최적화)
        if is_plank_pose and frame_count % evaluation_interval == 0:
            scores = evaluate_posture(
                                    l_shoulder, l_elbow, l_wrist,
                                    r_shoulder, r_elbow, r_wrist,
                                    l_hip, l_knee, r_hip, r_knee, l_ear, r_ear, l_ankle, r_ankle)

            avg_score = np.mean(list(scores.values()))
            grade = get_grade(avg_score)

            overlay_data = {
                "grade": grade,
                "avg_score": avg_score,
                "scores": scores
            }
            display_countdown = display_duration

        # 포즈 랜드마크 그리기
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        # 플랭크 자세 여부 표시
        pose_text = "PLANK DETECTED" if is_plank_pose else "NOT PLANK POSE"
        color = (0, 255, 0) if is_plank_pose else (0, 0, 255)
        cv2.putText(frame, pose_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 평가 결과 오버레이
    if display_countdown > 0 and overlay_data:
        # 등급과 총점 표시
        grade_text = f"Grade: {overlay_data['grade']} ({overlay_data['avg_score']:.1f})"
        cv2.putText(frame, grade_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # 세부 점수 표시
        y_offset = 140
        for item, score in overlay_data['scores'].items():
            detail_text = f"{item}: {score:.0f}"
            cv2.putText(frame, detail_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            y_offset += 30
        
        display_countdown -= 1

    out.write(frame)

cap.release()
out.release()
pose.close()
print(f"[완료] 개선된 결과 동영상 저장: {output_path}")
