import os
import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수: 세 점(A, B, C)의 각도(∠ABC)를 계산
# A, B, C: (x, y) 픽셀 좌표 튜플 또는 리스트
# 반환값은 도 단위 각도 (0° ~ 180°)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

# ─────────────────────────────────────────────────────────────────────────────
# 1) MediaPipe Pose 초기화
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) 비디오 열기
# ─────────────────────────────────────────────────────────────────────────────
video_path = "plank_video.mp4"  # 실제 파일명/경로로 바꿔주세요
if not os.path.exists(video_path):
    print(f"[Error] 비디오 파일이 존재하지 않습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[Error] 비디오를 열 수 없습니다: {video_path}")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS)
print(f"비디오 열림: 총 {frame_count}프레임, FPS={fps:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) 플랭크 조건 파라미터
# ─────────────────────────────────────────────────────────────────────────────
HIP_ANGLE_TH      = 160.0    # 엉덩이 각도 (어깨-엉덩이-무릎) ≥ 이 값 → 몸통이 곧게 펴진 상태
SHOULDER_ANGLE_TH = 160.0    # 어깨 각도 (팔꿈치-어깨-엉덩이) ≥ 이 값 → 상체가 일직선으로 펴진 상태

# ─────────────────────────────────────────────────────────────────────────────
# 4) 각 프레임마다 플랭크 여부 판단
#    results_list: [(frame_idx, is_plank_bool), ...]
# ─────────────────────────────────────────────────────────────────────────────
results_list = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(image_rgb)
    is_plank  = False

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm       = results.pose_landmarks.landmark

        # 1) 엉덩이 각도 계산: 어깨-엉덩이-무릎
        shoulder_pt = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                       lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        hip_pt      = (lm[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                       lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)
        knee_pt     = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                       lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
        hip_angle   = calculate_angle(shoulder_pt, hip_pt, knee_pt)

        # 2) 어깨 각도 계산: 팔꿈치-어깨-엉덩이
        elbow_pt  = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                     lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)
        shoulder2 = shoulder_pt  # 이미 정의됨
        hip2      = hip_pt
        shoulder_angle = calculate_angle(elbow_pt, shoulder2, hip2)

        # 3) 플랭크 판단: hip_angle 및 shoulder_angle 모두 임계값 이상일 때
        if hip_angle >= HIP_ANGLE_TH and shoulder_angle >= SHOULDER_ANGLE_TH:
            is_plank = True

        # (참고) 오른쪽 측면을 기준으로도 똑같이 계산해서
        # left/right 둘 중 하나만 충족해도 플랭크로 볼 수도 있습니다.
        # 예:
        #    r_shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
        #                  lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        #    r_hip      = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
        #                  lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h)
        #    r_knee     = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
        #                  lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)
        #    hip_angle_r = calculate_angle(r_shoulder, r_hip, r_knee)
        #    r_elbow   = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
        #                 lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)
        #    shoulder_angle_r = calculate_angle(r_elbow, r_shoulder, r_hip)
        #    if hip_angle_r >= HIP_ANGLE_TH and shoulder_angle_r >= SHOULDER_ANGLE_TH:
        #        is_plank = True

    # else: 포즈가 검출되지 않으면 is_plank=False 으로 둠

    results_list.append((frame_idx, is_plank))
    frame_idx += 1

cap.release()
pose.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5) 연속된 플랭크 구간(segment) 묶기
#    예: [(start_frame, end_frame), ...]
# ─────────────────────────────────────────────────────────────────────────────
segments = []
curr_state = False  # 현재 플랭크 상태(True/False)
seg_start  = None

for idx, is_plank in results_list:
    if is_plank and not curr_state:
        # False → True 전환: segment 시작
        seg_start = idx
        curr_state = True
    elif not is_plank and curr_state:
        # True → False 전환: segment 종료
        segments.append((seg_start, idx - 1))
        seg_start = None
        curr_state = False

# 마지막에 still in plank 상태라면
if curr_state and seg_start is not None:
    segments.append((seg_start, len(results_list) - 1))

print("\n=== Detected Plank Segments ===")
for s, e in segments:
    print(f"  frames {s} ~ {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6) 각 플랭크 구간의 “첫 프레임”만 이미지로 저장
#    (원한다면 구간의 중앙 프레임을 택해도 됩니다: mid = (s+e)//2)
# ─────────────────────────────────────────────────────────────────────────────
save_dir = "plank_captures"
os.makedirs(save_dir, exist_ok=True)

cap2 = cv2.VideoCapture(video_path)
if not cap2.isOpened():
    print(f"[Error] 두 번째 VideoCapture 열기 실패: {video_path}")
    exit(1)

for i, (start_f, end_f) in enumerate(segments, start=1):
    capture_frame = start_f  # 원하는 프레임을 변경할 수 있음 (예: (start_f+end_f)//2)

    cap2.set(cv2.CAP_PROP_POS_FRAMES, capture_frame)
    ret, frame = cap2.read()
    if not ret or frame is None:
        print(f"[Warning] 프레임 읽기 실패: {capture_frame}")
        continue

    filename = f"plank_{i:02d}_frame_{capture_frame:04d}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"[Saved] {filepath}")

cap2.release()
print("\n[Complete] 플랭크 자세 프레임을 저장했습니다.")
