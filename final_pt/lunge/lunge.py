import os
import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 0) 사용자 입력: 한 세트에 몇 번의 런지를 수행했는지
# ─────────────────────────────────────────────────────────────────────────────
while True:
    try:
        lunge_count = int(input("한 세트에 몇 번의 런지를 수행했는요? "))
        if lunge_count <= 0:
            print("1 이상의 정수를 입력해주세요.")
            continue
        break
    except ValueError:
        print("정수를 입력해주세요. 예: 3")

# ─────────────────────────────────────────────────────────────────────────────
# 1) calculate_angle 함수 (무릎 각도 계산)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    """
    A, B, C: (x, y) 픽셀 좌표 튜플
    B를 꼭짓점으로 하는 ∠ABC를 도 단위로 반환
    """
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

# ─────────────────────────────────────────────────────────────────────────────
# 2) MediaPipe Pose 초기화
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
# 3) 비디오 열기 및 “무릎 각도 + 힙 Y좌표” 수집
# ─────────────────────────────────────────────────────────────────────────────
video_path = "lunge_video.mp4"
if not os.path.exists(video_path):
    print(f"[Error] 비디오 파일이 존재하지 않습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[Error] 비디오를 열 수 없습니다: {video_path}")
    exit(1)

knee_angles = []    # 각 프레임마다 “더 굽혀진 쪽” 무릎 각도
hip_y_list    = []  # 각 프레임마다 왼쪽 힙 Y좌표 (화면 아래로 갈수록 값이 커짐)
last_valid_angle = None
last_valid_hip_y = None
frame_count = 0

# (옵션) 준비 동작(first_skip) 동안 bottom 오탐 방지를 위해 무시
first_skip = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    # 준비 동작 구간: 임시로 None 채워두거나 평균값 채움
    if frame_count <= first_skip:
        knee_angles.append(None)
        hip_y_list.append(h / 2)
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 왼쪽 힙 Y좌표
        left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
        # 오른쪽 힙 Y좌표(더 정확하게 잡기 위해 두 힙 중 더 아래로 내려간 쪽 선택)
        right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
        hip_y = max(left_hip_y, right_hip_y)
        last_valid_hip_y = hip_y
    else:
        hip_y = last_valid_hip_y if last_valid_hip_y is not None else (h / 2)

    hip_y_list.append(hip_y)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # 왼쪽 무릎 각도
        left_hip_pt   = (lm[mp_pose.PoseLandmark.LEFT_HIP].x  * w,
                         lm[mp_pose.PoseLandmark.LEFT_HIP].y  * h)
        left_knee_pt  = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                         lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
        left_ankle_pt = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                         lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)
        left_angle = calculate_angle(left_hip_pt, left_knee_pt, left_ankle_pt)

        # 오른쪽 무릎 각도
        right_hip_pt   = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x  * w,
                          lm[mp_pose.PoseLandmark.RIGHT_HIP].y  * h)
        right_knee_pt  = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)
        right_ankle_pt = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h)
        right_angle = calculate_angle(right_hip_pt, right_knee_pt, right_ankle_pt)

        # 둘 중 작은(=더 깊게 굽힌) 무릎 각도 선택
        angle = min(left_angle, right_angle)
        last_valid_angle = angle
    else:
        angle = last_valid_angle if last_valid_angle is not None else 180.0

    knee_angles.append(angle)

cap.release()
pose.close()

# first_skip 동안 None으로 채운 부분을 첫 valid 값으로 보간
for i in range(len(knee_angles)):
    if knee_angles[i] is None:
        knee_angles[i] = knee_angles[first_skip]
    else:
        break

knee_angles = np.array(knee_angles, dtype=np.float32)
hip_y_np     = np.array(hip_y_list, dtype=np.float32)
num_frames   = len(knee_angles)
if num_frames < 3:
    print("[Error] 프레임이 너무 적어 분석할 수 없습니다.")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4) 스무딩: 무릎 각도 및 힙 Y좌표에 이동 평균 적용
# ─────────────────────────────────────────────────────────────────────────────
k = 7
kernel = np.ones(k) / k
knee_smooth = np.convolve(knee_angles, kernel, mode='same')
hip_smooth  = np.convolve(hip_y_np,    kernel, mode='same')

# ─────────────────────────────────────────────────────────────────────────────
# 5) Global Threshold 계산
# ─────────────────────────────────────────────────────────────────────────────
mean_k = np.mean(knee_smooth)
std_k  = np.std(knee_smooth)
alpha_k = 0.4
knee_threshold = mean_k - std_k * alpha_k

mean_h = np.mean(hip_smooth)
std_h  = np.std(hip_smooth)
alpha_h = 0.3
hip_threshold = mean_h + std_h * alpha_h

print(f"[Debug] knee: mean={mean_k:.1f}, std={std_k:.1f}, knee_threshold={knee_threshold:.1f}")
print(f"[Debug] hip : mean={mean_h:.1f}, std={std_h:.1f}, hip_threshold={hip_threshold:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6) 로컬 최소(local‐minimum) + 절대 임계값 조건을 모두 만족하는 bottom 후보 검출
# ─────────────────────────────────────────────────────────────────────────────
bottom_candidates = []
delta = 3.0   # 앞뒤 프레임 대비 무릎각도가 얼마나 더 작아야 하는지(도 단위)

for i in range(1, num_frames - 1):
    # (1) 무릎각도 로컬 최소 조건
    is_local_min = (
        knee_smooth[i] < knee_smooth[i - 1] - delta
        and knee_smooth[i] < knee_smooth[i + 1] - delta
    )
    # (2) 무릎 각도 절대 임계값 (충분히 굽혀졌는지)
    is_knee_below_thr = (knee_smooth[i] < knee_threshold)
    # (3) 힙 Y좌표 절대 임계값 (충분히 낮아졌는지)
    is_hip_above_thr  = (hip_smooth[i] > hip_threshold)

    if is_local_min and is_knee_below_thr and is_hip_above_thr:
        bottom_candidates.append(i)

print(f"[Debug] bottom_candidates (before distance filter) = {bottom_candidates}")

# ─────────────────────────────────────────────────────────────────────────────
# 7) Minimum Distance 필터링: 하나의 런지당 1개만
# ─────────────────────────────────────────────────────────────────────────────
filtered_bottoms = []
min_distance = 30
last_sel = -min_distance
for idx in bottom_candidates:
    if idx - last_sel >= min_distance:
        filtered_bottoms.append(idx)
        last_sel = idx

print(f"[Debug] filtered_bottoms (after distance filter) = {filtered_bottoms}")

# ─────────────────────────────────────────────────────────────────────────────
# 8) 사용자 입력(lunge_count)에 따라 bottom 개수 제한
# ─────────────────────────────────────────────────────────────────────────────
if len(filtered_bottoms) < lunge_count:
    print(f"\n[Warning] bottom 후보 개수({len(filtered_bottoms)})가 충분하지 않습니다.")
    print("Fallback: 가장 작은 무릎각도 프레임을 대체 저장합니다.\n")
    # fallback으로 knee_smooth 기준 작은 순서대로 lunge_count개 선택
    sorted_idxs = np.argsort(knee_smooth)
    fallback_idxs = sorted_idxs[:lunge_count].tolist()
    selected_bottoms = fallback_idxs
else:
    selected_bottoms = filtered_bottoms[:lunge_count]

print("\n=== 최종 선택된 Bottom Frames (런지 기준, 강화됨) ===")
print(selected_bottoms)

# ─────────────────────────────────────────────────────────────────────────────
# 9) bottom 프레임 이미지를 저장
# ─────────────────────────────────────────────────────────────────────────────
save_dir = "lunge_knee_bottom_captures_refined"
os.makedirs(save_dir, exist_ok=True)

cap2 = cv2.VideoCapture(video_path)
if not cap2.isOpened():
    print(f"[Error] 두 번째 VideoCapture 열기 실패: {video_path}")
    exit(1)

for i, frame_idx in enumerate(selected_bottoms, start=1):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap2.read()
    if not ret or frame is None:
        print(f"[Warning] 프레임 읽기 실패: {frame_idx}")
        continue

    filename = f"lunge_bottom_{i:02d}_frame_{frame_idx:04d}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"[Saved] {filepath}")

cap2.release()
print("\n[Complete] 지정한 횟수만큼 런지 bottom 프레임 이미지를 저장했습니다.")
