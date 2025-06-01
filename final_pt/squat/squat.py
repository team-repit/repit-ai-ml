import os
import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 0) 사용자 입력: 한 세트에 몇 번의 스쿼트를 수행했는지
# ─────────────────────────────────────────────────────────────────────────────
while True:
    try:
        squat_count = int(input("한 세트에 몇 번의 스쿼트를 수행하셨나요? "))
        if squat_count <= 0:
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
    rad = np.arccos(cosine)
    return np.degrees(rad)

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
# 3) 비디오 열기 및 “무릎 각도” 수집
# ─────────────────────────────────────────────────────────────────────────────
video_path = "lunge_video.mp4"
if not os.path.exists(video_path):
    print(f"[Error] 비디오 파일이 존재하지 않습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[Error] 비디오를 열 수 없습니다: {video_path}")
    exit(1)

knee_angles = []      # 각 프레임의 무릎 각도
last_valid_angle = None
frame_count = 0

# (옵션) 준비 동작 중 기록은 우선 스킵하거나 일정 값으로 채울 수도 있음
first_skip = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    # 준비 동작(first_skip) 동안은 각도를 계산하지 않고 None으로 채워두거나,
    # 프레임이 정상으로 들어왔을 때 추후 보간할 수 있도록 임시 None
    if frame_count <= first_skip:
        knee_angles.append(None)
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 왼쪽 무릎 예시. 오른쪽 무릎을 쓰고 싶으면 RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE 로 바꿔주세요.
        hip    = (lm[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                  lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)
        knee   = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                  lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
        ankle  = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                  lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)

        angle = calculate_angle(hip, knee, ankle)
        last_valid_angle = angle
    else:
        # 랜드마크가 검출되지 않으면, 이전 값을 재사용하거나 180도로 채워넣을 수 있음
        if last_valid_angle is not None:
            angle = last_valid_angle
        else:
            angle = 180.0

    knee_angles.append(angle)

cap.release()
pose.close()

# None으로 채워진 첫 first_skip 프레임은 바로 앞 valid 값으로 채워 간단히 보간
for i in range(len(knee_angles)):
    if knee_angles[i] is None:
        knee_angles[i] = knee_angles[first_skip]  # first_skip 후 값으로 전부 채움
    else:
        break

knee_angles = np.array(knee_angles, dtype=np.float32)
num_frames = len(knee_angles)
if num_frames < 3:
    print("[Error] 프레임이 너무 적어 분석할 수 없습니다.")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4) 스무딩: 무릎 각도에 이동 평균 적용
# ─────────────────────────────────────────────────────────────────────────────
k = 7
kernel = np.ones(k) / k
knee_smooth = np.convolve(knee_angles, kernel, mode='same')

# ─────────────────────────────────────────────────────────────────────────────
# 5) 무릎 각도 local‐minimum 검출 (bottom)
#    (1) knee_smooth[i] < knee_smooth[i-1] - delta
#    (2) knee_smooth[i] < knee_smooth[i+1] - delta
#    (3) knee_smooth[i] < threshold_global
# ─────────────────────────────────────────────────────────────────────────────
mean_k = np.mean(knee_smooth)
std_k  = np.std(knee_smooth)
threshold_global = mean_k - std_k * 0.4  
# (무릎 각도가 평균보다 충분히 작아진 지점만 bottom 후보로)

bottom_candidates = []
delta = 3.0   # 앞뒤 프레임 대비 각도가 얼마나 더 작아야 하는지(도 단위)

for i in range(1, num_frames - 1):
    if (
        knee_smooth[i] < knee_smooth[i - 1] - delta
        and knee_smooth[i] < knee_smooth[i + 1] - delta
        and knee_smooth[i] < threshold_global
    ):
        bottom_candidates.append(i)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Minimum Distance 필터링: 한 사이클당 최대 하나만 남기기
# ─────────────────────────────────────────────────────────────────────────────
filtered_bottoms = []
min_distance = 30  # 스쿼트 하나당 최소 30프레임 간격

last_sel = -min_distance
for idx in bottom_candidates:
    if idx - last_sel >= min_distance:
        filtered_bottoms.append(idx)
        last_sel = idx

# ─────────────────────────────────────────────────────────────────────────────
# 7) 사용자 입력(squat_count)에 따라 bottom 개수 제한
# ─────────────────────────────────────────────────────────────────────────────
if len(filtered_bottoms) < squat_count:
    print(f"[Warning] 실제로 감지된 bottom 후보 개수는 {len(filtered_bottoms)}개입니다.")
    print(f"원하는 개수({squat_count})보다 적으므로, 가능한 만큼만 저장합니다.\n")
    selected_bottoms = filtered_bottoms
else:
    selected_bottoms = filtered_bottoms[:squat_count]

print("\n=== 최종 선택된 Bottom Frames (무릎 각도 기준) ===")
for i, idx in enumerate(selected_bottoms, start=1):
    print(f"  {i:2d}번째 bottom → frame {idx}")

# ─────────────────────────────────────────────────────────────────────────────
# 8) bottom 프레임 이미지를 저장
# ─────────────────────────────────────────────────────────────────────────────
save_dir = "lunge_knee_bottom_captures"
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

    filename = f"knee_bottom_{i:02d}_frame_{frame_idx:04d}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"[Saved] {filepath}")

cap2.release()
print("\n[Complete] 지정한 횟수만큼 knee‐bottom(무릎 최저점) 프레임 이미지를 저장했습니다.")
