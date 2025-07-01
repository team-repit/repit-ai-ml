import os
import cv2
import mediapipe as mp
import numpy as np

def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"
    
# 부위별 점수 계산
def evaluate_posture(shoulder_center, hip_center, knee_center,
                     l_shoulder, l_elbow, l_wrist,
                     r_shoulder, r_elbow, r_wrist,
                     l_hip, l_knee,
                     r_hip, r_knee,
                     arms_below_body):
    scores = {}

    # 1. 몸통 직선성
    body_angle = calculate_angle(shoulder_center, hip_center, knee_center)
    scores["body_straight"] = max(0, 100 - abs(180 - body_angle) * 2)

    # 2. 팔꿈치 각도
    l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    l_ok = ARM_ANGLE_MIN <= l_elbow_angle <= ARM_ANGLE_MAX
    r_ok = ARM_ANGLE_MIN <= r_elbow_angle <= ARM_ANGLE_MAX
    if l_ok or r_ok:
        scores["arm_angle"] = 100
    else:
        deviation = min(abs(l_elbow_angle - 90), abs(r_elbow_angle - 90))
        scores["arm_angle"] = max(0, 100 - deviation * 2)

    # 3. 엉덩이 높이
    hip_y_diff = abs(hip_center[1] - shoulder_center[1])
    body_height = abs(knee_center[1] - shoulder_center[1])
    hip_ratio = hip_y_diff / body_height if body_height > 0 else 1.0
    scores["hip_level"] = max(0, 100 - hip_ratio * 300)

    # 4. 팔 위치
    scores["arm_position"] = 100 if arms_below_body else 30

    # 5. 허리 과신전 검사
    l_waist_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    r_waist_angle = calculate_angle(r_shoulder, r_hip, r_knee)
    avg_waist_angle = (l_waist_angle + r_waist_angle) / 2

    if avg_waist_angle <= 195:
        scores["waist_bend"] = 100
    else:
        scores["waist_bend"] = max(0, 100 - (avg_waist_angle - 180) * 3)

    return scores

# 최종 점수 및 등급 산정

def calculate_final_score(frame_scores):
    """프레임별 점수 → 평균 점수 → 각 부위별 등급 반환"""
    import statistics

    # 부위별 평균 점수
    avg_scores = {
        k: statistics.mean([fs[k] for fs in frame_scores])
        for k in frame_scores[0]
    }

    # 부위별 등급
    grades = {
        k: get_grade(avg_scores[k])
        for k in avg_scores
    }

    return avg_scores, grades

# 결과 출력
def print_per_part_summary(avg_scores, grades):
    print("[부위별 플랭크 평가 결과]")
    print("──────────────────────────────")
    for part in avg_scores:
        print(f"{part:>14}: {avg_scores[part]:5.1f}점  →  등급: {grades[part]}")
    print("──────────────────────────────")


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
# 직선성 검사: 세 점이 일직선에 가까운지 확인
# ─────────────────────────────────────────────────────────────────────────────
def is_straight_line(A, B, C, threshold=20):
    """세 점이 일직선에 가까운지 확인 (각도가 180도에 가까운지)"""
    angle = calculate_angle(A, B, C)
    return angle >= (180 - threshold)

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
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "plank_man.mov")
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
# 3) 플랭크 조건 파라미터 (더 엄격하게 조정)
# ─────────────────────────────────────────────────────────────────────────────
BODY_STRAIGHT_THRESHOLD = 25    # 어깨-엉덩이-무릎이 일직선에서 벗어날 수 있는 각도 (도)
ARM_ANGLE_MIN = 45              # 팔꿈치 각도 최소값 (너무 구부러지면 안됨)
ARM_ANGLE_MAX = 120             # 팔꿈치 각도 최대값 (너무 펴져도 안됨)
CONFIDENCE_THRESHOLD = 0.3      # 랜드마크 신뢰도 최소값

# ─────────────────────────────────────────────────────────────────────────────
# 4) 각 프레임마다 플랭크 여부 판단
# ─────────────────────────────────────────────────────────────────────────────
results_list = []
frame_scores = []
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
        lm = results.pose_landmarks.landmark
        
        # 필요한 랜드마크들의 신뢰도 확인
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        
        # 모든 필요한 랜드마크가 충분한 신뢰도를 가지는지 확인
        if all(lm[landmark].visibility > CONFIDENCE_THRESHOLD for landmark in required_landmarks):
            
            # 양쪽 좌표 계산
            l_shoulder = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                         lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
            r_shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
            l_hip = (lm[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)
            r_hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h)
            l_knee = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                     lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
            r_knee = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                     lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)
            l_elbow = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                      lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)
            r_elbow = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)
            l_wrist = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                      lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
            r_wrist = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
            
            # 중앙 좌표 계산 (좌우 평균)
            shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2, 
                              (l_shoulder[1] + r_shoulder[1]) / 2)
            hip_center = ((l_hip[0] + r_hip[0]) / 2, 
                         (l_hip[1] + r_hip[1]) / 2)
            knee_center = ((l_knee[0] + r_knee[0]) / 2, 
                          (l_knee[1] + r_knee[1]) / 2)
            
            # 플랭크 조건 확인
            conditions_met = []
            
            # 1. 몸통 직선성 확인 (어깨-엉덩이-무릎)
            body_straight = is_straight_line(shoulder_center, hip_center, knee_center, 
                                           BODY_STRAIGHT_THRESHOLD)
            conditions_met.append(("Body straight", body_straight))
            
            # 2. 팔꿈치 각도 확인 (적절한 플랭크 자세)
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            l_arm_ok = ARM_ANGLE_MIN <= l_elbow_angle <= ARM_ANGLE_MAX
            r_arm_ok = ARM_ANGLE_MIN <= r_elbow_angle <= ARM_ANGLE_MAX
            arm_position_ok = l_arm_ok or r_arm_ok
            conditions_met.append(("Arm position", arm_position_ok))
            
            # 3. 엉덩이가 너무 높거나 낮지 않은지 확인
            # 어깨와 엉덩이의 y좌표 차이가 적당한지 확인
            hip_shoulder_y_diff = abs(hip_center[1] - shoulder_center[1])
            body_height = abs(knee_center[1] - shoulder_center[1])
            if body_height > 0:
                hip_ratio = hip_shoulder_y_diff / body_height
                hip_level_ok = hip_ratio < 0.3  # 엉덩이가 너무 높거나 낮지 않음
            else:
                hip_level_ok = False
            conditions_met.append(("Hip level", hip_level_ok))
            
            # 4. 팔이 몸 아래쪽에 있는지 확인 (플랭크 자세)
            arms_below_body = (l_elbow[1] > shoulder_center[1] and 
                              r_elbow[1] > shoulder_center[1])
            conditions_met.append(("Arms below body", arms_below_body))
            
            # 디버깅 정보 출력 (처음 몇 프레임만)
            if frame_idx < 10:
                print(f"\nFrame {frame_idx}:")
                for condition, met in conditions_met:
                    print(f"  {condition}: {met}")
                print(f"  Left elbow angle: {l_elbow_angle:.1f}°")
                print(f"  Right elbow angle: {r_elbow_angle:.1f}°")
                print(f"  Hip ratio: {hip_ratio:.3f}")
            
            # 모든 조건이 만족되면 플랭크
            is_plank = all(met for _, met in conditions_met)
            scores = evaluate_posture(
                shoulder_center, hip_center, knee_center,
                l_shoulder, l_elbow, l_wrist,
                r_shoulder, r_elbow, r_wrist,
                l_hip, l_knee,           
                r_hip, r_knee,          
                arms_below_body
            )
            frame_scores.append(scores)
    
    results_list.append((frame_idx, is_plank))
    frame_idx += 1

cap.release()
pose.close()

# ─────────────────────────────────────────────
# 평가 점수 계산 및 출력 (여기에 넣으세요!)
# ─────────────────────────────────────────────
if frame_scores:
    avg_scores, grades = calculate_final_score(frame_scores)
    print_per_part_summary(avg_scores, grades)

# ─────────────────────────────────────────────────────────────────────────────
# 5) 연속된 플랭크 구간(segment) 묶기 (최소 길이 필터 추가)
# ─────────────────────────────────────────────────────────────────────────────
MIN_PLANK_DURATION = int(fps * 0.5)  # 최소 0.5초 이상 지속되어야 함

segments = []
curr_state = False
seg_start = None

for idx, is_plank in results_list:
    if is_plank and not curr_state:
        seg_start = idx
        curr_state = True
    elif not is_plank and curr_state:
        if seg_start is not None and (idx - 1 - seg_start) >= MIN_PLANK_DURATION:
            segments.append((seg_start, idx - 1))
        seg_start = None
        curr_state = False

# 마지막 구간 처리
if curr_state and seg_start is not None:
    if (len(results_list) - 1 - seg_start) >= MIN_PLANK_DURATION:
        segments.append((seg_start, len(results_list) - 1))

print(f"\n=== Detected Plank Segments (min duration: {MIN_PLANK_DURATION} frames) ===")
for i, (s, e) in enumerate(segments):
    duration_sec = (e - s + 1) / fps
    print(f"  Segment {i+1}: frames {s} ~ {e} (duration: {duration_sec:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 6) 플랭크 구간별로 여러 프레임 저장 (시작, 중간, 끝)
# ─────────────────────────────────────────────────────────────────────────────
if segments:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "plank_captures")
    os.makedirs(save_dir, exist_ok=True)
    
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        print(f"[Error] 두 번째 VideoCapture 열기 실패: {video_path}")
        exit(1)
    
    # 저장할 프레임들 결정
    target_frames = set()
    for i, (s, e) in enumerate(segments):
        # 각 구간에서 시작, 중간, 끝 프레임 저장
        target_frames.add(s)  # 시작
        target_frames.add((s + e) // 2)  # 중간
        target_frames.add(e)  # 끝
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
            
        if frame_idx in target_frames:
            # 어느 구간에 속하는지 찾기
            segment_info = ""
            for i, (s, e) in enumerate(segments):
                if s <= frame_idx <= e:
                    if frame_idx == s:
                        segment_info = f"seg{i+1}_start"
                    elif frame_idx == e:
                        segment_info = f"seg{i+1}_end"
                    else:
                        segment_info = f"seg{i+1}_mid"
                    break
            
            filename = f"plank_{segment_info}_frame_{frame_idx:04d}.png"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"[Saved] {filepath}")
            saved_count += 1
        
        frame_idx += 1
    
    cap2.release()
    print(f"\n[Complete] 총 {saved_count}개의 플랭크 프레임을 저장했습니다.")
else:
    print("\n[Warning] 플랭크 구간이 감지되지 않았습니다.")
    print("다음을 확인해보세요:")
    print("1. 비디오에서 사람이 명확히 보이는지")
    print("2. 플랭크 자세가 올바른지 (팔꿈치가 어깨 아래, 몸이 일직선)")
    print("3. 조건 파라미터를 조정해야 하는지")