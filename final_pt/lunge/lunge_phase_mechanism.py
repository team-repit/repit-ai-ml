import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수: 세 점(A, B, C)의 각도(∠ABC)를 계산
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    """
    세 점 A, B, C를 사용하여 ∠ABC 각도를 계산합니다.
    B는 각도의 꼭짓점입니다.
    """
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
    """
    두 각도 간의 코사인 유사도를 계산합니다.
    각도 차이가 작을수록 유사도(코사인 값)는 1에 가까워집니다.
    """
    diff_rad = (angle1 - angle2) * np.pi / 180.0
    return np.cos(diff_rad)

# ─────────────────────────────────────────────────────────────────────────────
# 코사인 유사도(sim)에 따라 A, B, C, D, F 등급 반환
# ─────────────────────────────────────────────────────────────────────────────
def assign_grade_from_similarity(sim: float) -> Optional[str]:
    """
    주어진 코사인 유사도에 따라 등급을 할당합니다.
    유사도가 높을수록 좋은 등급을 받습니다.
    """
    if sim >= 0.9: return "A"
    elif sim >= 0.8: return "B"
    elif sim >= 0.7: return "C"
    elif sim >= 0.6: return "D"
    # [수정] 유사도가 0.3 '초과'일 경우에만 "자세를 잡아주세요"를 반환합니다.
    elif sim >= 0.3: return "F" 
        # 유사도 0.3이란 수치는 저희가 하면서 정해나가는 게 좋을 것 같아요,, 지금은 걍 제 뇌피셜입니다
    # 유사도가 0.3 이하인 경우는 'F'를 반환하여 표시되지 않도록 합니다.
    else: return "자세를 잡아주세요"

# ─────────────────────────────────────────────────────────────────────────────
# 이미지에서 런지 각도 추출
# ─────────────────────────────────────────────────────────────────────────────
def get_lunge_angles_from_image(image_path: str, pose_model) -> Optional[Tuple[float, float, float]]:
    """
    단일 이미지에서 런지 자세의 주요 각도를 추출합니다.
    (앞 무릎, 뒷 무릎, 엉덩이 각도)
    더 많이 굽혀진 다리(무릎 각도가 작은 쪽)를 '앞다리'로 판단합니다.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"참조 이미지를 열 수 없습니다: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError(f"참조 이미지에서 자세를 감지할 수 없습니다: {image_path}")

    h, w, _ = image.shape
    lm = results.pose_landmarks.landmark
    def get_point(landmark): return (int(landmark.x * w), int(landmark.y * h))

    # 랜드마크 포인트 정의
    left_hip, right_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP]), get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
    left_knee, right_knee = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE]), get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
    left_ankle, right_ankle = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE]), get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
    left_shoulder, right_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]), get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])

    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
    angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)

    # 무릎 각도가 더 작은 쪽을 '앞다리'로 간주하여 각도를 반환
    if angle_left_knee < angle_right_knee:
        return angle_left_knee, angle_right_knee, angle_left_hip
    else:
        return angle_right_knee, angle_left_knee, angle_right_hip

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose 초기화
# ─────────────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose

# ─────────────────────────────────────────────────────────────────────────────
# 파일 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
input_path = "lunge_video_2.mov"
output_path = "lunge_output_phase_based2.mp4"
reference_image_path = "lunge_reference_pose.png"  # 참조할 자세 이미지 경로

# ─────────────────────────────────────────────────────────────────────────────
# 참조 이미지에서 목표 각도 추출
# ─────────────────────────────────────────────────────────────────────────────
try:
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose_static:
        reference_front_knee_angle, reference_back_knee_angle, reference_hip_angle = get_lunge_angles_from_image(reference_image_path, pose_static)
    print("참조 이미지 분석 성공.")
    print(f"  - 목표 앞무릎 각도: {reference_front_knee_angle:.1f}°")
    print(f"  - 목표 뒷무릎 각도: {reference_back_knee_angle:.1f}°")
    print(f"  - 목표 엉덩이 각도: {reference_hip_angle:.1f}°")
except (IOError, ValueError) as e:
    print(f"오류: {e}")
    print("프로그램을 종료합니다.")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# 비디오 처리 준비
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"오류: 비디오를 열 수 없습니다: {input_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0

display_duration = int(fps * 2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 이전 프레임 데이터 저장 변수
front_knee_angle_prev2, front_knee_angle_prev1 = None, None
back_knee_angle_prev1, hip_angle_prev1 = None, None

# 등급 표시용 데이터 및 카운트다운 변수
display_countdown = 0
lunge_data = {}

# 런지 단계(phase)를 저장할 변수 추가
lunge_phase = "stand"  # 시작 상태는 'stand'로 설정

# ─────────────────────────────────────────────────────────────────────────────
# 비디오 프레임별 분석 루프
# ─────────────────────────────────────────────────────────────────────────────
with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, smooth_landmarks=True) as pose_video:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_video.process(image_rgb)

        front_knee_angle_curr, back_knee_angle_curr, hip_angle_curr = None, None, None

        if results.pose_landmarks:
            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark
            def get_point(landmark): return (int(landmark.x * w), int(landmark.y * h))

            left_hip, right_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP]), get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP])
            left_knee, right_knee = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE]), get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
            left_ankle, right_ankle = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE]), get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
            left_shoulder, right_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]), get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])

            angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
            angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
            angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
            
            if angle_left_knee < angle_right_knee:
                front_knee_angle_curr, back_knee_angle_curr, hip_angle_curr = angle_left_knee, angle_right_knee, angle_left_hip
            else:
                front_knee_angle_curr, back_knee_angle_curr, hip_angle_curr = angle_right_knee, angle_left_knee, angle_right_hip

        # 런지 단계(phase) 감지 로직
        if front_knee_angle_curr is not None and front_knee_angle_prev1 is not None:
            # Local Minimum (앞무릎 각도의 최소값) 감지
            is_bottom = (front_knee_angle_prev2 is not None and
                         front_knee_angle_prev1 is not None and
                         front_knee_angle_prev2 > front_knee_angle_prev1 < front_knee_angle_curr)

            if is_bottom:
                lunge_phase = "bottom"
                # 최저점에서 자세 평가
                sim_front_knee = cosine_similarity_scalar(front_knee_angle_prev1, reference_front_knee_angle)
                sim_back_knee = cosine_similarity_scalar(back_knee_angle_prev1, reference_back_knee_angle)
                sim_hip = cosine_similarity_scalar(hip_angle_prev1, reference_hip_angle)
                overall_sim = min(sim_front_knee, sim_back_knee, sim_hip)
                grade = assign_grade_from_similarity(overall_sim)
                
                # 'F' 등급이 아닐 경우에만 평가 데이터 저장 및 카운트다운 시작
                if grade != 'F':
                    lunge_data = {
                        "front_knee_angle": front_knee_angle_prev1,
                        "back_knee_angle": back_knee_angle_prev1,
                        "hip_angle": hip_angle_prev1,
                        "grade": grade,
                    }
                    display_countdown = display_duration
            # 올라가는 단계 (ascend)
            elif lunge_phase in ["descend", "bottom"] and front_knee_angle_curr > front_knee_angle_prev1:
                lunge_phase = "ascend"
            # 내려가는 단계 (descend)
            elif lunge_phase in ["stand", "ascend"] and front_knee_angle_curr < front_knee_angle_prev1:
                lunge_phase = "descend"
            # 서 있는 상태 (stand)
            elif lunge_phase == "ascend" and front_knee_angle_curr > 165: # 165도 이상이면 선 자세로 간주
                lunge_phase = "stand"


        # 결과 오버레이 및 스켈레톤 출력
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
        # 현재 런지 단계(phase)를 화면에 표시
        phase_text = f"PHASE: {lunge_phase.upper()}"
        cv2.putText(frame, phase_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # [수정] 'stand' 상태일 때와 아닐 때를 구분하여 텍스트 표시
        if lunge_phase == "stand":
            # 'stand' 상태일 경우 안내 문구 표시
            stand_text = "자세를 잡아주세요"
            font_face, font_scale, thickness, color = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2, (255, 255, 0) # 청록색
            (text_width, text_height), baseline = cv2.getTextSize(stand_text, font_face, font_scale, thickness)
            x_pos, y_pos = (frame_width - text_width) // 2, frame_height - 50
            cv2.rectangle(frame, (x_pos - 10, y_pos - text_height - 10), (x_pos + text_width + 10, y_pos + baseline), (0, 0, 0), -1)
            cv2.putText(frame, stand_text, (x_pos, y_pos), font_face, font_scale, color, thickness, lineType=cv2.LINE_AA)

        # 등급 표시 로직 (최저점 이후 일정 시간 동안만)
        elif display_countdown > 0:
            overlay_text = (
                f"Front Knee: {lunge_data['front_knee_angle']:.1f} | "
                f"Back Knee: {lunge_data['back_knee_angle']:.1f} | "
                f"Hip: {lunge_data['hip_angle']:.1f} | "
                f"Grade: {lunge_data['grade']}"
            )
            font_face, font_scale, thickness, color = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, (0, 0, 255) # 빨간색
            (text_width, text_height), baseline = cv2.getTextSize(overlay_text, font_face, font_scale, thickness)
            x_pos, y_pos = (frame_width - text_width) // 2, frame_height - 50 # 화면 하단에 표시
            cv2.rectangle(frame, (x_pos - 10, y_pos - text_height - 10), (x_pos + text_width + 10, y_pos + baseline), (0, 0, 0), -1)
            cv2.putText(frame, overlay_text, (x_pos, y_pos), font_face, font_scale, color, thickness, lineType=cv2.LINE_AA)
            display_countdown -= 1

        out.write(frame)

        # 이전 프레임 데이터 갱신
        front_knee_angle_prev2, front_knee_angle_prev1 = front_knee_angle_prev1, front_knee_angle_curr
        back_knee_angle_prev1, hip_angle_prev1 = back_knee_angle_curr, hip_angle_curr

cap.release()
out.release()

print(f"\n[완료] 결과 동영상 저장: {output_path}")
