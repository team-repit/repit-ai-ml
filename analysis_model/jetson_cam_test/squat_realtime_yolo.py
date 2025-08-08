import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple
from collections import Counter as GradeCounter
from ultralytics import YOLO

# YOLO Pose 모델 초기화 (TensorRT 지원)
def initialize_yolo_pose():
    """YOLO Pose 모델을 초기화하고 반환합니다. TensorRT 엔진을 우선 사용합니다."""
    import platform
    
    # TensorRT 지원 환경 감지
    def can_use_tensorrt():
        try:
            import torch
            # CUDA 사용 가능한지 확인
            if not torch.cuda.is_available():
                return False, "CUDA 사용 불가"
            
            # 플랫폼 확인
            platform_info = platform.platform().lower()
            
            # Jetson 환경
            if 'tegra' in platform_info or 'jetson' in platform_info:
                return True, "Jetson 환경"
            
            # 윈도우/리눅스 + NVIDIA GPU 환경
            if 'windows' in platform_info or 'linux' in platform_info:
                try:
                    # GPU 정보 확인
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    if 'nvidia' in gpu_name:
                        return True, f"NVIDIA GPU 환경: {torch.cuda.get_device_name(0)}"
                except:
                    pass
            
            return False, f"TensorRT 미지원 환경: {platform_info}"
        except Exception as e:
            return False, f"환경 감지 실패: {e}"
    
    can_tensorrt, tensorrt_reason = can_use_tensorrt()
    print(f"🔍 환경 감지: {tensorrt_reason}")
    
    # TensorRT 엔진 파일 우선 시도
    engine_path = 'yolo11n-pose.engine'
    pt_path = 'yolo11n-pose.pt'
    
    try:
        # 1. TensorRT 엔진이 이미 있으면 로드
        if os.path.exists(engine_path):
            print("🚀 기존 TensorRT 엔진을 로드합니다...")
            model = YOLO(engine_path)
            print("✅ TensorRT 엔진 로드 성공!")
            return model
    except Exception as e:
        print(f"⚠️ TensorRT 엔진 로드 실패: {e}")
    
    try:
        # 2. PyTorch 모델 로드
        print("📦 PyTorch 모델을 로드합니다...")
        model = YOLO(pt_path)
        print("✅ YOLO Pose 모델 로드 성공!")
        
        # 3. TensorRT 지원 환경이면 TensorRT로 변환 시도
        if can_tensorrt:
            try:
                print("⚡ TensorRT 엔진으로 변환합니다...")
                print("   (첫 실행 시 1-2분 소요될 수 있습니다)")
                
                # TensorRT 엔진 생성
                model.export(format='engine', device=0, half=True, workspace=4)
                
                # 변환된 엔진 로드
                model = YOLO(engine_path)
                print("🚀 TensorRT 엔진 변환 및 로드 완료!")
                print("   다음 실행부터는 빠른 속도로 시작됩니다.")
                
            except Exception as e:
                print(f"⚠️ TensorRT 변환 실패 (PyTorch 모델 사용): {e}")
                print("   일반 모델로도 정상 동작합니다.")
        
        return model
        
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        print("모델을 다운로드 중입니다...")
        try:
            model = YOLO(pt_path)
            print("✅ YOLO Pose 모델 다운로드 및 로드 성공!")
            
            # TensorRT 지원 환경에서 다운로드 후에도 TensorRT 변환 시도
            if can_tensorrt:
                try:
                    print("⚡ TensorRT 엔진으로 변환합니다...")
                    model.export(format='engine', device=0, half=True, workspace=4)
                    model = YOLO(engine_path)
                    print("🚀 TensorRT 엔진 변환 완료!")
                except Exception as e:
                    print(f"⚠️ TensorRT 변환 실패: {e}")
            
            return model
        except Exception as e2:
            print(f"❌ 모델 다운로드 실패: {e2}")
            return None

def calculate_angle(a: list, b: list, c: list) -> float:
    """세 점 사이의 각도를 계산하는 함수 (결과값: 0-180)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def yolo_to_landmarks(results, frame_shape):
    """YOLO 결과를 MediaPipe 형식의 landmarks로 변환"""
    if not results or len(results) == 0:
        return None
    
    # 키포인트가 있는지 확인
    if results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        return None
    
    # 첫 번째 검출된 사람의 키포인트 사용
    keypoints = results[0].keypoints.data[0].cpu().numpy()  # (17, 3) - x, y, confidence
    
    h, w = frame_shape[:2]
    
    # YOLO 키포인트 인덱스 매핑 (COCO 형식)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    landmarks = {}
    
    # 신뢰도가 0.5 이상인 키포인트만 사용
    confidence_threshold = 0.5
    
    # 어깨
    if keypoints[5][2] > confidence_threshold:  # left_shoulder
        landmarks['left_shoulder'] = [keypoints[5][0], keypoints[5][1]]
    if keypoints[6][2] > confidence_threshold:  # right_shoulder
        landmarks['right_shoulder'] = [keypoints[6][0], keypoints[6][1]]
    
    # 엉덩이
    if keypoints[11][2] > confidence_threshold:  # left_hip
        landmarks['left_hip'] = [keypoints[11][0], keypoints[11][1]]
    if keypoints[12][2] > confidence_threshold:  # right_hip
        landmarks['right_hip'] = [keypoints[12][0], keypoints[12][1]]
    
    # 무릎
    if keypoints[13][2] > confidence_threshold:  # left_knee
        landmarks['left_knee'] = [keypoints[13][0], keypoints[13][1]]
    if keypoints[14][2] > confidence_threshold:  # right_knee
        landmarks['right_knee'] = [keypoints[14][0], keypoints[14][1]]
    
    # 발목
    if keypoints[15][2] > confidence_threshold:  # left_ankle
        landmarks['left_ankle'] = [keypoints[15][0], keypoints[15][1]]
    if keypoints[16][2] > confidence_threshold:  # right_ankle
        landmarks['right_ankle'] = [keypoints[16][0], keypoints[16][1]]
    
    # 발가락 (발목 위치를 기반으로 추정)
    if 'left_ankle' in landmarks:
        landmarks['left_foot_index'] = [landmarks['left_ankle'][0], landmarks['left_ankle'][1] + 20]
    if 'right_ankle' in landmarks:
        landmarks['right_foot_index'] = [landmarks['right_ankle'][0], landmarks['right_ankle'][1] + 20]
    
    # 뒤꿈치 가시성 (발목 신뢰도 기반)
    landmarks['left_heel_visibility'] = keypoints[15][2] if keypoints[15][2] > confidence_threshold else 0.0
    landmarks['right_heel_visibility'] = keypoints[16][2] if keypoints[16][2] > confidence_threshold else 0.0
    
    return landmarks

def draw_yolo_pose(frame, results):
    """YOLO 포즈 결과를 프레임에 그립니다 (MediaPipe 스타일)."""
    if not results or len(results) == 0:
        return frame
    
    # 키포인트가 있는지 확인
    if results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        return frame
    
    # 첫 번째 검출된 사람의 키포인트 사용
    keypoints = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
    
    # YOLO COCO 포즈 연결 (MediaPipe 스타일로 변환)
    connections = [
        # 머리 영역
        (0, 1), (0, 2), (1, 3), (2, 4),  # 코-눈-귀
        # 몸통
        (5, 6),   # 어깨
        (5, 11), (6, 12),  # 어깨-엉덩이
        (11, 12),  # 엉덩이
        # 팔
        (5, 7), (7, 9),   # 왼팔
        (6, 8), (8, 10),  # 오른팔
        # 다리
        (11, 13), (13, 15),  # 왼다리
        (12, 14), (14, 16),  # 오른다리
    ]
    
    # 키포인트 그리기 (원)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:  # 신뢰도 임계값
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 5, (245, 117, 66), -1)  # MediaPipe와 동일한 색상
    
    # 연결선 그리기
    for start_idx, end_idx in connections:
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        
        if start_point[2] > 0.5 and end_point[2] > 0.5:
            start_x, start_y = int(start_point[0]), int(start_point[1])
            end_x, end_y = int(end_point[0]), int(end_point[1])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (245, 66, 230), 2)  # MediaPipe와 동일한 색상
    
    return frame

class ComprehensiveSquatGrader:
    """
    'AI 자세 교정을 위한 종합 평가 기준'을 기반으로 한 새로운 평가 클래스.
    계층적 피드백 구조(안전성 > 효과성 > 최적화)를 따릅니다.
    """
    def __init__(self):
        pass

    def evaluate_errors(self, landmarks: dict, angles: dict, phase: str, rep_start_hip_y: float) -> List[str]:
        """
        자세를 평가하고 발생한 모든 오류 목록을 계층적으로 반환합니다.
        """
        errors = []
        
        # 레벨 1: 안전성 (Safety) - 즉시 교정 대상
        if phase in ["DESCEND", "BOTTOM", "ASCEND"]:
            # 1-1. 허리 말림 (Butt Wink)
            if 'hip' in angles and angles['hip'] < 65:
                errors.append("허리 말림")
            
            # 1-2. 무릎 모임 (Knee Valgus)
            lk_pos, rk_pos = landmarks.get('left_knee'), landmarks.get('right_knee')
            la_pos, ra_pos = landmarks.get('left_ankle'), landmarks.get('right_ankle')
            if all([lk_pos, rk_pos, la_pos, ra_pos]):
                knee_dist = abs(lk_pos[0] - rk_pos[0])
                ankle_dist = abs(la_pos[0] - ra_pos[0])
                if ankle_dist > 0 and knee_dist < ankle_dist * 0.85:
                    errors.append("무릎 모임")

            # 1-3. "굿모닝" 스쿼트
            if phase == "ASCEND":
                hip_y = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
                shoulder_y = (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
                # 엉덩이가 어깨보다 유의미하게 먼저 올라가는지 확인
                if hip_y < (rep_start_hip_y * 0.9) and shoulder_y > (rep_start_hip_y * 0.95):
                     errors.append("굿모닝 스쿼트")

        # 레벨 2: 효과성 (Effectiveness) - 주요 교정 대상
        if phase in ["DESCEND", "BOTTOM"]:
            # 2-1. 과도한 상체 숙임 (Chest Drop)
            if 'torso' in angles and angles['torso'] < 45 and "허리 말림" not in errors:
                errors.append("상체 숙임")
            
            # 2-2. 뒤꿈치 들림 (Heel Lift)
            left_heel_vis = landmarks.get('left_heel_visibility', 1.0)
            right_heel_vis = landmarks.get('right_heel_visibility', 1.0)
            if left_heel_vis < 0.7 or right_heel_vis < 0.7:
                 errors.append("뒤꿈치 들림")

            # 2-3. 골반 치우침 (Pelvic Shift)
            hip_center_x = (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2
            ankle_center_x = (landmarks['left_ankle'][0] + landmarks['right_ankle'][0]) / 2
            shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
            if shoulder_width > 0 and abs(hip_center_x - ankle_center_x) > shoulder_width * 0.15:
                errors.append("골반 치우침")

        # 레벨 3: 최적화 (Optimization) - 미세 조정
        if phase == "BOTTOM":
            # 3-1. 깊이 부족 (Insufficient Depth)
            if 'knee' in angles and angles['knee'] > 120:
                errors.append("깊이 부족")
            
            # 3-2. 발목 가동성 부족 (Ankle Mobility)
            if 'ankle' in angles and angles['ankle'] > 80: # 배굴곡 각도가 충분하지 않음
                errors.append("발목 가동성 부족")

        return errors

    def get_grade_from_errors(self, errors: List[str]) -> str:
        """오류 개수에 따라 등급을 반환합니다."""
        num_errors = len(set(errors))
        if num_errors == 0: return "A"
        elif num_errors == 1: return "B"
        elif num_errors == 2: return "C"
        elif num_errors == 3: return "D"
        else: return "F"

# 오류 키와 상세 설명을 매핑하는 딕셔너리
ERROR_CRITERIA_MAP = {
    "허리 말림": "허리 말림 (Butt Wink): 하강 최저점에서 엉덩이가 안으로 말리며 허리의 중립이 무너지는 현상.",
    "무릎 모임": "무릎 모임 (Knee Valgus): 하강 또는 상승 시 무릎이 발보다 안쪽으로 무너지는 현상.",
    "굿모닝 스쿼트": '"굿모닝" 스쿼트: 상승 시 엉덩이가 상체보다 현저히 빠르게 올라와 허리에 과부하가 걸리는 현상.',
    "상체 숙임": "과도한 상체 숙임 (Chest Drop): 힙 힌지 범위를 넘어 상체가 과도하게 앞으로 쏠리는 자세.",
    "뒤꿈치 들림": "뒤꿈치 들림 (Heel Lift): 무게 중심이 앞으로 쏠려 뒤꿈치가 바닥에서 뜨는 현상.",
    "골반 치우침": "골반 치우침 (Pelvic Shift): 하강 또는 상승 시 골반이 좌우 한쪽으로 쏠리는 현상.",
    "깊이 부족": "깊이 부족 (Insufficient Depth): 허벅지가 지면과 평행이 되는 지점까지 충분히 하강하지 못하는 경우.",
    "발목 가동성 부족": "발목 가동성 부족 (Ankle Mobility): 스쿼트 최저점에서 발목 각도(배굴곡)가 충분하지 않은 경우."
}

def save_report(report_path: str, total_reps: int, results: List[Dict]):
    """분석 결과와 전체 평가 기준을 텍스트 파일로 저장합니다."""
    grades = [res['grade'] for res in results]
    grade_counts = GradeCounter(grades)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("실시간 스쿼트 자세 분석 리포트 (YOLO)\n")
        f.write("="*30 + "\n")
        f.write(f"총 스쿼트 횟수: {total_reps}회\n\n")
        
        # 스쿼트가 인식되지 않은 경우 특별한 피드백 제공
        if total_reps == 0:
            f.write("⚠️  스쿼트 동작이 인식되지 않았습니다.\n\n")
            f.write("가능한 원인과 해결 방법:\n")
            f.write("1. 카메라 거리 조정: 전신이 화면에 들어오도록 카메라와의 거리를 조정해주세요.\n")
            f.write("2. 조명 확인: 충분한 조명 환경에서 촬영해주세요.\n")
            f.write("3. 동작 크기: 무릎 각도가 100도 이하로 충분히 깊게 앉아주세요.\n")
            f.write("4. 측면 촬영: 정면보다는 측면에서 촬영하면 더 정확한 인식이 가능합니다.\n")
            f.write("5. 동작 속도: 너무 빠르지 않게 천천히 스쿼트를 수행해주세요.\n")
            f.write("6. 자세 확인: 발을 어깨 너비로 벌리고 올바른 스쿼트 자세를 유지해주세요.\n\n")
            f.write("💡 팁: 다음 번에는 위의 사항들을 확인한 후 다시 시도해보세요!\n\n")
        else:
            f.write("등급별 요약:\n")
            for grade in ["A", "B", "C", "D", "F"]:
                count = grade_counts.get(grade, 0)
                f.write(f"- 등급 {grade}: {count}회\n")
            
            f.write("\n" + "="*30 + "\n")
            f.write("반복별 상세 결과:\n")
            for res in results:
                f.write(f"\n--- {res['rep']}회차: 등급 {res['grade']} ---\n")
                if res['errors']:
                    f.write("  [수행하지 못한 기준]\n")
                    for error_key in sorted(res['errors']): # 오류를 가나다 순으로 정렬하여 출력
                        error_description = ERROR_CRITERIA_MAP.get(error_key, "알 수 없는 오류")
                        f.write(f"  - {error_description}\n")
                else:
                    f.write("  - 모든 기준을 만족했습니다.\n")

        # --- 전체 평가 기준 추가 ---
        f.write("\n\n" + "="*40 + "\n")
        f.write("          자세 평가 기준 (참고)\n")
        f.write("="*40 + "\n\n")

        # 스쿼트 기준
        f.write("1. 스쿼트 (Squat) 종합 기준\n")
        f.write("-------------------------\n")
        f.write("레벨 1: 안전성 (Safety) - 즉시 교정 대상\n")
        f.write("- 허리 말림 (Butt Wink): 하강 최저점에서 엉덩이가 안으로 말리며 허리의 중립이 무너지는 현상.\n")
        f.write("- 무릎 모임 (Knee Valgus): 하강 또는 상승 시 무릎이 발보다 안쪽으로 무너지는 현상.\n")
        f.write("- \"굿모닝\" 스쿼트: 상승 시 엉덩이가 상체보다 현저히 빠르게 올라와 허리에 과부하가 걸리는 현상.\n\n")
        f.write("레벨 2: 효과성 (Effectiveness) - 주요 교정 대상\n")
        f.write("- 과도한 상체 숙임 (Chest Drop): 힙 힌지 범위를 넘어 상체가 과도하게 앞으로 쏠리는 자세.\n")
        f.write("- 뒤꿈치 들림 (Heel Lift): 무게 중심이 앞으로 쏠려 뒤꿈치가 바닥에서 뜨는 현상.\n")
        f.write("- 골반 치우침 (Pelvic Shift): 하강 또는 상승 시 골반이 좌우 한쪽으로 쏠리는 현상.\n\n")
        f.write("레벨 3: 최적화 (Optimization) - 미세 조정\n")
        f.write("- 깊이 부족 (Insufficient Depth): 허벅지가 지면과 평행이 되는 지점(무릎 각도 약 110~120도)까지 충분히 하강하지 못하는 경우.\n")
        f.write("- 발목 가동성 부족 (Ankle Mobility): 스쿼트 최저점에서 발목 각도(배굴곡)가 약 20도 미만으로, 가동 범위가 제한되는 경우.\n\n")

    print(f"리포트가 '{report_path}'에 저장되었습니다.")

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    Jetson용 GStreamer 파이프라인 생성 (CSI 카메라용)
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, "
        f"width=(int){display_width}, height=(int){display_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def check_opencv_gstreamer():
    """OpenCV의 GStreamer 지원 여부 확인"""
    try:
        build_info = cv2.getBuildInformation()
        return "GStreamer:                   YES" in build_info
    except:
        return False

def initialize_jetson_camera():
    """Jetson 환경에서 카메라 초기화 (CSI + USB 지원)"""
    print("🔍 Jetson 카메라 환경 감지 중...")
    
    # OpenCV GStreamer 지원 확인
    gstreamer_supported = check_opencv_gstreamer()
    print(f"OpenCV GStreamer 지원: {'✅ YES' if gstreamer_supported else '❌ NO'}")
    
    # 1. CSI 카메라 시도 (GStreamer 지원 시)
    if gstreamer_supported:
        print("🎥 CSI 카메라 시도 중...")
        for sensor_id in range(2):  # 0, 1번 센서 시도
            try:
                pipeline = gstreamer_pipeline(sensor_id=sensor_id)
                print(f"   센서 {sensor_id}: {pipeline[:80]}...")
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"✅ CSI 카메라 {sensor_id} 성공: {width}x{height}")
                        return cap, f"CSI-{sensor_id}"
                    else:
                        print(f"❌ CSI 카메라 {sensor_id}: 프레임 읽기 실패")
                        cap.release()
                else:
                    print(f"❌ CSI 카메라 {sensor_id}: 열기 실패")
                    cap.release()
            except Exception as e:
                print(f"❌ CSI 카메라 {sensor_id} 오류: {e}")
    else:
        print("⚠️ GStreamer 미지원으로 CSI 카메라 건너뜀")
    
    # 2. USB 카메라 시도 (V4L2 백엔드)
    print("🔌 USB 카메라 시도 중...")
    available_usb_cameras = []
    
    for camera_index in range(4):
        try:
            # V4L2 백엔드로 시도
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ USB 카메라 {camera_index}: {width}x{height}")
                    available_usb_cameras.append({
                        'index': camera_index,
                        'cap': cap,
                        'width': width,
                        'height': height,
                        'score': width * height
                    })
                else:
                    print(f"❌ USB 카메라 {camera_index}: 프레임 읽기 실패")
                    cap.release()
            else:
                print(f"❌ USB 카메라 {camera_index}: 열기 실패")
                if cap.isOpened():
                    cap.release()
        except Exception as e:
            print(f"❌ USB 카메라 {camera_index} 오류: {e}")
    
    # USB 카메라 중 최적 선택
    if available_usb_cameras:
        best_camera = max(available_usb_cameras, key=lambda x: x['score'])
        
        # 다른 카메라들 해제
        for cam_info in available_usb_cameras:
            if cam_info['index'] != best_camera['index']:
                cam_info['cap'].release()
        
        cap = best_camera['cap']
        # 카메라 설정 최적화
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"🎯 USB 카메라 선택: 인덱스 {best_camera['index']}")
        return cap, f"USB-{best_camera['index']}"
    
    # 3. 일반 OpenCV 방식 (백업)
    print("🔄 일반 OpenCV 방식 시도 중...")
    for camera_index in range(4):
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ 일반 카메라 {camera_index}: {width}x{height}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return cap, f"DEFAULT-{camera_index}"
                else:
                    cap.release()
            else:
                if cap.isOpened():
                    cap.release()
        except Exception as e:
            print(f"❌ 일반 카메라 {camera_index} 오류: {e}")
    
    return None, None

def initialize_camera_cross_platform():
    """크로스 플랫폼 카메라 초기화"""
    import platform
    
    # 플랫폼 감지
    platform_info = platform.platform().lower()
    is_jetson = 'tegra' in platform_info or 'jetson' in platform_info
    
    print(f"🖥️ 플랫폼: {platform.platform()}")
    print(f"🤖 Jetson 환경: {'YES' if is_jetson else 'NO'}")
    
    if is_jetson:
        # Jetson 환경: CSI + USB 카메라 지원
        return initialize_jetson_camera()
    else:
        # 일반 환경: 기존 로직 사용
        print("💻 일반 환경에서 카메라 초기화...")
        available_cameras = []
        
        for camera_index in range(4):
            print(f"카메라 인덱스 {camera_index} 시도 중...")
            test_cap = cv2.VideoCapture(camera_index)
            
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ 카메라 인덱스 {camera_index}: {width}x{height}")
                    
                    available_cameras.append({
                        'index': camera_index,
                        'cap': test_cap,
                        'width': width,
                        'height': height,
                        'score': width * height
                    })
                else:
                    print(f"❌ 카메라 인덱스 {camera_index}: 프레임 읽기 실패")
                    test_cap.release()
            else:
                print(f"❌ 카메라 인덱스 {camera_index}: 열기 실패")
        
        if available_cameras:
            best_camera = max(available_cameras, key=lambda x: x['score'])
            
            # 다른 카메라들 해제
            for cam_info in available_cameras:
                if cam_info['index'] != best_camera['index']:
                    cam_info['cap'].release()
            
            cap = best_camera['cap']
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print(f"🎯 선택된 카메라: 인덱스 {best_camera['index']} ({best_camera['width']}x{best_camera['height']})")
            return cap, f"PC-{best_camera['index']}"
        
        return None, None

def main():
    """실시간 카메라를 통한 스쿼트 분석 메인 함수 (YOLO 버전)"""
    
    # YOLO 모델 초기화
    model = initialize_yolo_pose()
    if model is None:
        print("YOLO 모델을 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    # 크로스 플랫폼 카메라 초기화
    cap, camera_type = initialize_camera_cross_platform()
    
    if cap is None:
        print("❌ 사용 가능한 카메라를 찾을 수 없습니다.")
        print("\n🔧 해결 방법:")
        if check_opencv_gstreamer():
            print("1. CSI 카메라 연결 확인")
            print("2. USB 카메라 연결 확인")
        else:
            print("1. OpenCV GStreamer 지원 확인: pip install opencv-python-headless 대신")
            print("   Jetson에서는 JetPack과 함께 제공되는 OpenCV 사용 권장")
            print("2. 또는 다음 명령으로 GStreamer 지원 OpenCV 설치:")
            print("   sudo apt update")
            print("   sudo apt install python3-opencv")
        print("3. USB 카메라 /dev/video* 장치 확인: ls /dev/video*")
        print("4. 카메라 권한 확인: sudo usermod -a -G video $USER")
        return
    
    print(f"🎥 카메라 초기화 완료: {camera_type}")
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 영상 저장을 위한 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # output 디렉토리 생성 (없으면 생성)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 타임스탬프를 포함한 파일명 생성 (output 디렉토리 안에 저장)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"squat_realtime_yolo_analysis_{timestamp}.mp4")
    output_report_path = os.path.join(output_dir, f"squat_realtime_yolo_report_{timestamp}.txt")
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 변수 초기화
    counter = 0 
    stage = None
    grader = ComprehensiveSquatGrader()
    all_rep_results = []
    current_rep_errors = set()
    last_rep_grade = "N/A"
    rep_start_hip_y = 0
    current_phase = "READY"
    
    # 타이머 설정
    start_time = time.time()
    recording_duration = 30  # 30초
    
    print("YOLO 기반 스쿼트 분석을 시작합니다. 30초간 카메라가 켜집니다.")
    print("스쿼트 동작을 시작하세요!")
    print("종료하려면 'q'를 누르세요.")
    
    # 모델 정보 표시
    model_info = str(model.model).lower()
    if 'tensorrt' in model_info or 'engine' in model_info:
        print("🚀 TensorRT 가속 모드로 실행 중 - 최적화된 성능!")
    else:
        print("📦 PyTorch 모드로 실행 중")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 현재 시간 계산
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, recording_duration - elapsed_time)
        
        # 30초 경과 시 종료
        if elapsed_time >= recording_duration:
            break
        
        # YOLO 처리
        results = model(frame)
        
        # 스켈레톤 그리기
        frame = draw_yolo_pose(frame, results)
        
        try:
            # YOLO 결과를 landmarks 형식으로 변환
            lm_data = yolo_to_landmarks(results, frame.shape)
            
            if lm_data and all(key in lm_data for key in ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']):
                h, w, _ = frame.shape
                
                angles = {}
                # 왼쪽/오른쪽 중 더 잘 보이는 쪽 선택
                use_left_side = True  # YOLO에서는 단순화
                
                if use_left_side and all(key in lm_data for key in ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle', 'left_foot_index']):
                    angles['hip'] = calculate_angle(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_knee'])
                    angles['knee'] = calculate_angle(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle'])
                    angles['ankle'] = calculate_angle(lm_data['left_knee'], lm_data['left_ankle'], lm_data['left_foot_index'])
                    angles['torso'] = calculate_angle(lm_data['left_hip'], lm_data['left_shoulder'], [lm_data['left_shoulder'][0], lm_data['left_shoulder'][1] - 1])
                elif all(key in lm_data for key in ['right_shoulder', 'right_hip', 'right_knee', 'right_ankle', 'right_foot_index']):
                    angles['hip'] = calculate_angle(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_knee'])
                    angles['knee'] = calculate_angle(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle'])
                    angles['ankle'] = calculate_angle(lm_data['right_knee'], lm_data['right_ankle'], lm_data['right_foot_index'])
                    angles['torso'] = calculate_angle(lm_data['right_hip'], lm_data['right_shoulder'], [lm_data['right_shoulder'][0], lm_data['right_shoulder'][1] - 1])
                
                if 'knee' in angles:
                    knee_angle = angles['knee']
                    
                    if knee_angle > 160:
                        if stage == 'down': 
                            final_grade = grader.get_grade_from_errors(list(current_rep_errors))
                            all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})
                            last_rep_grade = final_grade
                            current_rep_errors.clear()
                        stage = "up"

                    if knee_angle < 100 and stage == 'up':
                        stage = "down"
                        counter += 1
                        rep_start_hip_y = (lm_data['left_hip'][1] + lm_data['right_hip'][1]) / 2

                    current_phase = ""
                    if stage == "up": current_phase = "ASCEND" if knee_angle < 170 else "READY"
                    elif stage == "down": current_phase = "BOTTOM" if knee_angle < 90 else "DESCEND"
                    
                    if stage == "down" or stage == "up":
                        errors_in_frame = grader.evaluate_errors(lm_data, angles, current_phase, rep_start_hip_y)
                        current_rep_errors.update(errors_in_frame)

        except Exception as e:
            pass
        
        # ------------------ 화면 표시 정보 (MediaPipe와 동일) ------------------
        # 상단 정보 박스
        cv2.rectangle(frame, (0,0), (frame_width, 120), (245,117,16), -1)
        
        # 타이머 표시
        cv2.putText(frame, f'TIME: {remaining_time:.1f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # REPS
        cv2.putText(frame, 'REPS', (int(frame_width * 0.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(counter), (int(frame_width * 0.3), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # PHASE
        cv2.putText(frame, 'PHASE', (int(frame_width * 0.5), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, current_phase, (int(frame_width * 0.5), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

        # LAST REP GRADE
        cv2.putText(frame, 'GRADE', (int(frame_width * 0.7), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, last_rep_grade, (int(frame_width * 0.7), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # 하단 안내 메시지 (카메라 타입 표시)
        cv2.rectangle(frame, (0, frame_height-50), (frame_width, frame_height), (0,0,0), -1)
        cv2.putText(frame, f'Press Q to quit | {camera_type} | YOLO Pose', (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        # ----------------------------------------------------
                   
        out.write(frame)
        cv2.imshow('Real-time Squat Analysis (YOLO)', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    # 마지막 스쿼트가 완료되지 않았다면 처리
    if stage == 'down' and current_rep_errors:
        final_grade = grader.get_grade_from_errors(list(current_rep_errors))
        all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 결과 저장
    save_report(output_report_path, counter, all_rep_results)
    print(f"분석 영상이 '{output_video_path}'에 저장되었습니다.")
    print(f"분석 리포트가 '{output_report_path}'에 저장되었습니다.")
    print(f"총 {counter}회의 스쿼트를 분석했습니다.")
    
    # 모델 정보 표시
    model_info = str(model.model).lower()
    if 'tensorrt' in model_info or 'engine' in model_info:
        print("🚀 TensorRT 가속 모드로 실행 중 - 최적화된 성능!")
    else:
        print("📦 PyTorch 모드로 실행 중")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 현재 시간 계산
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, recording_duration - elapsed_time)
        
        # 30초 경과 시 종료
        if elapsed_time >= recording_duration:
            break
        
        # YOLO 처리
        results = model(frame)
        
        # 스켈레톤 그리기
        frame = draw_yolo_pose(frame, results)
        
        try:
            # YOLO 결과를 landmarks 형식으로 변환
            lm_data = yolo_to_landmarks(results, frame.shape)
            
            if lm_data and all(key in lm_data for key in ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']):
                h, w, _ = frame.shape
                
                angles = {}
                # 왼쪽/오른쪽 중 더 잘 보이는 쪽 선택
                use_left_side = True  # YOLO에서는 단순화
                
                if use_left_side and all(key in lm_data for key in ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle', 'left_foot_index']):
                    angles['hip'] = calculate_angle(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_knee'])
                    angles['knee'] = calculate_angle(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle'])
                    angles['ankle'] = calculate_angle(lm_data['left_knee'], lm_data['left_ankle'], lm_data['left_foot_index'])
                    angles['torso'] = calculate_angle(lm_data['left_hip'], lm_data['left_shoulder'], [lm_data['left_shoulder'][0], lm_data['left_shoulder'][1] - 1])
                elif all(key in lm_data for key in ['right_shoulder', 'right_hip', 'right_knee', 'right_ankle', 'right_foot_index']):
                    angles['hip'] = calculate_angle(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_knee'])
                    angles['knee'] = calculate_angle(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle'])
                    angles['ankle'] = calculate_angle(lm_data['right_knee'], lm_data['right_ankle'], lm_data['right_foot_index'])
                    angles['torso'] = calculate_angle(lm_data['right_hip'], lm_data['right_shoulder'], [lm_data['right_shoulder'][0], lm_data['right_shoulder'][1] - 1])
                
                if 'knee' in angles:
                    knee_angle = angles['knee']
                    
                    if knee_angle > 160:
                        if stage == 'down': 
                            final_grade = grader.get_grade_from_errors(list(current_rep_errors))
                            all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})
                            last_rep_grade = final_grade
                            current_rep_errors.clear()
                        stage = "up"

                    if knee_angle < 100 and stage == 'up':
                        stage = "down"
                        counter += 1
                        rep_start_hip_y = (lm_data['left_hip'][1] + lm_data['right_hip'][1]) / 2

                    current_phase = ""
                    if stage == "up": current_phase = "ASCEND" if knee_angle < 170 else "READY"
                    elif stage == "down": current_phase = "BOTTOM" if knee_angle < 90 else "DESCEND"
                    
                    if stage == "down" or stage == "up":
                        errors_in_frame = grader.evaluate_errors(lm_data, angles, current_phase, rep_start_hip_y)
                        current_rep_errors.update(errors_in_frame)

        except Exception as e:
            pass
        
        # ------------------ 화면 표시 정보 (MediaPipe와 동일) ------------------
        # 상단 정보 박스
        cv2.rectangle(frame, (0,0), (frame_width, 120), (245,117,16), -1)
        
        # 타이머 표시
        cv2.putText(frame, f'TIME: {remaining_time:.1f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # REPS
        cv2.putText(frame, 'REPS', (int(frame_width * 0.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(counter), (int(frame_width * 0.3), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # PHASE
        cv2.putText(frame, 'PHASE', (int(frame_width * 0.5), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, current_phase, (int(frame_width * 0.5), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

        # LAST REP GRADE
        cv2.putText(frame, 'GRADE', (int(frame_width * 0.7), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, last_rep_grade, (int(frame_width * 0.7), 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
        
        # 하단 안내 메시지
        cv2.rectangle(frame, (0, frame_height-50), (frame_width, frame_height), (0,0,0), -1)
        cv2.putText(frame, 'Press Q to quit early | YOLO Pose Detection', (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        # ----------------------------------------------------
                   
        out.write(frame)
        cv2.imshow('Real-time Squat Analysis (YOLO)', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    # 마지막 스쿼트가 완료되지 않았다면 처리
    if stage == 'down' and current_rep_errors:
        final_grade = grader.get_grade_from_errors(list(current_rep_errors))
        all_rep_results.append({'rep': counter, 'grade': final_grade, 'errors': list(current_rep_errors)})

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 결과 저장
    save_report(output_report_path, counter, all_rep_results)
    print(f"분석 영상이 '{output_video_path}'에 저장되었습니다.")
    print(f"분석 리포트가 '{output_report_path}'에 저장되었습니다.")
    print(f"총 {counter}회의 스쿼트를 분석했습니다.")

if __name__ == "__main__":
    main() 