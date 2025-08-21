#!/usr/bin/env python3
"""
젯슨 카메라 연결 테스트 - JetsonHacksNano/CSI-Camera 기반
"""

import cv2
import time

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    GStreamer 파이프라인 생성
    JetsonHacksNano/CSI-Camera의 simple_camera.py에서 가져온 함수
    """
    return (
        "nvarguscamerasrc ! "
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

def initialize_camera():
    """젯슨 카메라 초기화"""
    print("젯슨 카메라 초기화 중...")
    
    # 방법 1: GStreamer 파이프라인 사용 (JetsonHacksNano 방식)
    try:
        gst_pipeline = gstreamer_pipeline(
            capture_width=1280,
            capture_height=720,
            display_width=1280,
            display_height=720,
            framerate=30,
            flip_method=0
        )
        
        print(f"GStreamer 파이프라인: {gst_pipeline}")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ GStreamer 카메라 초기화 성공!")
                return cap
            else:
                print("❌ GStreamer 카메라 열기 성공했지만 프레임 읽기 실패")
        else:
            print("❌ GStreamer 카메라 열기 실패")
    except Exception as e:
        print(f"GStreamer 에러: {e}")
    
    if 'cap' in locals():
        cap.release()
    
    # 방법 2: V4L2 백엔드 사용
    print("GStreamer 실패, V4L2 백엔드 시도...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            ret, frame = cap.read()
            if ret:
                print("✅ V4L2 카메라 초기화 성공!")
                return cap
            else:
                print("❌ V4L2 카메라 열기 성공했지만 프레임 읽기 실패")
        else:
            print("❌ V4L2 카메라 열기 실패")
    except Exception as e:
        print(f"V4L2 에러: {e}")
    
    if 'cap' in locals():
        cap.release()
    
    # 방법 3: 기본 백엔드 사용
    print("V4L2 실패, 기본 백엔드 시도...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            ret, frame = cap.read()
            if ret:
                print("✅ 기본 백엔드 카메라 초기화 성공!")
                return cap
            else:
                print("❌ 기본 백엔드 카메라 열기 성공했지만 프레임 읽기 실패")
        else:
            print("❌ 기본 백엔드 카메라 열기 실패")
    except Exception as e:
        print(f"기본 백엔드 에러: {e}")
    
    if 'cap' in locals():
        cap.release()
    
    print("❌ 모든 카메라 초기화 방법 실패")
    return None

def test_camera():
    """카메라 테스트"""
    cap = initialize_camera()
    if cap is None:
        print("카메라를 초기화할 수 없습니다.")
        return False
    
    # 카메라 정보 출력
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"카메라 정보: {width}x{height} @ {fps}fps")
    
    # 10초간 카메라 스트림 테스트
    print("10초간 카메라 스트림 테스트 중...")
    print("종료하려면 'q'를 누르세요.")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        
        frame_count += 1
        
        # 프레임 정보 표시
        cv2.putText(frame, 'Jetson Camera Test', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Size: {frame.shape[1]}x{frame.shape[0]}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Jetson Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"테스트 완료: {frame_count} 프레임, {elapsed_time:.1f}초, {actual_fps:.1f} FPS")
    
    if frame_count > 0:
        print("✅ 카메라가 정상적으로 작동합니다!")
        return True
    else:
        print("❌ 카메라에 문제가 있습니다.")
        return False

def get_working_camera():
    """작동하는 카메라 객체 반환 (squat_realtime_jetson에서 사용)"""
    return initialize_camera()

def main():
    """메인 함수"""
    print("젯슨 카메라 연결 테스트")
    print("=" * 50)
    
    success = test_camera()
    
    if success:
        print("\n✅ 카메라 테스트 성공!")
        print("이제 squat_realtime_jetson.py에서 이 카메라 초기화 방법을 사용할 수 있습니다.")
    else:
        print("\n❌ 카메라 테스트 실패!")
        print("하드웨어 연결이나 드라이버를 확인하세요.")

if __name__ == "__main__":
    main() 