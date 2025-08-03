#!/usr/bin/env python3
"""
젯슨 오린 나노 GStreamer 파이프라인 테스트
"""

import cv2
import time

def test_gstreamer_pipeline():
    """GStreamer 파이프라인 테스트"""
    print("=== 젯슨 오린 나노 GStreamer 파이프라인 테스트 ===")
    
    # 젯슨 오린 나노용 GStreamer 파이프라인
    gst_str = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    print(f"GStreamer 파이프라인: {gst_str}")
    print("카메라 초기화 중...")
    
    # GStreamer로 카메라 초기화
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ GStreamer로 카메라를 열 수 없습니다.")
        return False
    
    print("✅ GStreamer 카메라 초기화 성공!")
    
    # 카메라 정보 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"카메라 정보: {width}x{height} @ {fps}fps")
    
    # 5초간 스트림 테스트
    print("5초간 스트림 테스트 중...")
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        
        frame_count += 1
        
        # 프레임 정보 표시
        cv2.putText(frame, f'GStreamer Test', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Size: {frame.shape[1]}x{frame.shape[0]}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('GStreamer Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"테스트 완료: {frame_count} 프레임, {elapsed_time:.1f}초, {actual_fps:.1f} FPS")
    
    if frame_count > 0:
        print("✅ GStreamer 파이프라인이 정상적으로 작동합니다!")
        return True
    else:
        print("❌ GStreamer 파이프라인에 문제가 있습니다.")
        return False

def test_alternative_pipelines():
    """대안 GStreamer 파이프라인들 테스트"""
    print("\n=== 대안 GStreamer 파이프라인 테스트 ===")
    
    pipelines = [
        # 파이프라인 1: 기본 nvarguscamerasrc
        (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink"
        ),
        # 파이프라인 2: USB 카메라용
        (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink"
        ),
        # 파이프라인 3: 간단한 nvarguscamerasrc
        (
            "nvarguscamerasrc ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink"
        )
    ]
    
    for i, pipeline in enumerate(pipelines, 1):
        print(f"\n파이프라인 {i} 테스트 중...")
        print(f"파이프라인: {pipeline}")
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 파이프라인 {i} 성공!")
                print(f"   프레임 크기: {frame.shape}")
                
                # 2초간 표시
                start_time = time.time()
                while time.time() - start_time < 2:
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, f'Pipeline {i}', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow(f'Pipeline {i}', frame)
                        cv2.waitKey(1)
                
                cv2.destroyAllWindows()
                cap.release()
                return True
            else:
                print(f"❌ 파이프라인 {i} - 프레임 읽기 실패")
        else:
            print(f"❌ 파이프라인 {i} - 카메라 열기 실패")
        
        cap.release()
    
    print("❌ 모든 파이프라인이 실패했습니다.")
    return False

def main():
    print("젯슨 오린 나노 GStreamer 테스트")
    print("=" * 50)
    
    # 메인 GStreamer 파이프라인 테스트
    if test_gstreamer_pipeline():
        print("\n✅ 메인 파이프라인이 성공했습니다!")
    else:
        print("\n❌ 메인 파이프라인이 실패했습니다.")
        print("대안 파이프라인을 시도합니다...")
        test_alternative_pipelines()
    
    print("\n=== 테스트 완료 ===")
    print("성공한 파이프라인을 squat_realtime_jetson.py에 적용하세요.")

if __name__ == "__main__":
    main() 