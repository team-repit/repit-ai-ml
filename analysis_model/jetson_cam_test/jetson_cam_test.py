import cv2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    OpenCV에서 사용할 GStreamer 파이프라인 문자열을 생성합니다.
    터미널 명령어와 거의 동일하지만, 마지막 영상 출력(sink) 부분이
    OpenCV가 프레임을 받을 수 있도록 'appsink'로 변경된 점이 핵심입니다.
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    """GStreamer 파이프라인으로 카메라를 열고 영상을 출력하는 메인 함수"""

    # 위에서 정의한 함수를 이용해 파이프라인 문자열을 가져옵니다.
    pipeline = gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30)
    print("사용할 GStreamer 파이프라인:")
    print(pipeline)

    # cv2.VideoCapture에 파이프라인과 함께 cv2.CAP_GSTREAMER 플래그를 전달합니다.
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # 카메라가 성공적으로 열렸는지 확인합니다.
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 다음을 확인하세요:")
        print("1. OpenCV가 GStreamer를 지원하도록 설치되었는지 확인 (cv2.getBuildInformation())")
        print("2. 젯슨 보드 재부팅")
        print("3. 파이프라인 문자열에 오타가 없는지 확인")
        return

    print("카메라가 성공적으로 열렸습니다. 'q'를 누르면 종료됩니다.")

    window_title = "Jetson Camera with GStreamer"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    while True:
        # 프레임 단위로 비디오를 읽어옵니다.
        ret, frame = cap.read()

        # ret이 True이면 프레임을 성공적으로 읽은 것입니다.
        if not ret:
            print("프레임을 읽는 데 실패했습니다. 스트림이 종료되었을 수 있습니다.")
            break

        # 읽어온 프레임을 화면에 표시합니다.
        cv2.imshow(window_title, frame)

        # 'q' 키를 누르면 루프를 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 작업이 끝나면, 자원을 해제합니다.
    print("카메라를 닫고 창을 종료합니다.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()