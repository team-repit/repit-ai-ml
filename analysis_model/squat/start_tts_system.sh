#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 설정
RIVA_IMAGE="nvcr.io/nvidia/riva/riva-speech:23.12-riva-client"
RIVA_CONTAINER="riva-tts-server"
RIVA_PORT="8000"
PYTHON_APP="squat_real_tts.py"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Riva 서버 상태 확인
check_riva_server() {
    if docker ps | grep -q "$RIVA_CONTAINER"; then
        return 0  # 실행 중
    else
        return 1  # 중지됨
    fi
}

# Riva 서버 시작
start_riva_server() {
    print_status "Riva TTS 서버 시작 중..."
    
    # 기존 컨테이너가 있으면 제거
    if docker ps -a | grep -q "$RIVA_CONTAINER"; then
        docker rm -f "$RIVA_CONTAINER" 2>/dev/null
    fi
    
    # 새 컨테이너 실행
    docker run -d --gpus all \
        -p "$RIVA_PORT:$RIVA_PORT" \
        -v /tmp/riva_cache:/cache \
        --name "$RIVA_CONTAINER" \
        "$RIVA_IMAGE"
    
    if [ $? -eq 0 ]; then
        print_success "Riva 서버 시작 완료"
        
        # 서버 준비 대기
        print_status "서버 준비 대기 중... (최대 2분)"
        for i in {1..120}; do
            if curl -s "http://localhost:$RIVA_PORT/health" >/dev/null 2>&1; then
                print_success "Riva 서버 준비 완료!"
                return 0
            fi
            echo -n "."
            sleep 1
        done
        
        print_warning "서버 응답 대기 시간 초과 (계속 진행)"
    else
        print_error "Riva 서버 시작 실패"
        return 1
    fi
}

# Python 애플리케이션 실행
run_python_app() {
    print_status "Python 애플리케이션 실행 중..."
    
    # 가상환경 확인
    if [ ! -d "jetson_tts_env" ]; then
        print_error "가상환경이 없습니다. setup_jetson.sh를 먼저 실행하세요."
        return 1
    fi
    
    # 가상환경 활성화
    source jetson_tts_env/bin/activate
    
    # Python 파일 존재 확인
    if [ ! -f "$PYTHON_APP" ]; then
        print_error "Python 파일을 찾을 수 없습니다: $PYTHON_APP"
        return 1
    fi
    
    # 애플리케이션 실행
    print_success "스쿼트 분석 시작!"
    python "$PYTHON_APP"
}

# 메인 메뉴
show_menu() {
    echo ""
    echo "🎯 TTS 시스템 관리 메뉴"
    echo "========================"
    echo "1. Riva 서버 시작"
    echo "2. Riva 서버 중지"
    echo "3. Riva 서버 상태 확인"
    echo "4. Python 애플리케이션 실행"
    echo "5. 전체 시스템 시작 (Riva + Python)"
    echo "6. 전체 시스템 중지"
    echo "7. 종료"
    echo "========================"
    echo -n "선택하세요 (1-7): "
}

# 메뉴 처리
handle_menu() {
    case $1 in
        1)
            start_riva_server
            ;;
        2)
            print_status "Riva 서버 중지 중..."
            docker stop "$RIVA_CONTAINER" 2>/dev/null
            docker rm "$RIVA_CONTAINER" 2>/dev/null
            print_success "Riva 서버 중지 완료"
            ;;
        3)
            if check_riva_server; then
                print_success "Riva 서버 실행 중"
                docker ps | grep "$RIVA_CONTAINER"
            else
                print_warning "Riva 서버 중지됨"
            fi
            ;;
        4)
            run_python_app
            ;;
        5)
            print_status "전체 시스템 시작 중..."
            if start_riva_server; then
                sleep 5  # 서버 안정화 대기
                run_python_app
            else
                print_warning "Riva 없이 Python 애플리케이션만 실행"
                run_python_app
            fi
            ;;
        6)
            print_status "전체 시스템 중지 중..."
            docker stop "$RIVA_CONTAINER" 2>/dev/null
            docker rm "$RIVA_CONTAINER" 2>/dev/null
            print_success "전체 시스템 중지 완료"
            ;;
        7)
            print_success "프로그램 종료"
            exit 0
            ;;
        *)
            print_error "잘못된 선택입니다. 1-7 중에서 선택하세요."
            ;;
    esac
}

# 메인 실행
main() {
    echo "🚀 TTS 시스템 관리자 시작!"
    echo "============================"
    
    # Docker 확인
    if ! command -v docker &> /dev/null; then
        print_error "Docker가 설치되지 않았습니다."
        print_status "설치 방법: sudo apt-get install docker.io"
        exit 1
    fi
    
    # Docker 서비스 확인
    if ! docker info &>/dev/null; then
        print_error "Docker 서비스가 실행되지 않았습니다."
        print_status "시작 방법: sudo systemctl start docker"
        exit 1
    fi
    
    print_success "Docker 준비 완료"
    
    # 메뉴 루프
    while true; do
        show_menu
        read -r choice
        handle_menu "$choice"
        
        echo ""
        echo "계속하려면 Enter를 누르세요..."
        read -r
    done
}

# 스크립트 실행
main "$@" 