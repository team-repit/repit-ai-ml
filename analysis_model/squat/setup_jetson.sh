#!/bin/bash

echo "🚀 Jetson TTS 시스템 설치 스크립트 시작!"
echo "=========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
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

# 시스템 확인
print_status "시스템 정보 확인 중..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Architecture: $(uname -m)"
echo "Python: $(python3 --version)"

# 시스템 업데이트
print_status "시스템 패키지 업데이트 중..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 필수 시스템 패키지 설치
print_status "필수 시스템 패키지 설치 중..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    git \
    curl \
    wget

# Python 가상환경 생성
print_status "Python 가상환경 생성 중..."
if [ ! -d "jetson_tts_env" ]; then
    python3 -m venv jetson_tts_env
    print_success "가상환경 생성 완료"
else
    print_warning "가상환경이 이미 존재합니다"
fi

# 가상환경 활성화
print_status "가상환경 활성화 중..."
source jetson_tts_env/bin/activate

# pip 업그레이드
print_status "pip 업그레이드 중..."
pip install --upgrade pip

# Python 패키지 설치
print_status "Python 패키지 설치 중..."
pip install \
    opencv-python \
    mediapipe \
    numpy \
    gtts \
    pydub \
    wave

# TTS 도구 설치
print_status "TTS 도구 설치 중..."

# Festival TTS
if ! command -v festival &> /dev/null; then
    print_status "Festival TTS 설치 중..."
    sudo apt-get install -y festival festvox-kallpc16k
    print_success "Festival TTS 설치 완료"
else
    print_success "Festival TTS 이미 설치됨"
fi

# Pico TTS
if ! command -v pico2wave &> /dev/null; then
    print_status "Pico TTS 설치 중..."
    sudo apt-get install -y pico-utils
    print_success "Pico TTS 설치 완료"
else
    print_success "Pico TTS 이미 설치됨"
fi

# Flite TTS
if ! command -v flite &> /dev/null; then
    print_status "Flite TTS 설치 중..."
    sudo apt-get install -y flite
    print_success "Flite TTS 설치 완료"
else
    print_success "Flite TTS 이미 설치됨"
fi

# MP3 재생 도구
if ! command -v mpg123 &> /dev/null; then
    print_status "MP3 재생 도구 설치 중..."
    sudo apt-get install -y mpg123
    print_success "MP3 재생 도구 설치 완료"
else
    print_success "MP3 재생 도구 이미 설치됨"
fi

# NVIDIA Riva TTS 설치 (선택사항)
print_status "NVIDIA Riva TTS 설치 중..."
if pip show nvidia-riva-client &> /dev/null; then
    print_success "Riva 클라이언트 이미 설치됨"
else
    print_status "Riva 클라이언트 설치 중..."
    pip install nvidia-riva-client
    if [ $? -eq 0 ]; then
        print_success "Riva 클라이언트 설치 완료"
    else
        print_warning "Riva 클라이언트 설치 실패 (선택사항)"
    fi
fi

# Docker 설치 (Riva 서버용)
print_status "Docker 설치 중..."
if ! command -v docker &> /dev/null; then
    sudo apt-get install -y docker.io
    sudo usermod -aG docker $USER
    sudo systemctl start docker
    sudo systemctl enable docker
    print_success "Docker 설치 완료"
else
    print_success "Docker 이미 설치됨"
fi

# NVIDIA Container Toolkit 설치
print_status "NVIDIA Container Toolkit 설치 중..."
if ! command -v nvidia-container-toolkit &> /dev/null; then
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    print_success "NVIDIA Container Toolkit 설치 완료"
else
    print_success "NVIDIA Container Toolkit 이미 설치됨"
fi

# 테스트 실행
print_status "설치 테스트 중..."
python3 -c "
import cv2
import mediapipe as mp
import numpy as np
print('✅ OpenCV, MediaPipe, NumPy 정상 작동')
"

if [ $? -eq 0 ]; then
    print_success "기본 패키지 테스트 성공"
else
    print_error "기본 패키지 테스트 실패"
fi

# 사용법 안내
echo ""
echo "🎉 설치 완료!"
echo "=========================================="
echo "사용 방법:"
echo "1. 가상환경 활성화: source jetson_tts_env/bin/activate"
echo "2. 스쿼트 분석 실행: python squat_real_tts.py"
echo ""
echo "Riva TTS 사용 시:"
echo "1. Riva 서버 실행:"
echo "   docker run --gpus all -p 8000:8000 nvcr.io/nvidia/riva/riva-speech:23.12-riva-client"
echo "2. 새 터미널에서 스쿼트 분석 실행"
echo ""
echo "문제 해결:"
echo "- 가상환경이 활성화되지 않으면: source jetson_tts_env/bin/activate"
echo "- 권한 문제 시: sudo chown -R $USER:$USER jetson_tts_env"
echo "=========================================="

print_success "젯슨 TTS 시스템 설치가 완료되었습니다!" 