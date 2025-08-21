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

# pip 업그레이드
print_status "pip 업그레이드 중..."
pip install --upgrade pip

# Python 패키지 설치
print_status "Python 패키지 설치 중..."
pip install \
    gtts \
    pydub \
    wave

# TTS 도구 설치
print_status "TTS 도구 설치 중..."

# Google TTS (1순위, 한국어 품질 최고)
print_status "Google TTS (gTTS) 설치 확인 중..."
if pip show gtts &> /dev/null; then
    print_success "Google TTS (gTTS) 이미 설치됨"
else
    print_status "Google TTS (gTTS) 설치 중..."
    pip install gtts pydub
    if [ $? -eq 0 ]; then
        print_success "Google TTS (gTTS) 설치 완료"
    else
        print_warning "Google TTS (gTTS) 설치 실패"
    fi
fi

# Festival TTS (2순위, 한국어 품질 양호)
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

# espeak TTS (최종 백업)
if ! command -v espeak &> /dev/null; then
    print_status "espeak TTS 설치 중..."
    sudo apt-get install -y espeak
    print_success "espeak TTS 설치 완료"
else
    print_success "espeak TTS 이미 설치됨"
fi

# MP3 재생 도구
if ! command -v mpg123 &> /dev/null; then
    print_status "MP3 재생 도구 설치 중..."
    sudo apt-get install -y mpg123
    print_success "MP3 재생 도구 설치 완료"
else
    print_success "MP3 재생 도구 이미 설치됨"
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
echo "TTS 우선순위:"
echo "1. Google TTS (gTTS) - 한국어 품질 최고"
echo "2. Festival TTS - 한국어 품질 양호"
echo "3. Pico TTS - 가벼움"
echo "4. Flite TTS - 빠름"
echo "5. espeak TTS - 최종 백업"
echo ""
echo "문제 해결:"
echo "- 가상환경이 활성화되지 않으면: source jetson_tts_env/bin/activate"
echo "- 권한 문제 시: sudo chown -R $USER:$USER jetson_tts_env"
echo "- TTS 문제 시: pip install gtts pydub"
echo "=========================================="

print_success "젯슨 TTS 시스템 설치가 완료되었습니다!" 