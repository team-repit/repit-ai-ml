#!/usr/bin/env python3
"""
젯슨에서 다양한 TTS 모듈의 한국어 품질을 테스트하는 스크립트
"""

import subprocess
import os
import time
from gtts import gTTS

def test_espeak():
    """espeak TTS 테스트"""
    print("🔊 espeak TTS 테스트 중...")
    try:
        test_text = "안녕하세요. 스쿼트 자세를 교정해주세요."
        subprocess.run(['espeak', '-s', '120', test_text], check=True)
        print("✅ espeak TTS 성공")
        return True
    except Exception as e:
        print(f"❌ espeak TTS 실패: {e}")
        return False

def test_festival():
    """Festival TTS 테스트"""
    print("🔊 Festival TTS 테스트 중...")
    try:
        test_text = "안녕하세요. 스쿼트 자세를 교정해주세요."
        subprocess.run(['festival', '--tts', f'(SayText "{test_text}")'], check=True)
        print("✅ Festival TTS 성공")
        return True
    except Exception as e:
        print(f"❌ Festival TTS 실패: {e}")
        return False

def test_pico():
    """Pico TTS 테스트"""
    print("🔊 Pico TTS 테스트 중...")
    try:
        test_text = "안녕하세요. 스쿼트 자세를 교정해주세요."
        wav_file = "test_pico.wav"
        subprocess.run(['pico2wave', '-w', wav_file, test_text], check=True)
        subprocess.run(['aplay', wav_file], check=True)
        os.remove(wav_file)
        print("✅ Pico TTS 성공")
        return True
    except Exception as e:
        print(f"❌ Pico TTS 실패: {e}")
        return False

def test_flite():
    """Flite TTS 테스트"""
    print("🔊 Flite TTS 테스트 중...")
    try:
        test_text = "안녕하세요. 스쿼트 자세를 교정해주세요."
        subprocess.run(['flite', '-t', test_text], check=True)
        print("✅ Flite TTS 성공")
        return True
    except Exception as e:
        print(f"❌ Flite TTS 실패: {e}")
        return False

def test_gtts():
    """Google TTS 테스트"""
    print("🔊 Google TTS 테스트 중...")
    try:
        test_text = "안녕하세요. 스쿼트 자세를 교정해주세요."
        tts = gTTS(text=test_text, lang='ko')
        mp3_file = "test_gtts.mp3"
        tts.save(mp3_file)
        
        # MP3 재생 시도
        players = ['mpg123', 'ffplay', 'mpv', 'cvlc']
        played = False
        
        for player in players:
            try:
                subprocess.run([player, mp3_file], check=True, 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ Google TTS 성공 ({player}로 재생)")
                played = True
                break
            except:
                continue
        
        if not played:
            # WAV로 변환 후 재생
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_file)
                wav_file = "test_gtts.wav"
                audio.export(wav_file, format="wav")
                subprocess.run(['aplay', wav_file], check=True)
                os.remove(wav_file)
                print("✅ Google TTS 성공 (WAV 변환 후 재생)")
                played = True
            except:
                print("⚠️ Google TTS 파일 생성 성공, 재생 실패")
        
        os.remove(mp3_file)
        return played
        
    except Exception as e:
        print(f"❌ Google TTS 실패: {e}")
        return False

def test_all_tts():
    """모든 TTS 모듈 테스트"""
    print("🎯 젯슨 TTS 품질 테스트 시작!")
    print("=" * 50)
    
    results = {}
    
    # 각 TTS 모듈 테스트
    results['espeak'] = test_espeak()
    time.sleep(1)
    
    results['festival'] = test_festival()
    time.sleep(1)
    
    results['pico'] = test_pico()
    time.sleep(1)
    
    results['flite'] = test_flite()
    time.sleep(1)
    
    results['gtts'] = test_gtts()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 TTS 테스트 결과 요약")
    print("=" * 50)
    
    for tts_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{tts_name:10}: {status}")
    
    # 권장 TTS 추천
    working_tts = [name for name, success in results.items() if success]
    
    if working_tts:
        print(f"\n🎉 작동하는 TTS: {', '.join(working_tts)}")
        
        # 품질 기반 추천
        if 'gtts' in working_tts:
            print("🥇 1순위 추천: Google TTS (한국어 품질 최고)")
        if 'festival' in working_tts:
            print("🥈 2순위 추천: Festival TTS (안정적)")
        if 'pico' in working_tts:
            print("🥉 3순위 추천: Pico TTS (가벼움)")
        if 'flite' in working_tts:
            print("🏅 4순위 추천: Flite TTS (빠름)")
        if 'espeak' in working_tts:
            print("🏅 5순위 추천: espeak TTS (기본)")
    else:
        print("\n❌ 작동하는 TTS가 없습니다.")
        print("TTS 도구들을 설치해주세요.")
    
    print("=" * 50)

if __name__ == "__main__":
    test_all_tts() 