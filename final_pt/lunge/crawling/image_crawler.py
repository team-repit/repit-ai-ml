# 파일명: image_crawler.py

import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# --- 설정 ---
# 검색할 키워드
SEARCH_KEYWORD = "런지 자세"
# 다운로드할 이미지 개수
DOWNLOAD_COUNT = 100
# 이미지를 저장할 디렉토리
SAVE_DIRECTORY = "lunge_images/raw"
# ----------------

def setup_driver():
    """Selenium WebDriver를 설정하고 반환합니다."""
    print("WebDriver 설정 중...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print("WebDriver 설정 완료.")
        return driver
    except Exception as e:
        print(f"WebDriver 설정 오류: {e}")
        print("ChromeDriver를 수동으로 설치하고 경로를 지정해야 할 수 있습니다.")
        return None

def scroll_to_end(driver):
    """페이지 끝까지 스크롤하여 모든 이미지를 로드합니다."""
    print("페이지 스크롤 중...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # 새 이미지 로딩 대기
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                # '결과 더보기' 버튼이 있으면 클릭
                more_results_button = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
                if more_results_button:
                    more_results_button.click()
                    time.sleep(2)
                else:
                    break
            except:
                break
        last_height = new_height
    print("스크롤 완료.")

def download_images(driver, keyword, max_count, save_dir):
    """이미지를 검색하고 다운로드합니다."""
    # 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"디렉토리 생성: {save_dir}")

    # 구글 이미지 검색 URL
    search_url = f"https://www.google.com/search?q={keyword}&tbm=isch"
    driver.get(search_url)

    scroll_to_end(driver)

    # 이미지 썸네일 요소 찾기
    # Google의 클래스 이름은 변경될 수 있습니다.
    # 'Q4LuWd', 'rg_i' 등이 일반적인 클래스 이름입니다.
    thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    print(f"총 {len(thumbnails)}개의 썸네일 발견.")

    downloaded_count = 0
    for i, img in enumerate(thumbnails):
        if downloaded_count >= max_count:
            break
        try:
            # 썸네일 클릭하여 큰 이미지 로드
            img.click()
            time.sleep(1.5)

            # 큰 이미지의 URL 추출
            # 이 클래스 이름도 변경될 수 있습니다. 'sFlh5c', 'n3VNCb' 등이 사용됩니다.
            actual_images = driver.find_elements(By.CSS_SELECTOR, 'img.sFlh5c')
            
            image_url = ""
            for actual_image in actual_images:
                src = actual_image.get_attribute('src')
                if src and 'http' in src:
                    image_url = src
                    break
            
            if not image_url:
                print(f"{i+1}번째 이미지 URL을 찾을 수 없습니다. 건너뜁니다.")
                continue

            # 이미지 다운로드
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                file_path = os.path.join(save_dir, f"{keyword.replace(' ', '_')}_{downloaded_count + 1}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"({downloaded_count + 1}/{max_count}) 이미지 다운로드 완료: {file_path}")
                downloaded_count += 1
            else:
                print(f"{i+1}번째 이미지 다운로드 실패 (상태 코드: {response.status_code}).")

        except Exception as e:
            print(f"{i+1}번째 이미지 처리 중 오류 발생: {e}")

    print(f"\n총 {downloaded_count}개의 이미지를 다운로드했습니다.")

if __name__ == "__main__":
    # --- 사전 준비 ---
    # 1. 이 스크립트를 실행하기 전에 필요한 라이브러리를 설치하세요.
    #    pip install selenium webdriver-manager requests
    #
    # 2. Chrome 브라우저가 설치되어 있어야 합니다.
    # -----------------
    
    driver = setup_driver()
    if driver:
        download_images(driver, SEARCH_KEYWORD, DOWNLOAD_COUNT, SAVE_DIRECTORY)
        driver.quit()
        print("작업 완료.")

