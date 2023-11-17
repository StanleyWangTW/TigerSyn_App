from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import subprocess
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
test_input_file_path = os.path.join(ROOT_DIR, 'test_input_files\CANDI_BPDwoPsy_030_1mm.nii.gz')

def start_flask_app():
    # 啟動 Flask 應用
    flask_process = subprocess.Popen(['python', 'app.py'])  # 替換為你的 Flask 應用啟動腳本的名稱

    # 等待一些時間，確保 Flask 應用已經啟動
    time.sleep(3)

    return flask_process


def stop_flask_app(process):
    # 關閉 Flask 應用
    process.terminate()
    process.wait()


def test_upload_and_process():
    # 啟動 Flask 應用
    flask_process = start_flask_app()
    
    # 初始化 Selenium WebDriver
    driver = webdriver.Chrome()

    try:
        driver.get('http://127.0.0.1:5000')

        # Enter patient_id
        driver.find_element(By.ID, 'exampleFormControlInput1').send_keys('00000')
        time.sleep(1)
        # Push login button
        driver.find_element(By.XPATH, '/html/body/form/div/p/input').click()

        # Select input file
        input_file = driver.find_element(By.ID, 'formFile')
        input_file.send_keys(test_input_file_path)
        # Submit input file
        driver.find_element(By.XPATH, '/html/body/form[1]/div[2]/button').click()
        time.sleep(1)
        # Check Image is displayed or not
        display_image = driver.find_element(By.XPATH, '/html/body/div/div/div[2]/img')
        assert display_image.is_displayed()

        # Press logout
        driver.find_element(By.XPATH, '/html/body/form[4]/div/button').click()
        # Check if back to home page
        assert driver.current_url == 'http://127.0.0.1:5000/'

    finally:
        # Close browser
        driver.quit()
        # Close Flask server
        stop_flask_app(flask_process)


if __name__ == '__main__':
    test_upload_and_process()