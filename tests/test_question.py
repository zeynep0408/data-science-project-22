import pytest
import pandas as pd
import numpy as np
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import *

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Geralt","Ciri","Leshen","Drowner","Vesemir"],
        "character_class": ["witcher","witcher","monster","monster","witcher"],
        "region": ["Kaer Morhen","Skellige","Swamp","Swamp","Kaer Morhen"],
        "is_monster": [0,0,1,1,0]
    })

def test_class_distribution(sample_df):
    dist = get_class_distribution(sample_df, "is_monster")
    assert sum(dist.values()) == 5

def test_label_encoding(sample_df):
    df = apply_label_encoding(sample_df.copy(), "character_class")
    assert df["character_class"].dtype in [np.int32, np.int64]

def test_one_hot_encoding(sample_df):
    df = apply_one_hot_encoding(sample_df.copy(), "region")
    assert "region_Swamp" in df.columns

def test_down_up_sampling(sample_df):
    df_down = down_sample(sample_df.copy(), "is_monster")
    df_up = up_sample(sample_df.copy(), "is_monster")
    assert df_down.shape[0] <= 5
    assert df_up.shape[0] >= 5

def test_split_features_target(sample_df):
    X, y = split_features_target(sample_df, "is_monster")
    assert len(X) == len(y)

def test_summarize(sample_df):
    summary = summarize_dataset(sample_df)
    assert "shape" in summary

def send_post_request(url: str, data: dict, headers: dict = None):
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # hata varsa exception fırlatır
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
    except Exception as err:
        print(f"Other error occurred: {err}")

class ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1

def run_tests():
    collector = ResultCollector()
    pytest.main(["tests"], plugins=[collector])
    print(f"\nToplam Başarılı: {collector.passed}")
    print(f"Toplam Başarısız: {collector.failed}")
    
    user_score = (collector.passed / (collector.passed + collector.failed)) * 100
    print(round(user_score, 2))
    
    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": 403,
        "project_id": 697,
        "user_score": round(user_score, 2),
        "is_auto": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    send_post_request(url, payload, headers)

if __name__ == "__main__":
    run_tests()
