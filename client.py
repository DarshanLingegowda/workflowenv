# Minimal client for OpenEnv validation

import requests

BASE_URL = "http://localhost:7860"

def reset(task_id="easy"):
    return requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}).json()

def step(action):
    return requests.post(f"{BASE_URL}/step", json=action).json()

def state():
    return requests.get(f"{BASE_URL}/state").json()
