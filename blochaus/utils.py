from typing import Optional
from pycognito import Cognito
from os import getenv
import requests
import base64
import lzma
import json

# Your Cognito App settings
USER_POOL_ID = getenv("USER_POOL_ID")
CLIENT_ID = getenv("CLIENT_ID")

def decompress(data: str) -> list:
    return json.loads(lzma.decompress(base64.b64decode(data.encode())).decode())

def compress(data: list) -> str:
    return base64.b64encode(lzma.compress(json.dumps(data).encode())).decode()

def retrieve_auth(username: str, password: str) -> str:
    # Create Cognito object
    u = Cognito(USER_POOL_ID, CLIENT_ID, username=username)
    # Authenticate with SRP
    u.authenticate(password=password)
    # Access the ID token (the one you're manually copying now)
    return u.id_token

def verify_auth(username:str, password:str):
    try:
        retrieve_auth(username, password)
        return True
    except:
        return False
        
def retrieve_data(
    auth_token, start_date: Optional[str] = None, end_date: Optional[str] = None
):
    # === API Request ===
    BASE_URL = "https://portal.api.au-p.tilefive.com/customers/checkins"
    params = {
        "pdf": "false",
        "startDT": start_date,
        "endDT": end_date,
        "page": 1,
        "pagesize": 90,
        "order": "DESC",
        "sort": "createdAt",
    }
    headers = {
        "Authorization": auth_token,
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Origin": "https://blochaus.portal.approach.app",
        "Referer": "https://blochaus.portal.approach.app/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"❌ Request failed: {e}")
    return response.json()["data"]