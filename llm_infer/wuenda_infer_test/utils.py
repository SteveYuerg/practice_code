##Setup for predibase token access, endpoint url and headers

import os
from dotenv import load_dotenv, find_dotenv


# Initailize global variables
_ = load_dotenv(find_dotenv())

predibase_api_token = os.getenv("DEEPSEEK_API_KEY")

endpoint_url = "https://api.deepseek.com"

headers = {
    "Authorization": f"Bearer {predibase_api_token}"
}