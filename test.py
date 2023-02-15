# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import base64
import requests
from io import BytesIO
import banana_dev as banana


with open(f'demo.mp3','rb') as file:
    mp3bytes = BytesIO(file.read())
mp3 = base64.b64encode(mp3bytes.getvalue()).decode("ISO-8859-1")

model_payload = {"mp3BytesString":mp3}

res = requests.post("http://localhost:8000/",json=model_payload)

print(res.text)

