import requests

data = {"text": "Congratulations! You have won a free gift card. Redeem now!"}
response = requests.post("http://127.0.0.1:5000/analyze", json=data)

print(response.json())
