import requests

# Path to the image you want to send
image_path = "C:/Users/Administrator/Desktop/Signscribe/Signscribe/asl_alphabet/C/C24.jpg"

url = "http://127.0.0.1:5000/predict"

# Open the image file and send it as a POST request
with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

# Debugging the response
try:
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Error decoding JSON:", e)
    print("Response text:", response.text)  # Show the raw response text
