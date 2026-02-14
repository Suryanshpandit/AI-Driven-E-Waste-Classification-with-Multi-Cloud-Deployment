import requests

url = "http://127.0.0.1:5000/predict"

image_paths = ["t_1.jpg", "t_2.jpg","t_3.jpg"]  # add more if you want

for img_path in image_paths:
    with open(img_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)
        print(img_path, "->", response.json())

