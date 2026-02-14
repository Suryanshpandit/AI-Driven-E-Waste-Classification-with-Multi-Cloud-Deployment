from flask import Flask, request, jsonify
import csv
from datetime import datetime

app = Flask(__name__)

@app.route("/log", methods=["POST"])
def log_result():
    data = request.json
    with open("cloud2_logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), data["image_name"], data["prediction"]])
    return jsonify({"status": "logged on cloud 2"})

if __name__ == "__main__":
    app.run(port=6000)
