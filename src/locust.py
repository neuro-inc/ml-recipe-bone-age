import os
import json
import random
from pathlib import Path
import logging
import csv
import base64

from locust import task
from locust.contrib.fasthttp import FastHttpUser

DATA_DIR = Path(os.environ["DATA_DIR"])
IMG_DIR = DATA_DIR / "images"
DATA_FILE = DATA_DIR / "data.csv"

DATA = {}
with open(DATA_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        try:
            DATA[row[1].strip()] = {
                "boneage": int(row[2].strip()),
                "male": row[3].strip() == "True"
                }
        except Exception:
            # dirty way to handle header row
            pass

# Allowed model error in months
EPSILON = 20


class LoadGenerator(FastHttpUser):
    @task
    def prediction(self) -> None:
        img_id, data = random.choice(list(DATA.items()))
        target_image = IMG_DIR / f"{img_id}.png"

        payload = dict(strData=base64.encodebytes(target_image.read_bytes()))
        with self.client.post("", files=payload, catch_response=True) as response:
            model_response = json.loads(response.text)["data"]
            model_predicted_boneage = float(model_response["tensor"]["values"][0])

            if abs(model_predicted_boneage - data["boneage"]) >= EPSILON:
                    response.failure(
                        "Wrong model prediction, "
                        f"got {int(model_predicted_boneage)}, "
                        f"need {data['boneage']}!"
                    )
                    logging.warning(
                        "Wrong model prediction, "
                        f"got {int(model_predicted_boneage)}, "
                        f"need {data['boneage']}!"
                    )
