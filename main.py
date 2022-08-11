import os
from fastapi import FastAPI, HTTPException
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
app = FastAPI()

load_dotenv()
cloudinary.config( 
  cloud_name = os.getenv('CLOUD_NAME'), 
  api_key = os.getenv('API_KEY'),
  api_secret = os.getenv('API_SECRET'),
  secure = True
)
def check_valid_image(url):
  request_obj = requests.head(url)
  if request_obj.status_code == 200:
    return True
  else:
    return False

def download_file_from_cloudinary(url):
    request_obj = requests.get(url)
    open('car_damage/car_damage/real_test/input_image.jpg', 'wb').write(request_obj.content)
def uploadImage(path):
  details = cloudinary.uploader.upload(path)
  return details['url']
@app.get("/")
def read_root():
    return {"Welcome to": "Car Damage Detection"}
@app.get("/send_image")
def send_image(url):
    if not check_valid_image(url):
        raise HTTPException(status_code=400, detail="Invalid Image URL")
    try:
        download_file_from_cloudinary(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error Downloading Image - " + str(e))
    try:
        os.system("python eval.py --trained_model=./weights/yolact_plus_resnet50_car_damage_127_2675_interrupt1.pth --config=yolact_resnet50_car_damage_config --score_threshold=0.3 --top_k=15 --images=car_damage/car_damage/real_test:output_images")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error Evaluating Image - " + str(e))
    os.remove("car_damage/car_damage/real_test/input_image.jpg")
    try:
        image_url = uploadImage('output_images/input_image.png')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not upload image - " + str(e))
    os.remove("output_images/input_image.png")
    return {"image_url": image_url}
