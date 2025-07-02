import os
import requests
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask import Flask, render_template

# Set eager execution to True
tf.config.experimental_run_functions_eagerly(True)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def download_image(image_url):
    # Create a 'downloaded_images' directory inside the 'static' folder
    static_dir = "static"
    downloaded_images_dir = os.path.join(static_dir, "downloaded_images")
    os.makedirs(downloaded_images_dir, exist_ok=True)

    try:
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {image_url}")
        print(e)
        return None

    # Save the image to the 'static/downloaded_images' directory
    image_name = os.path.basename(image_url)
    image_path = os.path.join(downloaded_images_dir, image_name)

    with open(image_path, 'wb') as f:
        f.write(image_response.content)
        print("Image downloaded: {image_url}")

    return image_path


def calculate_similarity(image_path1, image_path2):
    img1 = Image.open(image_path1).resize((224, 224))
    img2 = Image.open(image_path2).resize((224, 224))

    image1 = img_to_array(img1)
    image2 = img_to_array(img2)

    image1 = preprocess_input(image1)
    image2 = preprocess_input(image2)

    image1 = tf.expand_dims(image1, axis=0)
    image2 = tf.expand_dims(image2, axis=0)

    features1 = model.predict(image1)
    features2 = model.predict(image2)

    similarity_score = tf.keras.losses.cosine_similarity(features1, features2, axis=1)
    similarity_percentage = (similarity_score + 1) * 50  
    print(similarity_percentage)

    return similarity_score
     
     
   

# Initialize the Flask app
app = Flask(__name__, template_folder='template',static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    image_url1 = "C:/Users/student/Downloads/today/today/face.jpeg"
    image_url2 = "https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg"

    # Download the images and get the downloaded paths
    image_path1 = image_url1
    image_path2 = download_image(image_url2)

    if image_path1 is not None and image_path2 is not None:
        # Calculate the similarity score between the images
        similarity_score = calculate_similarity(image_path1, image_path2)

        # Check cyclone possibility
        cyclone_possible = check_cyclone_possibility(similarity_score)

        # Return the result using the result.html template
        return render_template('result.html', similarity_score=similarity_score, cyclone_possible=cyclone_possible)
    else:
        return "Failed to download or process images."

def check_cyclone_possibility(similarity_score):
    if -1 <= similarity_score <= -0.75:
        return True
    else:
        return False
  

   
if __name__ == '__main__':
    app.run()
