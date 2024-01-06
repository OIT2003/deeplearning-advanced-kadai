from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
#from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import random

def predict(request):
    model = VGG16(weights = "imagenet")
    #save_model(model, "vgg16.h5")
    
    if request.method == "GET":
        form = ImageUploadForm()
        
        return render(request, "home.html", {"form": form})
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            img_file = form.cleaned_data["image"]
            img_file = BytesIO(img_file.read())
            unknown_img = load_img(img_file, target_size = (224, 224))
            unknown_array = img_to_array(unknown_img)
            unknown_array = unknown_array.reshape((1, 224, 224, 3))
            unknown_array = preprocess_input(unknown_array)
            result = model.predict(unknown_array)
            #print(decode_predictions(result))

            #prediction = random.choice(["猫", "犬"])
            prediction = decode_predictions(result)
            
            img_data = request.POST.get("img_data")
            print(prediction[0][1][1])
            return render(request, "home.html", {"form": form, "prediction": prediction, "img_data": img_data})
        else:
            form = ImageUploadForm()
            
            return render(request, "home.html", {"form": form})