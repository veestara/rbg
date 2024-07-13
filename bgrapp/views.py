import numpy as np
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import UploadedImage
from .background_remover import load_model, remove_background
import cv2
import os
from .forms import ImageUploadForm

# Load the model when the server starts
model = load_model()

def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            model = load_model()
            processed_image = remove_background(image)
            return render(request, 'result.html', {'processed_image': processed_image})
    else:
        form = ImageUploadForm()
    return render(request, 'home.html', {'form': form})

def remove_background_view(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        uploaded_image = UploadedImage(original_image=image)
        uploaded_image.save()

        image_path = uploaded_image.original_image.path
        try:
            result = remove_background(image_path, model)
        except Exception as e:
            return HttpResponse(f"Error: {e}")

        processed_image_path = f'media/processed/processed_image_{uploaded_image.id}.png'

        # Convert result to numpy array
        result_np = np.array(result)

        # Check if result_np is not empty
        if result_np.size > 0:
            # Save processed image using OpenCV
            cv2.imwrite(processed_image_path, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

            # Update the processed_image field and save
            uploaded_image.processed_image.name = f'processed/processed_image_{uploaded_image.id}.png'
            uploaded_image.save()

            return render(request, 'result.html', {'result_image': uploaded_image.processed_image.url})
        else:
            return HttpResponse("Error: Processed image is empty.")
    
    return HttpResponse("Error: Please upload an image.")
