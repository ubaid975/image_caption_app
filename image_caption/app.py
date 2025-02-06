import PIL.Image
from transformers import pipeline
import PIL
import cv2
import gradio as gr
pipe=pipeline(model="Salesforce/blip-image-captioning-large")

def caption(image):
    image_1=PIL.Image.fromarray(image)
    pred=pipe.predict(image_1)
    return str(pred[0]['generated_text'])
app=gr.Interface(fn=caption,inputs=['image'],outputs=['label'],title='Image Caption Genrator')
app.launch(share=True)

