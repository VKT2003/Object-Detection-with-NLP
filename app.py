from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont  # Import ImageFont
import gradio as gr
import requests
import random

def detect_objects(image):
    # Load the pre-trained DETR model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    detected_objects = []
    for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        box = [round(i, 2) for i in box.tolist()]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle(box, outline=color, width=3)
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        # Larger and bolder font
        draw.text((box[0], box[1]), label_text, fill=color,)
        detected_objects.append(model.config.id2label[label.item()])

    return image, ', '.join(detected_objects)


def upload_image(file):
    image = Image.open(file.name)
    image_with_boxes, detected_objects = detect_objects(image)
    return image_with_boxes, detected_objects

iface = gr.Interface(
    fn=upload_image,
    inputs="file",
    outputs=["image", "text"],
    title="Object Detection",
    description="Upload an image and detect objects using DETR model.",
    allow_flagging=False,
    css="style.css"  # Path to your custom CSS file
)

iface.launch()
