from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    raw_image = Image.fromarray(image)
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

demo = gr.Interface(
    fn=generate_caption,
    inputs=[gr.Image(label="Upload Image")],
    outputs=[gr.Textbox(label="Generated Caption")],
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption using the BLIP model.",
)

demo.launch()