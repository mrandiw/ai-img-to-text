import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import gradio as gr

# Initialize model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

# OCR function
def ocr_image(image):
    inputs = processor(image, return_tensors="pt").to(device)
    
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=4096,
    )
    
    result = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image OCR",
    description="Upload an image to extract text using GOT-OCR-2.0"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
