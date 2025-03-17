# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextStreamer
from PIL import Image
import requests
import torch
import time

def initialize_model(model_id="google/gemma-3-12b-it"):
    """Initialize the model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_response(image_path, text_prompt, model=None, processor=None, max_new_tokens=4096):
    start_time = time.time()
    
    """Generate a response for an image and text prompt."""
    if model is None or processor is None:
        model, processor = initialize_model()

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        streamer = TextStreamer(processor)
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer
        )
        generation = generation[0][input_len:]

    end_time = time.time()
    time_taken = end_time - start_time

    return processor.decode(generation, skip_special_tokens=True), time_taken


if __name__ == "__main__":
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    prompt = "Describe this image in detail."
    
    model, processor = initialize_model()
    
    response, time_taken = generate_response(image_url, prompt, model, processor, max_new_tokens=100)
    print(f"### INFO: Time taken: {time_taken} seconds")

    import ipdb; ipdb.set_trace()

