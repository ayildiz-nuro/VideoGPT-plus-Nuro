# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
from transformers import TextStreamer
import time

def initialize_model(model_id="google/gemma-3-12b-it"):
    """Initialize the model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_response(image_path, text_prompt, model=None, processor=None, max_new_tokens=4096):
    """Generate a response for an image and text prompt.
    
    Args:
        image_path: Path to local image file or URL to an image
        text_prompt: Text prompt to send to the model
        model: Pre-loaded model (optional)
        processor: Pre-loaded processor (optional)
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple of (generated response text, time taken in seconds)
    """
    start_time = time.time()
    
    if model is None or processor is None:
        model, processor = initialize_model()
    
    # Handle both local files and URLs
    if image_path.startswith(('http://', 'https://')):
        # It's a URL, keep as is
        image = image_path
    else:
        # It's a local file, load it with PIL
        from PIL import Image
        image = Image.open(image_path)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
    # image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    # prompt = "Describe this image in detail."

    image_path = '/home/ayildiz/VideoGPT-plus-Nuro/car_city.jpg'
    prompt = """
    Answer the following questions based on the image. Reply with a single word, yes or no, for each question.

    1. Is the car in the image yellow?  
    2. Is the car a vintage model?  
    3. Is the car a Fiat?  
    4. Is the car driving on a cobblestone street?  
    5. Is the license plate blue and white?  
    6. Is the car in motion?  
    7. Is the roof of the car open?  
    8. Is the image taken in a European city?  
    9. Is the car a two-door vehicle?  
    10. Is the car small in size?  
    11. Are there pedestrians in the background?  
    12. Is the driver visible through the windshield?  
    13. Is the car's front bumper silver?  
    14. Does the car have round headlights?  
    15. Is there a shop with a display window in the background?  
    16. Is the car in good condition?  
    17. Is the image taken during the daytime?  
    18. Is the street narrow?  
    19. Are there potted plants visible in the image?  
    20. Is the car's side mirror round?  
    21. Is the car's front grille small?  
    22. Is the image taken from a slightly tilted angle?  
    23. Are there reflections visible on the car's windshield?  
    24. Is the car's license plate number "43564"?  
    25. Is the car's manufacturer logo red?  
    26. Are the tires black with silver hubcaps?  
    27. Are the windows of the car rolled down?  
    28. Is the car parked?  
    29. Is there a woman walking in the background?  
    30. Is there a man wearing a hat in the background?  
    31. Are there awnings above the shops?  
    32. Are there any bicycles in the image?  
    33. Is there a visible shadow of the car on the ground?  
    34. Are there multiple people in the car?  
    35. Is the driver wearing sunglasses?  
    36. Is the car missing a side mirror?  
    37. Are there plants placed in rectangular pots?  
    38. Is the image taken in a tourist area?  
    39. Are there any visible street signs?  
    40. Is the car facing the viewer?  
    41. Is the road made of asphalt?  
    42. Is there a reflection of buildings on the car's windshield?  
    43. Does the car have a retractable fabric sunroof?  
    44. Is the car moving at a fast speed?  
    45. Is there any visible text on the shop signs?  
    46. Is the car's design reminiscent of the 1960s?  
    47. Is the driver wearing a seatbelt?  
    48. Is the license plate attached to the front bumper?  
    49. Is the car's paint shiny?  
    50. Is the image taken from a low angle?  
    """
    
    response, time_taken = generate_response(image_path, prompt)
    print(f"Response: {response}")
    print(f"Time taken: {time_taken:.2f} seconds")

    import ipdb; ipdb.set_trace()

