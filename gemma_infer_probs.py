# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
from transformers import TextStreamer
import time
import torch.nn.functional as F

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
        Tuple of (generated response text, time taken in seconds, token probabilities)
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
        # For streaming output
        streamer = TextStreamer(processor)
        
        # For getting probabilities, we need a separate generation call
        # that doesn't use the streamer but returns scores
        generation_with_scores = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # # For the actual streaming output to the user
        # model.generate(
        #     **inputs, 
        #     max_new_tokens=max_new_tokens, 
        #     do_sample=False,
        #     streamer=streamer
        # )
        
        # Process the scores to get probabilities using softmax
        generated_tokens = generation_with_scores.sequences[0][input_len:]
        
        # Convert scores to percentage probabilities using softmax
        token_probs = []
        for i, score in enumerate(generation_with_scores.scores):
            # Apply softmax to get probabilities
            probs = F.softmax(score[0], dim=-1)
            
            # Get the token that was actually generated
            token_id = generated_tokens[i].item()
            
            # Get the probability for that token (as a percentage between 0 and 1)
            token_prob = probs[token_id].item()
            token_probs.append((token_id, token_prob))

    end_time = time.time()
    time_taken = end_time - start_time
    
    # Decode the generated text
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, time_taken, token_probs

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
    
    model, processor = initialize_model()  # Initialize both model and processor first
    response, time_taken, token_probs = generate_response(image_path, prompt, model, processor)
    print(f"Response: {response}")
    print(f"Time taken: {time_taken:.2f} seconds")
    
    # Print token probabilities using the initialized processor
    print("Token probabilities:")
    for token_id, token_prob in token_probs:
        token = processor.decode([token_id])
        print(f"Token: '{token}', Probability: {token_prob:.4f}")

    import ipdb; ipdb.set_trace()

