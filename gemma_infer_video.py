# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextStreamer
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import torch
import time
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import cv2
from PIL import Image
import argparse
import numpy as np


# @contextmanager
# def suppress_output():
#     """Suppress print output."""
#     with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
#         yield

def get_keyframes(video_path, output_dir, num_keyframes=5):
    os.makedirs(output_dir, exist_ok=True)

    """Extract keyframes using Katna from the video."""
    vd = Video()
    diskwriter = KeyFrameDiskWriter(location=output_dir)
    
    # with suppress_output():
    vd.extract_video_keyframes(
        no_of_frames=num_keyframes, 
        file_path=video_path, 
        writer=diskwriter
    )
    
    # Get list of extracted keyframe paths
    keyframe_paths = [
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]
    return keyframe_paths


def get_uniform_keyframes(video_path, num_keyframes=30):
    """Extract uniform keyframes from the video."""
    print(f"Extracting {num_keyframes} keyframes from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print("Error: Could not read video frames.")
        return []
    
    # Calculate frame indices to extract
    indices = [int(i * total_frames / num_keyframes) for i in range(num_keyframes)]
    
    keyframes = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keyframes.append(Image.fromarray(frame_rgb))
    
    cap.release()
    print(f"Extracted {len(keyframes)} keyframes.")
    return keyframes


def initialize_model(model_id="google/gemma-3-12b-it"):
    """Initialize the model and processor."""
    print("Loading model and processor...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded successfully.")
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

def analyze_video(video_path, text_prompt, model, processor, max_new_tokens=4096):
    """Analyze video keyframes using Gemma model."""
    print(f"Analyzing video: {video_path}")
    
    # Extract keyframes
    keyframes = get_uniform_keyframes(video_path)
    
    if not keyframes:
        return "Error: Failed to extract keyframes from the video."
    
    # Prepare messages for the model
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": frame} for frame in keyframes],
                {"type": "text", "text": text_prompt}
            ]
        }
    ]

    start_time = time.time()
    
    # Process the messages
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    # Generate response
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generation = generation[0][input_len:]

    end_time = time.time()
    time_taken = end_time - start_time
    
    # Decode the response
    response = processor.decode(generation, skip_special_tokens=True)
    
    print(f"Analysis completed in {time_taken:.2f} seconds.")
    return response

DEFAULT_PROMPT = """
You are seeing key image frames of a video footage from a dashcam from the ego vehicle. You need to explain the conflicts that occur in the scene.
Start by breaking the scene down into 3 main parts: 
1. Description of road geometry. Static elements, and the environment. E.g. description of the traffic intersection, road markings, traffic lights, etc.
2. Which agents are present in the scene, and what are they doing to cause a conflict? Be sure to mention their directions of travel from the perspective of the ego vehicle, and clarify whether their trajectories are parallel, orthogonal, or intersecting. Explain how their movements affect the potential conflict with the ego vehicle."
3. What is the stage of the scenario progression? Each of these stages should capture agents' progress, intent, and interactions with the ego vehicle. Be very specific about the intent of the agents in the scene, and how they are causing a conflict with the ego vehicle.

Keep the description of each part limited to a single sentence.
Only reply with the answer, don't include anything else.
"""

def infer(video_path, output_path, prompt=None):
    """Main inference function to be called from other scripts or command line."""
    if prompt is None:
        prompt = DEFAULT_PROMPT
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    # Initialize model and processor
    model, processor = initialize_model()
    
    try:
        # Analyze the video
        result = analyze_video(video_path, prompt, model, processor)
        
        # Save the result to the output file
        with open(output_path, 'w') as f:
            f.write(result)
            
        print(f"Result saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return False
    
    finally:
        # Clean up resources
        if model is not None:
            try:
                model.to('cpu')
                del model
            except:
                pass
        
        if processor is not None:
            try:
                del processor
            except:
                pass
        
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video using Gemma model")
    parser.add_argument("--video_path", required=True, help="Path to the video file")
    parser.add_argument("--output_path", required=True, help="Path to save the analysis result")
    parser.add_argument("--prompt", default=None, help="Custom prompt for analysis")
    
    args = parser.parse_args()
    
    success = infer(args.video_path, args.output_path, args.prompt)
    
    # Exit with appropriate code
    exit(0 if success else 1)
