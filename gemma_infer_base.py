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


def get_uniform_keyframes(video_path, output_dir, num_keyframes=5):
    """
    Extract num_keyframes frames from the video, 
    spaced uniformly from the first to the last available frame.

    1) Reads all frames into a list (frames in memory).
    2) Computes uniform indices across that list.
    3) Saves those frames as images.
    """
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)

    # Read all frames sequentially
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Compute uniformly spaced indices across all frames
    if num_keyframes == 1:
        # If only 1 keyframe requested, just take first (or last)
        frame_indices = [0]
    else:
        # Distribute frames from 0 to total_frames-1 (inclusive)
        frame_indices = [
            int(i * (total_frames - 1) / (num_keyframes - 1))
            for i in range(num_keyframes)
        ]

    # Extract & save the selected frames
    keyframe_paths = []
    for i, idx in enumerate(frame_indices):
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        Image.fromarray(rgb_frame).save(frame_path)
        keyframe_paths.append(frame_path)

    return keyframe_paths


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

def analyze_video(video_path, text_prompt, output_dir, keyframe_paths, model, processor, max_new_tokens=4096):
    """Analyze video keyframes using Gemma model."""

    # Extract keyframes
    if keyframe_paths is None:
        keyframe_paths = get_keyframes(video_path, output_dir)

    # Initialize model
    if model is None or processor is None:
        model, processor = initialize_model()
    
    # Create a single message with all images
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": Image.open(frame_path)} for frame_path in keyframe_paths],
                {"type": "text", "text": text_prompt}
            ]
        }
    ]

    print(f"\nAnalyzing {len(keyframe_paths)} frames together...")
    start_time = time.time()
    
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
    
    response = processor.decode(generation, skip_special_tokens=True)
    
    result = {
        'frames': keyframe_paths,
        'response': response,
        'time_taken': time_taken
    }
    
    print(f"Time taken: {time_taken:.2f} seconds")
    return result

if __name__ == "__main__":
    video_path = "/home/ayildiz/sample_videos/video_nexar_perf_trim.mov"
    # video_path = "/home/ayildiz/sample_videos/video_nexar_badperf_trim.mov"
    output_dir = "keyframes_output"
    prompt = """
    You are seeing key image frames of a video footage from a dashcam from the ego vehicle.
    These images as sequential frames from a video.
    You need to explain the conflicts that occur in the scene.
    Start by breaking the scene down into 3 main parts: 
    1. Description of road geometry. Static elements, and the environment. E.g. description of the traffic intersection, road markings, traffic lights, etc.
    2. Which agents are present in the scene, and what are they doing to cause a conflict? Be sure to mention their directions of travel from the perspective of the ego vehicle, and clarify whether their trajectories are parallel, orthogonal, or intersecting. Explain how their movements affect the potential conflict with the ego vehicle.
    3. What is the stage of the scenario progression? Each of these stages should capture agents' progress, intent, and interactions with the ego vehicle. Be very specific about the intent of the agents in the scene, and how they are causing a conflict with the ego vehicle.

    When you reply, think about the progression of the scenario through the frames.
    Explain the movements of each agent (other cars, pedestrians, cyclists, etc.) within the scene, and how they interact with the ego vehicle.
    When you are referring to other agents, make sure to include their direction of travel, and their speed. Also indicate identifying features of the them, such as colors of vehicles or bikes, or clothing of pedestrians.
    The agent or agents that are causing the conflict are usually the ones that are closest to the ego vehicle and their intended trajectories are intersecting with the ego vehicle's trajectory.
    There might be other agents in the scene that are not moving (i.e. stopped at a red light or parked at the side of the road), or moving in a direction that does not intersect with the ego vehicle's trajectory. Mention their presence, but also their irrelevance to the conflict.

    There are certain cues that imply a crash has happened, i.e. a cyclist or pedestrian falling down (or disappears unexplainably across frames), cars coming to a stop, etc. Make sure to detect them.
    """
    num_keyframes = 10

    keyframe_paths = get_uniform_keyframes(video_path=video_path,
                                           output_dir=output_dir,
                                           num_keyframes=num_keyframes)
    
    model, processor = initialize_model()
    
    result = analyze_video(
        video_path=video_path,
        text_prompt=prompt,
        output_dir=output_dir,
        keyframe_paths=keyframe_paths,
        model=model,
        processor=processor
        )
    
    # Print results
    print(f"\nAnalyzed {len(result['frames'])} frames")
    print(f"Response: {result['response']}")
    print(f"Time taken: {result['time_taken']:.2f} seconds")

    import ipdb; ipdb.set_trace()

