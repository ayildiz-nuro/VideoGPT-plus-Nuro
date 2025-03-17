import streamlit as st
from gemma_infer_video import analyze_video, get_uniform_keyframes
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextStreamer
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import torch
import time
import os
import cv2
from PIL import Image
import tempfile
import gc

if "analyzing" not in st.session_state:
    st.session_state["analyzing"] = False

def initialize_model(model_id="google/gemma-3-12b-it"):
    """Initialize the model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

class StreamlitTextStreamer(TextStreamer):
    """Custom streamer that writes to both Streamlit container and terminal."""
    def __init__(self, processor, container):
        self.processor = processor
        self.container = container
        self.text = ""
        self.model_response_started = False

    def put(self, value):
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                tokens = [value.item()]
            else:
                tokens = value.tolist()
                if isinstance(tokens[0], list):  # Handle batch dimension
                    tokens = tokens[0]
            try:
                decoded = self.processor.decode(tokens, skip_special_tokens=False)
                if decoded:  # Only add non-empty strings
                    self.text += decoded
                    
                    # Print full output to terminal
                    print(decoded, end="", flush=True)
                    
                    # Check if we've reached the model's response
                    if "<start_of_turn>model" in self.text and not self.model_response_started:
                        self.model_response_started = True
                        # Reset text to only include content after "<start_of_turn>model"
                        self.text = self.text.split("<start_of_turn>model", 1)[1] if "<start_of_turn>model" in self.text else ""
                    
                    # Only update Streamlit with model's response
                    if self.model_response_started:
                        self.container.markdown(self.text)
                        
            except Exception as e:
                print(f"Error decoding tokens: {e}")
        else:
            self.text += str(value)
            print(str(value), end="", flush=True)  # Print to terminal
            
            # Only update Streamlit with model's response
            if self.model_response_started:
                self.container.markdown(self.text)

    def end(self):
        """Called at the end of generation."""
        if self.text:
            if self.model_response_started:
                self.container.markdown(self.text)
            print("")  # Add newline at the end in terminal

def analyze_video(video_path, text_prompt, output_dir, keyframe_paths, model, processor, stream_container, max_new_tokens=4096):
    """Analyze video keyframes using Gemma model."""
    if keyframe_paths is None:
        keyframe_paths = get_keyframes(video_path, output_dir)

    if model is None or processor is None:
        # Load model only when needed
        with st.spinner("Loading model within analyze_video..."):
            model, processor = initialize_model()
    
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
        # Use custom streamer that writes to Streamlit
        streamer = StreamlitTextStreamer(processor, stream_container)
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer
        )
        generation = generation[0][input_len:]

    end_time = time.time()
    time_taken = end_time - start_time
    
    response = processor.decode(generation, skip_special_tokens=False)
    
    result = {
        'frames': keyframe_paths,
        'response': response,
        'time_taken': time_taken
    }
    
    del model
    del processor
    return result

def cleanup_model_resources(model=None, processor=None):
    """Clean up model and processor resources."""
    if model is not None:
        try:
            model.to('cpu')  # Move model to CPU first
            del model
        except:
            pass
    
    if processor is not None:
        try:
            del processor
        except:
            pass
    
    gc.collect()  # Run garbage collector
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure CUDA operations are finished

def main():
    st.title("Nuro Video Conflict Analysis")
    st.write("Upload a video to analyze its contents.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    
    # # Slider for number of keyframes
    # st.subheader("Number of Keyframes")
    # num_keyframes = st.slider("Number of keyframes to analyze", 
    #                         min_value=1, 
    #                         max_value=30, 
    #                         value=5)
    
    # Hard-code the number of keyframes
    num_keyframes = 30
    
    # # Text area for custom prompt
    default_prompt =  """
    You are seeing key image frames of a video footage from a dashcam from the ego vehicle. You need to explain the conflicts that occur in the scene.
    Start by breaking the scene down into 3 main parts: 
    1. Description of road geometry. Static elements, and the environment. E.g. description of the traffic intersection, road markings, traffic lights, etc.
    2. Which agents are present in the scene, and what are they doing to cause a conflict? Be sure to mention their directions of travel from the perspective of the ego vehicle, and clarify whether their trajectories are parallel, orthogonal, or intersecting. Explain how their movements affect the potential conflict with the ego vehicle."
    3. What is the stage of the scenario progression? Each of these stages should capture agents' progress, intent, and interactions with the ego vehicle. Be very specific about the intent of the agents in the scene, and how they are causing a conflict with the ego vehicle.

    Keep the description of each part limited to a single sentence.
    Only reply with the answer, don't include anything else.
    """
    # st.subheader("Customize the Prompt")
    # prompt = st.text_area("Customize the prompt", value=default_prompt, height=300)
    
    # Use default prompt directly
    prompt = default_prompt

    # Show analysis and cancel button if analyzing
    if st.session_state["analyzing"]:
        if st.button("Cancel Analysis"):
            st.session_state["analyzing"] = False
            st.rerun()
        else:
            with st.spinner("Loading model..."):
                model, processor = initialize_model()

            # Create a temporary directory for the uploaded file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file to temporary directory
                temp_video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_video_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Create output directory for keyframes
                output_dir = os.path.join(temp_dir, "keyframes_output")
                
                with st.spinner("Extracting keyframes..."):
                    keyframe_paths = get_uniform_keyframes(
                        video_path=temp_video_path,
                        output_dir=output_dir,
                        num_keyframes=num_keyframes
                    )

                    # Display keyframes in an accordion
                    with st.expander("Extracted Keyframes", expanded=False):
                        cols = st.columns(min(5, num_keyframes))
                        for idx, (frame_path, col) in enumerate(zip(keyframe_paths, cols * (len(keyframe_paths) // len(cols) + 1))):
                            if idx < len(keyframe_paths):
                                col.image(frame_path, caption=f"Frame {idx+1}")

                st.subheader("Analysis Results")
                # Create an empty container for streaming output
                stream_container = st.empty()
                
                with st.spinner("Analyzing video..."):
                    result = analyze_video(
                        video_path=temp_video_path,
                        text_prompt=prompt,
                        output_dir=output_dir,
                        keyframe_paths=keyframe_paths,
                        model=model,
                        processor=processor,
                        stream_container=stream_container
                    )

                # Display final timing
                st.write(f"Time taken: {result['time_taken']:.2f} seconds")

            # After analysis is complete, reset the analyzing state
            st.session_state["analyzing"] = False

    # Show analyze button if not analyzing and file is uploaded
    elif uploaded_file:
        if st.button("Analyze Video"):
            st.session_state["analyzing"] = True
            st.rerun()

if __name__ == "__main__":
    main() 