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

# Initialize session state at the top level
if "analyzing" not in st.session_state:
    st.session_state["analyzing"] = False

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def initialize_model(model_id="google/gemma-3-12b-it"):
    """Initialize the model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# ... existing code for get_uniform_keyframes and analyze_video functions ...

class StreamlitTextStreamer(TextStreamer):
    """Custom streamer that writes to a Streamlit container."""
    def __init__(self, processor, container):
        self.processor = processor
        self.container = container
        self.text = ""

    def put(self, value):
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                tokens = [value.item()]
            else:
                tokens = value.tolist()
                if isinstance(tokens[0], list):  # Handle batch dimension
                    tokens = tokens[0]
            try:
                decoded = self.processor.decode(tokens, skip_special_tokens=True)
                if decoded:  # Only add non-empty strings
                    self.text += decoded
                    self.container.markdown(self.text)
            except Exception as e:
                print(f"Error decoding tokens: {e}")
        else:
            self.text += str(value)
            self.container.markdown(self.text)

    def end(self):
        """Called at the end of generation."""
        if self.text:
            self.container.markdown(self.text)

def analyze_video(video_path, text_prompt, output_dir, keyframe_paths, model, processor, stream_container, max_new_tokens=4096):
    """Analyze video keyframes using Gemma model."""
    if keyframe_paths is None:
        keyframe_paths = get_keyframes(video_path, output_dir)

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
    
    response = processor.decode(generation, skip_special_tokens=True)
    
    result = {
        'frames': keyframe_paths,
        'response': response,
        'time_taken': time_taken
    }
    
    return result

def main():
    st.title("Video Analysis")
    st.write("Upload a video to analyze its contents.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    
    # Slider for number of keyframes
    num_keyframes = st.slider("Number of keyframes to analyze", 
                            min_value=1, 
                            max_value=30, 
                            value=5)
    
    # Text area for custom prompt
    default_prompt = """How many cars are there in the scene? Respond with a single number."""
    prompt = st.text_area("Customize the prompt", value=default_prompt, height=300)

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

                    # Display keyframes
                    st.subheader("Extracted Keyframes")
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