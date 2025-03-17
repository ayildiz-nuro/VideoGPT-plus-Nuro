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

def main():
    st.title("Video Analysis with Gemma-3")
    st.write("Upload a video to analyze it with the Gemma-3 model")

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

    if uploaded_file and st.button("Analyze Video"):
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

            with st.spinner("Analyzing video..."):
                result = analyze_video(
                    video_path=temp_video_path,
                    text_prompt=prompt,
                    output_dir=output_dir,
                    keyframe_paths=keyframe_paths,
                    model=model,
                    processor=processor
                )

            # Display results
            st.subheader("Analysis Results")
            st.write(result['response'])
            st.write(f"Time taken: {result['time_taken']:.2f} seconds")

if __name__ == "__main__":
    main() 