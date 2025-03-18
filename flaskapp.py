from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import torch

# For keyframe extraction and inference:
#   - You can move or rename your custom analyze_video function if you prefer.
#   - Alternatively, import the logic directly from gemma_infer_video, 
#     if you placed it there. For demonstration, we're defining it inline.
from gemma_infer_video import get_uniform_keyframes, initialize_model

from PIL import Image

app = Flask(__name__)

# Configure where uploads and keyframes are saved
UPLOAD_FOLDER = 'uploads'
KEYFRAME_FOLDER = 'keyframes_output'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KEYFRAME_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is valid."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video(video_path, text_prompt, output_dir, keyframe_paths=None, model=None, processor=None, max_new_tokens=512):
    """
    Adapted from your Streamlit version of analyze_video (gemma_infer_video_streamlit.py).
    This function:
      1. Extracts keyframes if they aren't already provided.
      2. Runs the Gemma model on these keyframes + text prompt.
      3. Returns the generated response and inference time.
    """

    # If keyframe_paths were not precomputed, compute them.
    if keyframe_paths is None:
        keyframe_paths = get_uniform_keyframes(video_path, output_dir, num_keyframes=30)

    # Load model and processor if not provided
    if model is None or processor is None:
        model, processor = initialize_model()

    # Construct the “messages” structure used by Gemma
    # In your Streamlit code, you had a system + user role. We replicate that.
    # The user content includes images + the text prompt.
    # For demonstration, we do a simpler approach here: each frame is an image,
    # plus the text prompt as text. (If your custom parse is different, adapt accordingly.)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": (
                [
                    {"type": "image", "image": Image.open(frame_path)}
                    for frame_path in keyframe_paths
                ]
                + [{"type": "text", "text": text_prompt}]
            ),
        }
    ]

    # Tokenize via Gemma’s processor
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # We only measure the generation (inference) time here
    start_time = time.time()

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        # Slice away the original prompt portion
        generation = generation[0][input_len:]

    end_time = time.time()
    time_taken = end_time - start_time

    response = processor.decode(generation, skip_special_tokens=True)

    # Clean up GPU memory
    torch.cuda.empty_cache()

    return {
        'frames': keyframe_paths,
        'response': response,
        'time_taken': time_taken
    }

@app.route('/')
def index():
    """
    Show the main upload form.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle the form submission: upload video, read prompt, run analysis,
    and return JSON response.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in request'}), 400

    video = request.files['video']
    prompt = request.form.get('prompt', '')

    # Fallback if user leaves prompt blank
    if not prompt.strip():
        prompt = "How many cars are there in the scene? Return the number as a single integer."

    if video.filename == '':
        return jsonify({'error': 'Empty filename provided'}), 400

    if video and allowed_file(video.filename):
        # Secure the file name and save
        filename = secure_filename(video.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(save_path)

        # Use a function from above to analyze
        result = analyze_video(
            video_path=save_path,
            text_prompt=prompt,
            output_dir=KEYFRAME_FOLDER
        )

        return jsonify({
            'response': result['response'],
            'time_taken': f"{result['time_taken']:.2f} seconds",
            'num_frames': len(result['frames'])
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
