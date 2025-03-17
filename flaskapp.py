from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from gemma_infer_video import analyze_video, get_uniform_keyframes, initialize_model

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload and keyframe directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('keyframes_output', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if video and allowed_file(video.filename):
        # Initialize model when needed
        model, processor = initialize_model()
        
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)
        
        # Use the same prompt as in your original script
        prompt = """
        You are seeing key image frames of a video footage from a dashcam from the ego vehicle. You need to explain the conflicts that occur in the scene.
        Start by breaking the scene down into 3 main parts: 
        1. Description of road geometry. Static elements, and the environment. E.g. description of the traffic intersection, road markings, traffic lights, etc.
        2. Which agents are present in the scene, and what are they doing to cause a conflict? Be sure to mention their directions of travel from the perspective of the ego vehicle, and clarify whether their trajectories are parallel, orthogonal, or intersecting. Explain how their movements affect the potential conflict with the ego vehicle."
        3. What is the stage of the scenario progression? Each of these stages should capture agents' progress, intent, and interactions with the ego vehicle. Be very specific about the intent of the agents in the scene, and how they are causing a conflict with the ego vehicle.

        Keep the description of each part limited to a single sentence.
        Only reply with the answer, don't include anything else.
        """
        
        # Extract keyframes
        keyframe_paths = get_uniform_keyframes(
            video_path=video_path,
            output_dir='keyframes_output',
            num_keyframes=30
        )
        
        # Analyze video
        result = analyze_video(
            video_path=video_path,
            text_prompt=prompt,
            output_dir='keyframes_output',
            keyframe_paths=keyframe_paths,
            model=model,
            processor=processor
        )
        
        return jsonify({
            'response': result['response'],
            'time_taken': f"{result['time_taken']:.2f}",
            'num_frames': len(result['frames'])
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)