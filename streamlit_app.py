import streamlit as st
import os
import tempfile
import pandas as pd
import time
import uuid
from datetime import datetime
import cv2
from PIL import Image

# Define base directory
BASE_DIR = "localhost_inference"

# Create base directory and subdirectories if they don't exist
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

# Define file paths
QUEUE_FILE = os.path.join(BASE_DIR, "queue.csv")
COMPLETED_FILE = os.path.join(BASE_DIR, "completed.csv")

# Initialize queue file if it doesn't exist
if not os.path.exists(QUEUE_FILE):
    pd.DataFrame(columns=["id", "timestamp", "video_path", "status"]).to_csv(QUEUE_FILE, index=False)

# Initialize completed file if it doesn't exist
if not os.path.exists(COMPLETED_FILE):
    pd.DataFrame(columns=["id", "timestamp", "video_path", "status", "result_path", "completed_at"]).to_csv(COMPLETED_FILE, index=False)

def get_queue_position(request_id):
    """Get the position of the request in the queue."""
    queue = pd.read_csv(QUEUE_FILE)
    pending_requests = queue[queue["status"] == "pending"].sort_values("timestamp")
    if request_id in pending_requests["id"].values:
        return pending_requests["id"].tolist().index(request_id) + 1
    return None

def add_to_queue(video_path):
    """Add a new request to the queue."""
    queue = pd.read_csv(QUEUE_FILE)
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    new_request = pd.DataFrame({
        "id": [request_id],
        "timestamp": [timestamp],
        "video_path": [video_path],
        "status": ["pending"]
    })
    
    queue = pd.concat([queue, new_request], ignore_index=True)
    queue.to_csv(QUEUE_FILE, index=False)
    return request_id

def get_request_status(request_id):
    """Get the status of a request from either queue or completed files."""
    # First check the queue
    queue = pd.read_csv(QUEUE_FILE)
    if request_id in queue["id"].values:
        request = queue[queue["id"] == request_id].iloc[0]
        return request["status"], ""  # No result path for queue items
    
    # If not in queue, check completed jobs
    completed = pd.read_csv(COMPLETED_FILE)
    if request_id in completed["id"].values:
        request = completed[completed["id"] == request_id].iloc[0]
        return request["status"], request["result_path"]
    
    return None, None

def display_keyframes(video_path, num_keyframes=5):
    """Extract and display uniform keyframes from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        st.error("Could not read video frames.")
        return []
    
    # Calculate frame indices to extract
    indices = [int(i * total_frames / num_keyframes) for i in range(num_keyframes)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    # Display frames in columns
    if frames:
        cols = st.columns(min(5, len(frames)))
        for idx, (frame, col) in enumerate(zip(frames, cols * (len(frames) // len(cols) + 1))):
            if idx < len(frames):
                col.image(frame, caption=f"Frame {idx+1}")
    
    return frames

def main():
    st.title("Nuro Video Conflict Analysis")
    st.write("Upload a scene video to analyze its contents; the video will be analyzed for conflicts between the ego vehicle and other agents.")
    st.write("The current version of the model has trouble narrowing its focus directly on the conflict, so shorter videos (4-7 seconds) are highly recommended.")

    st.markdown("Known issues:\n- Agents/Events that are too fast are sometimes missed.\n- Agents not moving or parked are sometimes detected as a part of the conflict.")


    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    
    # Hard-code the number of keyframes
    num_keyframes = 30
    
    # Check if user has an active request
    if "request_id" in st.session_state:
        request_id = st.session_state["request_id"]
        status, result_path = get_request_status(request_id)
        
        # Display the request ID to the user
        st.info(f"Request ID: {request_id}")
        
        if status == "pending":
            queue_position = get_queue_position(request_id)
            st.info(f"Your request is in queue position {queue_position}.\nPlease check back in ~{1.5*(queue_position)} minutes and click the Refresh Status button below.")
            
            # Add a refresh button
            if st.button("Refresh Status"):
                st.rerun()
                
        elif status == "processing":
            queue_position = get_queue_position(request_id)
            st.info(f"Your video is currently being analyzed. You're next in queue, please check back in a minute.")
            
            # Add a refresh button
            if st.button("Refresh Status"):
                st.rerun()
                
        elif status == "success":
            st.success("Analysis completed!")

            st.write(" ")
            st.subheader("Analysis Metrics")
            st.write("1. Static elements of the scene (e.g. road markings, traffic lights, etc.)")
            st.write("2. Agents present in the scene, and their directions of travel.")
            st.write("3. Stage of the scenario progression.")

            
            # Display the result
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result = f.read()
                st.subheader("Analysis Results")
                st.markdown(result)
                
                # Get the video path from completed jobs
                completed = pd.read_csv(COMPLETED_FILE)
                video_path = completed[completed["id"] == request_id].iloc[0]["video_path"]
                
                # Option to start a new analysis
                if st.button("Start New Analysis"):
                    del st.session_state["request_id"]
                    st.rerun()
            else:
                st.error("Result file not found. Please contact support.")
                
        elif status == "error":
            st.error("An error occurred during analysis. Please try again or contact support.")
            
            # Option to start a new analysis
            if st.button("Start New Analysis"):
                del st.session_state["request_id"]
                st.rerun()
        
        elif status is None:
            st.error("Request not found. It may have been removed from the system.")
            
            # Option to start a new analysis
            if st.button("Start New Analysis"):
                del st.session_state["request_id"]
                st.rerun()
    
    # Show analyze button if not analyzing and file is uploaded
    elif uploaded_file:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_video_path = tmp_file.name
        
        # Save the file to a permanent location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{timestamp}_{uploaded_file.name}"
        video_path = os.path.join(BASE_DIR, "uploads", video_filename)
        
        # Copy from temp to permanent location
        with open(temp_video_path, 'rb') as src, open(video_path, 'wb') as dst:
            dst.write(src.read())
        
        # Remove the temporary file
        os.unlink(temp_video_path)
        
        if st.button("Add to Analysis Queue"):
            request_id = add_to_queue(video_path)
            st.session_state["request_id"] = request_id
            st.success("Video added to the analysis queue!")
            st.rerun()

if __name__ == "__main__":
    main() 