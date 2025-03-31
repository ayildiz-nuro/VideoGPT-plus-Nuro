import pandas as pd
import os
import time
import subprocess
import sys

# Define base directory
BASE_DIR = "localhost_inference"

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

# Define file paths
QUEUE_FILE = os.path.join(BASE_DIR, "queue.csv")
COMPLETED_FILE = os.path.join(BASE_DIR, "completed.csv")

# Initialize queue file if it doesn't exist
if not os.path.exists(QUEUE_FILE):
    pd.DataFrame(columns=["id", "timestamp", "video_path", "status"]).to_csv(QUEUE_FILE, index=False)
    print(f"Created new queue file: {QUEUE_FILE}")

# Initialize completed file if it doesn't exist
if not os.path.exists(COMPLETED_FILE):
    pd.DataFrame(columns=["id", "timestamp", "video_path", "status", "result_path", "completed_at"]).to_csv(COMPLETED_FILE, index=False)
    print(f"Created new completed jobs file: {COMPLETED_FILE}")


def move_to_completed(request_id, status, result_path=""):
    """Move a request from the queue to the completed jobs file."""
    # Read both files
    queue = pd.read_csv(QUEUE_FILE)
    completed = pd.read_csv(COMPLETED_FILE)
    
    # Find the request in the queue
    request = queue[queue["id"] == request_id]
    
    if request.empty:
        print(f"Warning: Request {request_id} not found in queue")
        return False
    
    # Add completion timestamp
    request = request.copy()
    request["status"] = status
    request["result_path"] = result_path
    request["completed_at"] = pd.Timestamp.now().isoformat()
    
    # Add to completed jobs
    completed = pd.concat([completed, request], ignore_index=True)
    completed.to_csv(COMPLETED_FILE, index=False)
    
    # Remove from queue
    queue = queue[queue["id"] != request_id]
    queue.to_csv(QUEUE_FILE, index=False)
    
    print(f"Moved request {request_id} to completed jobs with status: {status}")
    return True

def process_next_request():
    """Process the next pending request in the queue."""
    # Read the queue
    queue = pd.read_csv(QUEUE_FILE)
    
    # Check if there are any pending requests
    pending_requests = queue[queue["status"] == "pending"].sort_values("timestamp")
    
    if pending_requests.empty:
        return False
    
    # Get the oldest pending request
    request = pending_requests.iloc[0]
    request_id = request["id"]
    video_path = request["video_path"]
    
    # Update status to processing
    queue.loc[queue["id"] == request_id, "status"] = "processing"
    queue.to_csv(QUEUE_FILE, index=False)
    print(f"Processing request {request_id} for video {video_path}")
    
    # Prepare the output path
    result_filename = f"result_{request_id}.txt"
    result_path = os.path.join(BASE_DIR, "results", result_filename)
    
    try:
        # Run gemma_infer_video.py as a separate process
        cmd = [
            sys.executable,  # Current Python interpreter
            "gemma_infer_video.py",
            "--video_path", video_path,
            "--output_path", result_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the process and wait for it to complete
        process = subprocess.run(
            cmd,
            check=True,  # Raise exception if process returns non-zero exit code
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print the output for debugging
        print("Process stdout:", process.stdout)
        if process.stderr:
            print("Process stderr:", process.stderr)
        
        # Check if the result file was created
        if os.path.exists(result_path):
            # Move the request to completed jobs
            move_to_completed(request_id, "success", result_path)
            print(f"Request {request_id} completed successfully.")
        else:
            raise Exception(f"Result file {result_path} was not created")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running inference process: {str(e)}")
        print(f"Process stdout: {e.stdout}")
        print(f"Process stderr: {e.stderr}")
        
        # Move to completed with error status
        move_to_completed(request_id, "error")
    
    except Exception as e:
        print(f"Error processing request {request_id}: {str(e)}")
        
        # Move to completed with error status
        move_to_completed(request_id, "error")
    
    return True

def main():
    print("Starting worker process...")
    
    while True:
        try:
            # Process the next request if available
            request_processed = process_next_request()
            
            if not request_processed:
                print("No pending requests. Waiting...")
                time.sleep(10)  # Wait for 10 seconds before checking again
            else:
                # Small delay between processing requests
                time.sleep(2)
                
        except Exception as e:
            print(f"Error in worker process: {str(e)}")
            time.sleep(30)  # Wait longer if there's an error

if __name__ == "__main__":
    main() 