from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter

class MyKeyFrameWriter(KeyFrameDiskWriter):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.timestamps = []  # store frame numbers or times

    def save_keyframe(self, frame, frame_number=None, fps=None):
        # Call parent method to actually save the frame
        super().save_keyframe(frame)
        import ipdb; ipdb.set_trace()
        
        # Save timestamp (in seconds) if fps is provided
        if frame_number is not None and fps is not None:
            time_in_seconds = frame_number / fps
            self.timestamps.append((frame_number, time_in_seconds))

# Usage
no_of_frames_to_returned = 5
video_file_path = "/home/ayildiz/sample_videos/video_nexar_perf_trim.mov"
save_path = "./keyframes/"

video = Video()

# Custom writer to also store timestamps
custom_writer = MyKeyFrameWriter(save_path=save_path)

# Extract keyframes (internally this will call our custom writer)
video.extract_video_keyframes(
    no_of_frames=no_of_frames_to_returned,
    file_path=video_file_path,
    writer=custom_writer
)

import ipdb; ipdb.set_trace()

# Now access timestamps
print("Extracted keyframe timestamps (frame number, seconds):")
for frame_no, time_sec in custom_writer.timestamps:
    print(f"Frame: {frame_no}, Time: {time_sec:.2f}s")
