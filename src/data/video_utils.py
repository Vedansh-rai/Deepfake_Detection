import cv2
import numpy as np

def extract_frames(video_path, num_frames=20):
    """
    Extracts num_frames uniformly from the video.
    Returns a list of numpy arrays (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    
    if frame_count <= 0:
        # Fallback if frame count cannot be determined
        pass
    else:
        # Uniform sampling
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    # Placeholder if even first frame fails
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    
    # Padding if we couldn't extract enough frames
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
             frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
             
    # Truncate if we somehow got more (unlikely with this logic but good for safety)
    return frames[:num_frames]
