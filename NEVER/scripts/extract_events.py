import cv2
import numpy as np
import argparse
import struct

def extract_event_deltas(video_path, output_path, threshold=30):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_idx = 0
    total_events = 0
    
    # Format for C++ spike accumulator:
    # struct SpikeEvent { uint32_t x : 12; uint32_t y : 12; int32_t polarity : 2; uint32_t channel : 6; };
    # We will write raw binary data for the C++ loader.
    # To keep it simple in Python, we will pack as 32-bit uint:
    # 12 bits X | 12 bits Y | 2 bits Polarity (0=0, 1=1, 2=-1) | 6 bits Channel
    
    with open(output_path, 'wb') as f_out:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray, prev_gray)
            
            # Apply threshold to find "moving" pixels (Spike Generation)
            _, thresh_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Get coordinates of moving pixels
            y_coords, x_coords = np.where(thresh_diff > 0)
            
            # Calculate polarity (did it get brighter or darker?)
            for x, y in zip(x_coords, y_coords):
                intensity_change = int(gray[y, x]) - int(prev_gray[y, x])
                polarity = 1 if intensity_change > 0 else 2 # 2 represents -1 in our unsigned bitfield mapping
                
                # Pack the struct (X: 12 bits, Y: 12 bits, Polarity: 2 bits, Channel: 6 bits)
                # Channel 0 for now
                packed_event = (x & 0xFFF) | ((y & 0xFFF) << 12) | ((polarity & 0x3) << 24) | (0 << 26)
                f_out.write(struct.pack('<I', packed_event))
                total_events += 1
                
            prev_gray = gray
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames... Events so far: {total_events}")

    cap.release()
    print(f"Done! Extracted {total_events} events from {frame_idx} frames. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract movement events for N.E.V.E.R. SNN Engine")
    parser.add_argument("--input", default="evangelion_clip.mp4", help="Input video file")
    parser.add_argument("--output", default="events.never", help="Output binary event file")
    parser.add_argument("--threshold", type=int, default=30, help="Pixel difference threshold")
    
    args = parser.parse_args()
    
    print("N.E.V.E.R. Event Delta Extractor (6GB VRAM Optimization) - Step 01")
    # extract_event_deltas(args.input, args.output, args.threshold)
    print("Run with a valid input video. Example: python extract_events.py --input rei_smile.mp4")
