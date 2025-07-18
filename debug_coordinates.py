#!/usr/bin/env python3
"""
Debug coordinate mapping issues
"""

import json
import cv2
import numpy as np
import os

def debug_coordinate_mapping():
    """Debug the coordinate mapping between keystone points and actual video"""
    
    # Load predefined keystone points
    pixel_points_file = "./FingeringDetection/pixel_points/2024-09-05_13-50-00_pixel_points.json"
    
    if not os.path.exists(pixel_points_file):
        print(f"❌ File not found: {pixel_points_file}")
        return
    
    with open(pixel_points_file, 'r') as f:
        data = json.load(f)
    
    pixel_points = data['pixel_points']
    video_name = data.get('video_name', 'Unknown')
    
    print(f"📁 Loaded keystone data for: {video_name}")
    print(f"📍 Number of pixel points: {len(pixel_points)}")
    
    # Show first few pixel coordinates
    print(f"\n🔍 First 5 pixel coordinates:")
    for i, (x, y) in enumerate(pixel_points[:5]):
        print(f"  Point {i}: ({x}, {y})")
    
    # Load the video frames
    video_file = f"./FingeringDetection/videocapture/{video_name}.mp4"
    
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        return
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read video frame")
        return
    
    actual_height, actual_width = frame.shape[:2]
    print(f"\n📐 Actual video dimensions: {actual_width}x{actual_height}")
    
    # Check if dimensions match assumption
    assumed_width, assumed_height = 1920, 1080
    print(f"📐 Assumed dimensions: {assumed_width}x{assumed_height}")
    
    if actual_width != assumed_width or actual_height != assumed_height:
        print("⚠️  DIMENSION MISMATCH DETECTED!")
        print(f"   Scale factor X: {actual_width / assumed_width:.4f}")
        print(f"   Scale factor Y: {actual_height / assumed_height:.4f}")
    else:
        print("✅ Dimensions match assumptions")
    
    # Convert pixel coordinates to normalized (using both actual and assumed dimensions)
    print(f"\n🔄 Coordinate conversion comparison:")
    print(f"{'Point':<8} {'Pixel':<12} {'Norm(assumed)':<15} {'Norm(actual)':<15}")
    print("-" * 60)
    
    for i, (x, y) in enumerate(pixel_points[:5]):
        # Using assumed dimensions (current code)
        norm_x_assumed = x / assumed_width
        norm_y_assumed = y / assumed_height
        
        # Using actual dimensions (corrected)
        norm_x_actual = x / actual_width  
        norm_y_actual = y / actual_height
        
        print(f"{i:<8} ({x:4},{y:3})    ({norm_x_assumed:.3f},{norm_y_assumed:.3f})    ({norm_x_actual:.3f},{norm_y_actual:.3f})")
    
    # Draw keystone points on frame for visual inspection
    debug_frame = frame.copy()
    
    # Draw pixel coordinates as red circles
    for i, (x, y) in enumerate(pixel_points):
        cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)  # Red circles
        if i < 10:  # Label first 10 points
            cv2.putText(debug_frame, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save debug frame
    debug_output = "debug_keystone_overlay.jpg"
    cv2.imwrite(debug_output, debug_frame)
    print(f"\n💾 Debug frame saved to: {debug_output}")
    print(f"   Red circles show the exact pixel coordinates from the keystone file")
    
    # Check if keystone points look reasonable
    x_coords = [p[0] for p in pixel_points]
    y_coords = [p[1] for p in pixel_points]
    
    print(f"\n📊 Keystone point statistics:")
    print(f"   X range: {min(x_coords)} to {max(x_coords)} (span: {max(x_coords) - min(x_coords)})")
    print(f"   Y range: {min(y_coords)} to {max(y_coords)} (span: {max(y_coords) - min(y_coords)})")
    print(f"   X coverage: {(max(x_coords) - min(x_coords)) / actual_width * 100:.1f}% of frame width")
    print(f"   Y coverage: {(max(y_coords) - min(y_coords)) / actual_height * 100:.1f}% of frame height")

if __name__ == "__main__":
    debug_coordinate_mapping() 