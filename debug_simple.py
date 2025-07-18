#!/usr/bin/env python3
"""
Simple debug to check file access
"""

import os
import json

def simple_debug():
    """Simple debug to check file access"""
    
    pixel_points_file = "./FingeringDetection/pixel_points/2024-03-27_11-49-04_pixel_points.json"
    
    print(f"🔍 Checking file: {pixel_points_file}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   File exists: {os.path.exists(pixel_points_file)}")
    
    if os.path.exists(pixel_points_file):
        try:
            with open(pixel_points_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Successfully loaded JSON")
            print(f"   Video name: {data.get('video_name', 'Unknown')}")
            print(f"   Number of points: {len(data.get('pixel_points', []))}")
            print(f"   First point: {data['pixel_points'][0] if data.get('pixel_points') else 'None'}")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
    else:
        print("❌ File not found")
        # List directory contents to debug
        dir_path = "./FingeringDetection/pixel_points/"
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"📁 Directory contents: {files[:5]}")  # Show first 5
        else:
            print(f"❌ Directory not found: {dir_path}")

if __name__ == "__main__":
    simple_debug() 