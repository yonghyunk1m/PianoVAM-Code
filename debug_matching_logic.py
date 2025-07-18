#!/usr/bin/env python3
"""
Debug the file matching logic used in ASDF.py
"""

import os
import glob

def get_available_predefined_files():
    """Get list of available predefined keypoint files (copied from ASDF.py)"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pixel_points_dir = os.path.join(script_dir, "FingeringDetection", "pixel_points")
    
    # Check if pixel_points directory exists
    if not os.path.exists(pixel_points_dir):
        return []
    
    # Find all JSON files matching the pattern
    pattern = os.path.join(pixel_points_dir, "*_pixel_points.json")
    json_files = glob.glob(pattern)
    return sorted(json_files, reverse=True)  # Most recent first

def test_matching_logic():
    """Test the file matching logic"""
    
    # Test with the actual video name
    selected_option = "2024-09-05_13-50-00.mp4"
    video_key = selected_option[:-4]  # Remove .mp4 extension
    
    print(f"🎬 Selected video: {selected_option}")
    print(f"🔑 Video key: {video_key}")
    
    # Get available predefined files
    predefined_files = get_available_predefined_files()
    
    print(f"\n📁 Found {len(predefined_files)} predefined files")
    print("🔍 First 5 files:")
    for i, file_path in enumerate(predefined_files[:5]):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    
    print(f"\n🔍 Looking for files containing: '{video_key.lower()}'")
    
    # Look for a matching predefined file (same logic as ASDF.py)
    matching_file = None
    for file_path in predefined_files:
        filename = os.path.basename(file_path)
        print(f"  Testing: {filename}")
        print(f"    Contains '{video_key.lower()}'? {video_key.lower() in filename.lower()}")
        
        if video_key.lower() in filename.lower():
            matching_file = file_path
            print(f"    ✅ MATCH FOUND!")
            break
    
    print(f"\n🎯 Result:")
    if matching_file:
        print(f"✅ Matching file: {os.path.basename(matching_file)}")
        print(f"📍 Full path: {matching_file}")
    else:
        print(f"❌ No matching file found for '{video_key}'")
        
        # Debug: Show what would be the expected filename
        expected_filename = f"{video_key}_pixel_points.json"
        print(f"💡 Expected filename: {expected_filename}")
        
        # Check if expected file exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        expected_path = os.path.join(script_dir, "FingeringDetection", "pixel_points", expected_filename)
        print(f"💡 Expected path exists: {os.path.exists(expected_path)}")

if __name__ == "__main__":
    test_matching_logic() 