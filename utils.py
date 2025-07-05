import os
import argparse

def validate_file_path(path, file_type="file"):
    """Validate that a file path exists"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_type.capitalize()} not found: {path}")
    return path

def get_video_output_path(input_path):
    """Generate output path for processed video"""
    base, ext = os.path.splitext(input_path)
    return f"{base}_tracking{ext}"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Person Tracking System")
    parser.add_argument("-i", "--input", required=True, 
                       help="Path to input video file")
    parser.add_argument("-r", "--reference", 
                       help="Path to reference image file")
    parser.add_argument("-o", "--output", 
                       help="Path to output video file (optional)")
    parser.add_argument("--display_scale", type=float, default=0.7,
                       help="Display scale factor (default: 0.7)")
    parser.add_argument("--processing_scale", type=float, default=0.5,
                       help="Processing scale factor (default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate input paths
    validate_file_path(args.input, "input video")
    if args.reference:
        validate_file_path(args.reference, "reference image")
    
    # Set output path if not provided
    if not args.output:
        args.output = get_video_output_path(args.input)
    
    return args