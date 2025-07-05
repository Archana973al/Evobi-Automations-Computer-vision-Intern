from tracker import PersonTracker
from utils import parse_arguments
import os

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("Starting person tracker...")
    print(f"Input video: {args.input}")
    if args.reference:
        print(f"Reference image: {args.reference}")
    print(f"Output will be saved to: {args.output}")
    
    # Initialize tracker
    tracker = PersonTracker(
        reference_img_path=args.reference,
        display_scale=args.display_scale,
        processing_scale=args.processing_scale
    )
    
    # Process video
    tracker.process_video(args.input, args.output)
    
    print("Tracking completed.")

if __name__ == "__main__":
    main()