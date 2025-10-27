"""
Demo Script for Image Annotation Pipeline
This script demonstrates how to use the annotation pipeline programmatically
"""

from utils.annotator import GPTAnnotator
from utils.visualizer import visualize_annotations
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import json


def demo_annotation():
    """
    Demonstrate the annotation pipeline with a sample workflow
    """
    print("=" * 60)
    print("  Image Annotation Pipeline - Demo")
    print("=" * 60)
    print()
    
    # Load environment variables
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        script_env = Path(__file__).resolve().parent / '.env'
        if script_env.exists():
            load_dotenv(dotenv_path=script_env, override=False)

    # Initialize annotator
    print("1. Initializing GPT Annotator...")
    try:
        annotator = GPTAnnotator()
        print("   ✓ Annotator ready")
    except ValueError as e:
        print(f"   ❌ Error: {e}")
        print()
        print("Please set your OpenAI API key in the .env file:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        print()
        return
    print()
    
    # Check for images
    images_dir = Path("data/images")
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    
    if not image_files:
        print("2. No images found in data/images/")
        print("   Please add some images to the data/images/ folder first.")
        print()
        print("   You can:")
        print("   - Copy images manually to data/images/")
        print("   - Upload via the web interface")
        print()
        return
    
    print(f"2. Found {len(image_files)} image(s) in data/images/")
    for img in image_files[:5]:  # Show first 5
        print(f"   - {img.name}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more")
    print()
    
    # Annotate first image
    test_image = image_files[0]
    print(f"3. Generating annotation for: {test_image.name}")
    annotation = annotator.annotate(str(test_image))
    print(f"   ✓ Generated {len(annotation['element'])} elements")
    print()
    
    # Save annotation
    annotation_path = Path("data/annotations") / f"{test_image.stem}.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    print(f"4. Saved annotation to: {annotation_path.name}")
    print()
    
    # Display annotation details
    print("5. Annotation Details:")
    print(f"   Image Size: {annotation['img_size']}")
    print(f"   Elements: {len(annotation['element'])}")
    print()
    for i, elem in enumerate(annotation['element'][:3], 1):  # Show first 3
        print(f"   Element #{i}:")
        print(f"     Instruction: {elem['instruction']}")
        print(f"     BBox: {elem['bbox']}")
        print(f"     Point: {elem['point']}")
        print()
    if len(annotation['element']) > 3:
        print(f"   ... and {len(annotation['element']) - 3} more elements")
        print()
    
    # Generate visualization
    print("6. Generating visualization...")
    vis_base64 = visualize_annotations(str(test_image), annotation)
    print(f"   ✓ Visualization generated ({len(vis_base64)} bytes)")
    print()
    
    print("=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("  1. Start the web server: python app.py")
    print("  2. Open browser: http://localhost:5000")
    print("  3. Upload more images and explore the interface")
    print()
    print("To integrate GPT-5 API:")
    print("  - Edit utils/annotator.py")
    print("  - Implement the _call_gpt5_api() method")
    print("  - Add your custom prompt")
    print()


if __name__ == "__main__":
    demo_annotation()

