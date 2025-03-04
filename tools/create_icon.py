import sys
from pathlib import Path
from PIL import Image

def create_ico(input_image_path: Path, output_ico_path: Path):
    """Convert an image to a multi-size .ico file with alpha transparency."""
    img = Image.open(input_image_path).convert("RGBA")

    icon_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
        (48, 48),
        (32, 32),
        (16, 16)
    ]

    # Create assets directory if it doesn't exist
    output_ico_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .ico with multiple resolutions
    img.save(output_ico_path, format='ICO', sizes=icon_sizes)
    print(f"Successfully created icon: {output_ico_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_icon.py <input_image> <output_icon>")
        print("Example: python create_icon.py snake-logo.png src/assets/app.ico")
        return

    root_dir = Path("c:/devdrive/thInk")
    input_path = Path(sys.argv[1])
    output_path = root_dir / sys.argv[2]
    
    if not input_path.exists():
        print(f"Error: Input image not found at {input_path}")
        return
        
    create_ico(input_path, output_path)

if __name__ == "__main__":
    main()