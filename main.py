import os
from src import corrosion_annotation

def main():
    # Define input and output directories
    input_dir = "input_images"
    output_base = "output_images"

    # Create output subdirectories for raw images, masks, and overlays.
    raw_dir = os.path.join(output_base, "raw")
    mask_dir = os.path.join(output_base, "segmentation_mask")
    overlay_dir = os.path.join(output_base, "overlay")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Define the annotation file path and clear its contents if it exists.
    annotation_file = os.path.join(output_base, "annotations.txt")
    with open(annotation_file, "w") as f:
        f.write("")  # Clear any previous annotations.

    # Process each image in the input directory.
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_dir, file)
            corrosion_annotation.process_image(
                image_path,
                raw_dir,
                mask_dir,
                overlay_dir,
                annotation_file
            )

if __name__ == "__main__":
    main()
