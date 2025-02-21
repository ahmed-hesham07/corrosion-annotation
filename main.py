import os
from src import corrosion_annotation

def main():
    input_dir = "input_images"
    output_dir = "output_images"

    raw_dir = os.path.join(output_dir, "raw")
    mask_dir = os.path.join(output_dir, "segmentation_mask")
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    annotation_file = os.path.join(output_dir, "annotations.txt")
    with open(annotation_file, "w") as f:
        f.write("")

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
