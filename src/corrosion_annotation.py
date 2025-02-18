import cv2
import numpy as np
import os

def detect_corrosion(image):
    """
    Detects corrosion regions using color thresholding in HSV space.
    Adjust the HSV range based on the corrosion characteristics.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define HSV range for rust/corrosion (tune these values for your data)
    lower_bound = np.array([5, 50, 50])
    upper_bound = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Reduce noise in the mask using morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def classify_corrosion(mask):
    """
    Classifies corrosion severity based on the ratio of the corrosion area (non-zero pixels)
    to the total image area.
    """
    corrosion_area = cv2.countNonZero(mask)
    total_area = mask.shape[0] * mask.shape[1]
    ratio = corrosion_area / total_area

    if ratio > 0.5:
        level = "HIGH"
    elif ratio > 0.3:
        level = "Severe"
    elif ratio > 0.1:
        level = "Medium"
    elif ratio > 0.02:
        level = "Low"
    else:
        level = "None"

    return level, ratio

def overlay_mask(image, mask):
    """
    Creates an overlay image that highlights corrosion regions in red.
    """
    overlay = image.copy()
    red_layer = np.zeros_like(image)
    red_layer[:] = (0, 0, 255)  # Red color in BGR format.
    alpha = 0.5  # Transparency factor.
    overlay = cv2.addWeighted(overlay, 1, red_layer, alpha, 0, mask=mask)
    return overlay

def process_image(image_path, raw_dir, mask_dir, overlay_dir, annotation_file):
    """
    Processes a single image:
      - Reads the image.
      - Detects corrosion areas and generates a binary mask.
      - Classifies the corrosion severity.
      - Creates an overlay image with highlighted corrosion.
      - Saves the raw image, mask, and overlay to their respective directories.
      - Appends the classification annotation to the annotation file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return

    # Detect corrosion regions.
    mask = detect_corrosion(image)
    # Classify the corrosion severity.
    level, ratio = classify_corrosion(mask)
    # Create an overlay image.
    segmentation = overlay_mask(image, mask)

    # Prepare file names.
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)

    raw_output = os.path.join(raw_dir, f"{name}_raw{ext}")
    mask_output = os.path.join(mask_dir, f"{name}_mask{ext}")
    overlay_output = os.path.join(overlay_dir, f"{name}_overlay{ext}")

    # Save the images.
    cv2.imwrite(raw_output, image)
    cv2.imwrite(mask_output, mask)
    cv2.imwrite(overlay_output, segmentation)

    # Prepare the annotation text.
    annotation = f"Corrosion Level: {level} (Area ratio: {ratio:.2f})"

    # Append annotation to the annotations file.
    with open(annotation_file, "a") as f:
        f.write(f"{name}: {annotation}\n")

    print(f"Processed {image_path}: {annotation}")
