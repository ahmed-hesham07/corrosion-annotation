import cv2
import numpy as np
import os

def preprocess_image(image):
    """
    Applies advanced preprocessing:
      - Bilateral filtering for edge-preserving smoothing.
      - Contrast enhancement using CLAHE on the L channel of LAB.
    """
    # Edge-preserving smoothing
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to LAB and apply CLAHE to the L channel
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def threshold_hsv(image):
    """
    Applies HSV thresholding to detect rust-like colors.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([5, 50, 50])
    upper_hsv = np.array([20, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask_hsv

def threshold_lab(image):
    """
    Uses the a-channel in LAB color space to detect reddish tones.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    # Adjust the threshold value as needed (150 is an example)
    _, mask_lab = cv2.threshold(a_channel, 150, 255, cv2.THRESH_BINARY)
    return mask_lab

def threshold_ycrcb(image):
    """
    Uses the Cr channel in YCrCb color space to detect red components.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cr_channel = ycrcb[:, :, 1]
    # Adjust the threshold value as needed (140 is an example)
    _, mask_ycrcb = cv2.threshold(cr_channel, 140, 255, cv2.THRESH_BINARY)
    return mask_ycrcb

def remove_small_blobs(mask, min_area=500):
    """
    Removes small connected components from the mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
    return filtered_mask

def detect_corrosion(image):
    """
    Detects corrosion regions using:
      - Advanced preprocessing.
      - Multi-color space thresholding (HSV, LAB, YCrCb).
      - Mask fusion, morphological operations, and blob filtering.
    """
    # Preprocess the image for noise reduction and contrast enhancement.
    enhanced = preprocess_image(image)

    # Generate masks from different color spaces.
    mask_hsv = threshold_hsv(enhanced)
    mask_lab = threshold_lab(enhanced)
    mask_ycrcb = threshold_ycrcb(enhanced)

    # Combine the masks.
    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_ycrcb)

    # Refine the mask with morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small blobs to reduce false positives.
    refined_mask = remove_small_blobs(refined_mask, min_area=500)
    return refined_mask

def classify_corrosion(mask):
    """
    Classifies the corrosion severity based on the ratio of the detected corrosion area
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
    Creates an overlay image highlighting the corrosion areas in red.
    """
    red_layer = np.zeros_like(image)
    red_layer[:] = (0, 0, 255)  # Red color in BGR.
    alpha = 0.5  # Transparency factor.

    # Blend the entire image with the red layer.
    blended = cv2.addWeighted(image, 1, red_layer, alpha, 0)

    # Create an overlay that only applies the blended effect where the mask is non-zero.
    overlay = image.copy()
    overlay[mask > 0] = blended[mask > 0]
    return overlay

def process_image(image_path, raw_dir, mask_dir, overlay_dir, annotation_file):
    """
    Processes a single image:
      - Reads the image.
      - Detects corrosion regions using the improved method.
      - Classifies the corrosion severity.
      - Creates an overlay image.
      - Saves the raw image, mask, and overlay.
      - Appends the annotation to a text file.
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

    # Write the annotation.
    annotation = f"Corrosion Level: {level} (Area ratio: {ratio:.2f})"
    with open(annotation_file, "a") as f:
        f.write(f"{name}: {annotation}\n")

    print(f"Processed {image_path}: {annotation}")
