import cv2
import numpy as np
import os

# Attempt to load the pre-trained deep learning segmentation model.
try:
    from tensorflow.keras.models import load_model
    dl_model = load_model("model.h5")
    print("Deep Learning model loaded successfully.")
except Exception as e:
    dl_model = None
    print("Warning: Deep Learning model not loaded. Using traditional methods only.")

def preprocess_image(image):
    """
    Applies advanced preprocessing:
      - Bilateral filtering for edge-preserving smoothing.
      - Contrast enhancement using CLAHE on the L channel of LAB.
    """
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def threshold_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([5, 50, 50])
    upper_hsv = np.array([20, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask_hsv

def threshold_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    _, mask_lab = cv2.threshold(a_channel, 150, 255, cv2.THRESH_BINARY)
    return mask_lab

def threshold_ycrcb(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cr_channel = ycrcb[:, :, 1]
    _, mask_ycrcb = cv2.threshold(cr_channel, 140, 255, cv2.THRESH_BINARY)
    return mask_ycrcb

def remove_small_blobs(mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
    return filtered_mask

def detect_corrosion_traditional(image):
    """
    Detects corrosion using traditional methods:
      - Advanced preprocessing.
      - Thresholding in HSV, LAB, and YCrCb.
      - Morphological operations and small blob removal.
    """
    enhanced = preprocess_image(image)
    mask_hsv = threshold_hsv(enhanced)
    mask_lab = threshold_lab(enhanced)
    mask_ycrcb = threshold_ycrcb(enhanced)

    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    refined_mask = remove_small_blobs(refined_mask, min_area=500)
    return refined_mask

def predict_mask_deep(image):
    """
    Uses the deep learning model to predict a corrosion mask.
    Assumes the model expects 256x256 input images.
    """
    if dl_model is None:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    input_size = (256, 256)
    image_resized = cv2.resize(image, input_size)
    image_input = image_resized.astype('float32') / 255.0
    image_input = np.expand_dims(image_input, axis=0)  # Shape: (1, 256, 256, 3)

    pred_mask = dl_model.predict(image_input)[0]  # Shape: (256, 256, 1)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_deep = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
    return mask_deep

def ensemble_mask(image):
    """
    Combines the traditional and deep learning masks using a bitwise OR.
    A small additional morphological opening is applied for cleanup.
    """
    mask_traditional = detect_corrosion_traditional(image)
    mask_dl = predict_mask_deep(image)
    combined_mask = cv2.bitwise_or(mask_traditional, mask_dl)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return final_mask

def classify_corrosion(mask):
    """
    Classifies corrosion severity based on the ratio of the detected area to the total area.
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
    Overlays the corrosion mask on the original image in red.
    """
    red_layer = np.zeros_like(image)
    red_layer[:] = (0, 0, 255)
    alpha = 0.5
    blended = cv2.addWeighted(image, 1, red_layer, alpha, 0)

    overlay = image.copy()
    overlay[mask > 0] = blended[mask > 0]
    return overlay

def process_image(image_path, raw_dir, mask_dir, overlay_dir, annotation_file):
    """
    Processes an image:
      - Reads the image.
      - Computes the ensemble corrosion mask.
      - Classifies the corrosion severity.
      - Creates an overlay.
      - Saves raw, mask, and overlay images.
      - Appends an annotation to the annotation file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return

    mask = ensemble_mask(image)
    level, ratio = classify_corrosion(mask)
    segmentation = overlay_mask(image, mask)

    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)

    raw_output = os.path.join(raw_dir, f"{name}_raw{ext}")
    mask_output = os.path.join(mask_dir, f"{name}_mask{ext}")
    overlay_output = os.path.join(overlay_dir, f"{name}_overlay{ext}")

    cv2.imwrite(raw_output, image)
    cv2.imwrite(mask_output, mask)
    cv2.imwrite(overlay_output, segmentation)

    annotation = f"Corrosion Level: {level} (Area ratio: {ratio:.2f})"
    with open(annotation_file, "a") as f:
        f.write(f"{name}: {annotation}\n")

    print(f"Processed {image_path}: {annotation}")
