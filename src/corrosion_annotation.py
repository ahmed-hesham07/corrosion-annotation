import os
import numpy as np
import cv2

# Attempt to load the trained model
try:
    import tensorflow as tf
    import segmentation_models as sm
    dl_model = tf.keras.models.load_model("model.h5", compile=False)
    print("Deep Learning model loaded successfully.")
except Exception as e:
    dl_model = None
    print("Warning: No valid deep learning model found. Using only traditional methods.")

# Optional: CRF post-processing using pydensecrf
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("pydensecrf not available. Skipping CRF post-processing.")

def color_threshold_mask(image):
    """Create a mask using HSV thresholding to capture rust/corrosion colors."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 50, 50])
    upper = np.array([20, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_cleaned

def predict_dl_mask(image):
    """Predict a segmentation mask using the deep learning model."""
    if dl_model is None:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    input_size = (256, 256)
    resized = cv2.resize(image, input_size)
    inp = resized.astype("float32") / 255.0
    inp = np.expand_dims(inp, axis=0)
    pred = dl_model.predict(inp)[0]
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
    return mask

def apply_crf(image, softmax_prob):
    """
    Refine the segmentation using DenseCRF.
    softmax_prob should be a numpy array of shape (2, H, W) with probabilities.
    """
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(softmax_prob)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=image, compat=10)
    Q = d.inference(5)
    refined = np.array(Q).reshape((2, h, w))
    refined_mask = (refined[1] > refined[0]).astype(np.uint8) * 255
    return refined_mask

def refine_mask_with_crf(image, mask):
    """
    Optionally refine the predicted mask using CRF if available.
    Here, we approximate softmax probabilities using the mask.
    """
    if not CRF_AVAILABLE:
        return mask
    # Create a two-channel probability map
    prob = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
    prob[1] = mask.astype(np.float32) / 255.0
    prob[0] = 1 - prob[1]
    refined = apply_crf(image, prob)
    return refined

def ensemble_mask(image):
    """Combine deep learning mask and color threshold mask, then post-process."""
    mask_dl = predict_dl_mask(image)
    mask_color = color_threshold_mask(image)
    combined = cv2.bitwise_or(mask_dl, mask_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    post_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    if CRF_AVAILABLE:
        post_mask = refine_mask_with_crf(image, post_mask)
    return post_mask

def classify_corrosion(mask):
    """Classify the corrosion severity based on the ratio of the segmented area."""
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
    """Overlay the mask on the original image in red."""
    red_layer = np.zeros_like(image)
    red_layer[:] = (0, 0, 255)
    alpha = 0.5
    blended = cv2.addWeighted(image, 1, red_layer, alpha, 0)
    overlay = image.copy()
    overlay[mask > 0] = blended[mask > 0]
    return overlay

def process_image(image_path, raw_dir, mask_dir, overlay_dir, annotation_file):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        return

    mask = ensemble_mask(image)
    level, ratio = classify_corrosion(mask)
    overlay_img = overlay_mask(image, mask)

    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)

    raw_output = os.path.join(raw_dir, f"{name}_raw{ext}")
    mask_output = os.path.join(mask_dir, f"{name}_mask{ext}")
    overlay_output = os.path.join(overlay_dir, f"{name}_overlay{ext}")

    cv2.imwrite(raw_output, image)
    cv2.imwrite(mask_output, mask)
    cv2.imwrite(overlay_output, overlay_img)

    annotation_text = f"{name}: Corrosion Level: {level} (Area ratio: {ratio:.2f})\n"
    with open(annotation_file, "a") as f:
        f.write(annotation_text)

    print(f"Processed {image_path}: {annotation_text.strip()}")
