
# Corrosion Detection and Annotation Project

This project uses Python and OpenCV to automatically detect corrosion areas in images, classify the corrosion severity (e.g., HIGH, Severe, Medium, Low, None), and generate outputs including:

- **Raw images:** A copy of the original image.
- **Segmentation Masks:** Binary masks showing the corrosion regions.
- **Overlay Images:** The original image with corrosion areas highlighted.
- **Annotations:** A text file summarizing the corrosion classification for each image.

## Enhancements in version 1.3

- **Preprocessing:** Bilateral filtering and CLAHE-based contrast enhancement.
- **Multi-Color Space Thresholding:** Combines HSV, LAB, and YCrCb thresholds with morphological refinement.
- **Deep Learning Segmentation:** Uses a pre-trained model (`model.h5`) for segmentation.
- **Ensemble Fusion:** Combines traditional and deep learning masks.
- **Robust Classification:** Determines corrosion severity from the final mask.

## Directory Structure

```
corrosion_detection/
├── input_images/                # Place your raw images here.
├── output_images/               # Contains the processed images and annotations.
│   ├── raw/                    # Raw images.
│   ├── segmentation_mask/      # Binary segmentation masks.
│   ├── overlay/                # Images with corrosion highlighted.
│   └── annotations.txt         # Combined annotation file.
├── src/
│   ├── __init__.py             # Python package indicator.
│   └── corrosion_annotation.py # Image processing functions.
├── main.py                     # Main script.
├── requirements.txt            # Python dependencies.
└── README.md                   # This file.
```

## Setup and Usage

1. **Install Dependencies**

   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**

   Place your images in the `input_images/` directory.

3. **Run the Project**

   From the project root directory, execute:
   ```bash
   python main.py
   ```

   This will process each image, saving outputs into the respective folders under `output_images/` and appending annotations to `annotations.txt`.

Feel free to modify the code in `src/corrosion_annotation.py` to fine-tune the corrosion detection for your specific images.
