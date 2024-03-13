import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Helper function to convert image to black and white
def convert_BW(img, threshold=50):
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img


# Function to calculate SSIM
def calculate_ssim(image1, image2):
    similarity = ssim(image1, image2)
    return similarity


def calculate_iou(image1, image2):
    # Calculating intersection and union
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)

    # Computing the IoU
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score




# Main processing function
def process_image_enhanced(image_path, ground_truth_path):
    # Read the image and ground truth
    image = cv2.imread(image_path)
    ground_truth_original = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = convert_BW(ground_truth_original)

    # Apply Median Blur to reduce noise while keeping edges sharp
    median_blurred = cv2.medianBlur(image, 15)

    # Convert to grayscale
    gray_image = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, binary_thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_binary_image = cv2.bitwise_not(binary_thresholded_image)

    # Morphological operations to clean up the segmentation
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Calculate SSIM
    similarity = calculate_ssim(opening, ground_truth)
    iou = calculate_iou(opening, ground_truth)

    # Prepare result dict to return
    result = {
        "original": image,
        "gray": gray_image,
        "median_blurred": median_blurred,
        "binary_thresholded": inverted_binary_image,
        "morph_opening": opening,
        "ground_truth": ground_truth,
        "ssim": similarity,
        "iou": iou
    }

    return result


folders = ['easy', 'medium', 'hard']
# Set the base directory for your dataset
base_dir = 'Dataset/Dataset'

# File paths to your images and ground truths
for folder in folders:
    image_paths = [f"{base_dir}/input_images/{folder}/{folder}_{i}.jpg" for i in range(1, 4)]
    ground_truth_paths = [f"{base_dir}/ground_truths/{folder}/{folder}_{i}.png" for i in range(1, 4)]

    ssim_values = []
    iou_values = []

    for i in range(len(image_paths)):
        results = process_image_enhanced(image_paths[i], ground_truth_paths[i])
        ssim_values.append(results['ssim'])
        iou_values.append(results['iou'])

        titles = ['Original Image', 'Grayscale Image', 'Median Blurred Image',
                  'Inverted Binary Image', 'Morphological Opening Image', 'Simple Threshold Ground Truth']

        plt.figure(figsize=(12, 8))  # Adjust the size as needed
        for j, (key, title) in enumerate(zip(['original', 'gray', 'median_blurred', 'binary_thresholded',
                                              'morph_opening', 'ground_truth'], titles)):
            plt.subplot(2, 3, j + 1)
            img = results[key]
            if img.ndim == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')

            plt.title(f"{title}\nSSIM: {results['ssim']:.4f}" if key == 'morph_opening' else title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"SSIM values for {folder} folder:", ssim_values)
    print(f"IOU values for {folder} folder:", iou_values)
