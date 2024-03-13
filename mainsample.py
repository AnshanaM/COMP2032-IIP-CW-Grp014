import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Convert image to black and white based on a threshold
def convert_BW(img, threshold=50):
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img


# Calculate SSIM (Structural Similarity Index)
def calculate_ssim(image1, image2):
    similarity = ssim(image1, image2)
    return similarity


# Calculate IoU (Intersection over Union)
def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# Find the Region of Interest (ROI) in the image by locating the flower
def find_flower_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h), largest_contour


# Enhanced image processing function focused on the ROI
def process_image_enhanced_roi(image_path, ground_truth_path):
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Detect the ROI in the original image
    roi_coords, _ = find_flower_roi(image)
    x, y, w, h = roi_coords

    # Extract the ROI from both the image and the ground truth
    roi_image = image[y:y + h, x:x + w]
    roi_ground_truth = ground_truth[y:y + h, x:x + w]
    roi_ground_truth_binary = convert_BW(roi_ground_truth)

    # Processing steps applied to the ROI
    median_blurred = cv2.medianBlur(roi_image, 15)
    gray_image = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, binary_thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_binary_image = cv2.bitwise_not(binary_thresholded_image)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Calculate metrics for the processed ROI
    similarity = calculate_ssim(opening, roi_ground_truth_binary)
    iou = calculate_iou(opening, roi_ground_truth_binary)

    # Store results for display
    result = {
        "original": roi_image,
        "gray": gray_image,
        "median_blurred": median_blurred,
        "binary_thresholded": binary_thresholded_image,
        "inverted_binary": inverted_binary_image,
        "closing": closing,
        "opening": opening,
        "ground_truth": roi_ground_truth_binary,
        "ssim": similarity,
        "iou": iou
    }

    return result


# Modify this path to your specific setup
image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'  # Replace with your image path
ground_truth_path = 'Dataset/Dataset/ground_truths/easy/easy_1.png'

results = process_image_enhanced_roi(image_path, ground_truth_path)

# Titles for the graph
titles = ['Original ROI', 'Grayscale ROI', 'Median Blurred ROI',
          'Binary Threshold ROI', 'Inverted Binary ROI', 'Closing', 'Opening', 'Ground Truth ROI']

# Display each step
plt.figure(figsize=(18, 12))  # Adjust the size as needed
for j, (key, title) in enumerate(zip(['original', 'gray', 'median_blurred', 'binary_thresholded',
                                      'inverted_binary', 'closing', 'opening', 'ground_truth'], titles)):
    plt.subplot(3, 3, j + 1)
    img = results[key]
    if img.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
        plt.title(f"{title}\nSSIM: {results['ssim']:.4f}" if key == 'opening' else title)
        plt.axis('off')
plt.tight_layout()
plt.show()

# Output the SSIM and IoU values for the ROI
print(f"SSIM value for ROI: {results['ssim']:.4f}")
print(f"IOU value for ROI: {results['iou']:.4f}")

        # import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the images
# flower_image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'
# ground_truth_path = 'Dataset/Dataset/ground_truths/easy/easy_1.png'
#
# flower_image = cv2.imread(flower_image_path)
# ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
#
# # Convert the flower image to grayscale
# gray_flower_image = cv2.cvtColor(flower_image, cv2.COLOR_BGR2GRAY)
#
# # Apply contrast stretching to the grayscale image
# min_val, max_val = np.min(gray_flower_image), np.max(gray_flower_image)
# contrast_stretched_flower = ((gray_flower_image - min_val) / (max_val - min_val)) * 255
# contrast_stretched_flower = contrast_stretched_flower.astype(np.uint8)
#
# # Apply binary thresholding to separate the flower from the background
# _, binary_flower_image = cv2.threshold(contrast_stretched_flower, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# # Invert binary image if necessary so the flower is white and the background is black
# # This is done based on the assumption that the flower should be white (255) in the binary image
# # Check if the flower is darker than the background
# if np.mean(gray_flower_image[ground_truth > 0]) < np.mean(gray_flower_image[ground_truth == 0]):
#     binary_flower_image = cv2.bitwise_not(binary_flower_image)
#
# # Show the results
# plt.figure(figsize=(15, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(flower_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')
#
# plt.subplot(1, 3, 2)
# plt.imshow(contrast_stretched_flower, cmap='gray')
# plt.title('Contrast Stretched')
# plt.axis('off')
#
# plt.subplot(1, 3, 3)
# plt.imshow(binary_flower_image, cmap='gray')
# plt.title('Binary Image')
# plt.axis('off')
#
# plt.show()
#
# # Evaluate against the ground truth
# # Calculate the Intersection over Union (IoU) for evaluation
# intersection = np.logical_and(binary_flower_image > 0, ground_truth > 0)
# union = np.logical_or(binary_flower_image > 0, ground_truth > 0)
# iou_score = np.sum(intersection) / np.sum(union)
#
# print(f"The Intersection over Union (IoU) score is: {iou_score}")
#
