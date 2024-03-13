import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Function to apply contrast stretching
def contrast_stretching(img):
    minval, maxval = np.min(img), np.max(img)
    stretched_img = ((img - minval) / (maxval - minval)) * 255
    return stretched_img


def calculate_ssim(image1, image2):
    # image3 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    similarity = ssim(image1, image2)
    return similarity

# Function to apply noise reduction
def noise_reduction(img):
    blurred_img = cv2.GaussianBlur(img, (9, 9), 100)
    return blurred_img


# Function to apply thresholding
def thresholding(img):
    _, binary_img = cv2.threshold(img, 33, 225, cv2.THRESH_BINARY)
    # binary_img[binary_img > 255] =
    return binary_img


# Function to segment the flower using color space conversion and thresholding
def segment_flower(image):
    # Convert to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Use noise reduction
    hsv_image = noise_reduction(hsv_image)
    # Use only the hue value for segmentation
    hue_channel = hsv_image[:, :, 0]
    # Apply thresholding to the hue channel
    segmented_flower = thresholding(hue_channel)
    return segmented_flower.astype(np.uint8)


# Main processing function
def process_image(image_path, ground_truth_path):
    # Read the image and ground truth
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    noise_reduced_image = noise_reduction(gray_image)

    # Contrast stretching
    contrast_stretched_image = contrast_stretching(noise_reduced_image)

    # Convert to the HSV color space and extract the hue channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]

    # Apply thresholding to create a binary image
    binary_thresholded_image = thresholding(hue_channel)

    # inverted_ground_truth = 255 - ground_truth

    # Display the images
    plt.figure(figsize=(25, 20))

    plt.subplot(331)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(332)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(333)
    plt.imshow(noise_reduced_image, cmap='gray')
    plt.title('Noise Reduced Image')
    plt.axis('off')

    plt.subplot(334)
    plt.imshow(contrast_stretched_image, cmap='gray')
    plt.title('Contrast Stretched Image')
    plt.axis('off')

    plt.subplot(335)
    plt.imshow(hue_channel, cmap='gray')
    plt.title('Hue Channel Image')
    plt.axis('off')

    plt.subplot(336)
    plt.imshow(binary_thresholded_image, cmap='gray')
    plt.title('Binary Thresholded Image')
    plt.axis('off')

    plt.subplot(337)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Binary Thresholded Image')
    plt.axis('off')
    plt.show()

    # Save the binary image
    output_path = image_path.replace('.jpg', '_binary.jpg')
    cv2.imwrite(output_path, binary_thresholded_image)

    # Evaluate and print the IoU score for evaluation
    iou_score = np.sum((binary_thresholded_image > 0) & (ground_truth > 0)) / np.sum(
        (binary_thresholded_image > 0) | (ground_truth > 0))
    print(f"The Intersection over Union (IoU) score is: {iou_score}")

    similarity = calculate_ssim(binary_thresholded_image, ground_truth)
    print("SSIM:", similarity)
    return output_path


# Replace with the actual path of your images
image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'
ground_truth_path = 'Dataset/Dataset/ground_truths/easy/easy_1.png'
process_image(image_path, ground_truth_path)



