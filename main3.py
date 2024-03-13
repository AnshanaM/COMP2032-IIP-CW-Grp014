import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Helper functions
def convert_BW(img, threshold=50):
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

def calculate_ssim(image1, image2):
    similarity = ssim(image1, image2)
    return similarity

def contrast_stretching(img):
    minval, maxval = np.min(img), np.max(img)
    if minval == maxval:
        return img
    stretched_img = ((img - minval) / (maxval - minval)) * 255
    return stretched_img.astype(np.uint8)

# Main processing function
def process_image_enhanced(image_path, ground_truth_path):
    # Read the image and ground truth
    image = cv2.imread(image_path)
    ground_truth_original = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = convert_BW(ground_truth_original)

    # Apply Median Blur to reduce noise
    median_blurred = cv2.medianBlur(image, 5)

    # Convert to HSV and use the hue channel
    hsv_image = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]

    # Apply contrast stretching to the hue channel
    stretched_hue = contrast_stretching(hue_channel)

    # Apply Otsu's thresholding
    _, binary_thresholded_hue = cv2.threshold(stretched_hue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up the segmentation
    kernel = np.ones((5, 5), np.uint8)
    morph_opening = cv2.morphologyEx(binary_thresholded_hue, cv2.MORPH_OPEN, kernel)

    # Invert the binary image to match the ground truth if necessary
    if np.mean(ground_truth) > 127:
        morph_opening = cv2.bitwise_not(morph_opening)

    # Calculate SSIM
    similarity = calculate_ssim(morph_opening, ground_truth)

    # Prepare result dict to return
    result = {
        "original": image,
        "median_blurred": median_blurred,
        "hue_channel": hue_channel,
        "stretched_hue": stretched_hue,
        "binary_thresholded_hue": binary_thresholded_hue,
        "morph_opening": morph_opening,
        "ground_truth": ground_truth,
        "ssim": similarity
    }

    return result

base_dir = 'Dataset/Dataset'
# File paths to your images and ground truths
image_paths = [f"{base_dir}/input_images/hard/hard_{i}.jpg" for i in range(1, 4)]
ground_truth_paths = [f"{base_dir}/ground_truths/hard/hard_{i}.png" for i in range(1, 4)]


# Process each image and display
for i in range(len(image_paths)):
    # Process the image and get the results with the enhanced pipeline
    results = process_image_enhanced(image_paths[i], ground_truth_paths[i])

    # Define the list of titles for subplots
    titles = [
        'Original Image', 'Median Blurred Image', 'Hue Channel',
        'Stretched Hue', 'Binary Thresholded Hue', 'Morphological Opening Image',
        'Ground Truth'
    ]

    # Display results
    plt.figure(figsize=(18, 12))  # Adjust the size as needed
    for j, key in enumerate(results.keys()):
        plt.subplot(2, 4, j + 1)
        img = results[key]
        if img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')

        title = f"{titles[j]}\nSSIM: {results['ssim']:.4f}" if key == 'morph_opening' else titles[j]
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# To just print out SSIM values:
ssim_values = [process_image_enhanced(image_path, gt_path)['ssim'] for image_path, gt_path in zip(image_paths, ground_truth_paths)]
print(ssim_values)


# Initialize a large figure to accommodate all images
n = len(image_paths)
plt.figure(figsize=(20, n * 10))

# Process each image and display
for i in range(n):
    # Process the image and get the results
    results = process_image(image_paths[i], ground_truth_paths[i])

    # Define the list of titles for subplots
    titles = ['Original Image', 'Grayscale Image', 'Noise Reduced Image',
              'Contrast Stretched Image', 'Binary Thresholded Image', 'Ground Truth Image']

    # Display results
    for j, (key, title) in enumerate(zip(['original', 'gray', 'noise_reduced', 'contrast_stretched',
                                          'binary_thresholded', 'ground_truth'], titles)):
        plt.subplot(n, 6, i * 6 + j + 1)
        img = results[key]
        # Adjusting display method for color and grayscale images
        if img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # plt.show()  # Make sure this is called at the end of your script.

        else:
            plt.imshow(img, cmap='gray')

            # plt.show()  # Make sure this is called at the end of your script.

        # IoU and SSIM are only relevant to the binary thresholded image and ground truth
        if key in ['binary_thresholded']:
            plt.title(f"IoU: {results['iou_score']:.4f} | SSIM: {results['ssim']:.4f}")
plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.show()
# def convert_BW(img):  # converts the image to grayscale
#     img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img_bw
#
#
# def binary_threshold(img, threshold_num, max_num):
#     _, bi_image = cv2.threshold(img, threshold_num, max_num, cv2.THRESH_BINARY)
#     return bi_image
#
#
# def contrast_stretch(img, min_num, max_num):  # adjusts the image to the new minimum and maximum values
#     contrast_img = cv2.normalize(img, None, min_num, max_num, norm_type=cv2.NORM_MINMAX)
#     return contrast_img
#
#
# def image_pipeline(img):  # function to apply all processing functions to image
#     bw_img = convert_BW(img)
#     contrast_img = contrast_stretch(bw_img, 0, 255)
#     final_img = binary_threshold(contrast_img, 140, 255)
#     return final_img
#
#
# def saveImage(img, img_name, img_difficulty, img_no):
#
#
# # save image to the image pipeline with the img_name
#
#
# def display_final_images(
#         image_pipeline):  # using your imagepipeline() function, it will be applied on all images and displayed
#     file_names = ["easy", "medium", "hard"]
#     # getting all images file directories
#     # C:\\Users\\Manoharan\\Desktop\\IP CW\\Dataset\\input_images\\easy\\easy_1.jpg
#
#     default_file_directory = "C:\\Users\\Manoharan\\Desktop\\IP CW\\Dataset\\input_images\\"
#
#     for j in file_names:
#         first_file_directory = default_file_directory + j + "\\" + j + "_" + "1" + ".jpg"
#         sec_file_directory = default_file_directory + j + "\\" + j + "_" + "2" + ".jpg"
#         third_file_directory = default_file_directory + j + "\\" + j + "_" + "3" + ".jpg"
#
#         image1 = cv2.imread(first_file_directory, cv2.IMREAD_COLOR)
#         image2 = cv2.imread(sec_file_directory, cv2.IMREAD_COLOR)
#         image3 = cv2.imread(third_file_directory, cv2.IMREAD_COLOR)
#
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 6))
#         ax1.imshow(image_pipeline(image1), cmap="gray")
#         ax1.set_title(j + "_1")
#         ax2.imshow(image_pipeline(image2), cmap="gray")
#         ax2.set_title(j + "_2")
#         ax3.imshow(image_pipeline(image3), cmap="gray")
#         ax3.set_title(j + "_3")
#
#         plt.tight_layout()
#         plt.show()
#
#
# display_final_images(image_pipeline)