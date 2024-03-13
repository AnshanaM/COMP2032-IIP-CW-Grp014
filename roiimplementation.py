import cv2
import numpy as np
from matplotlib import pyplot as plt

def create_flower_mask(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bilateral filter
    bilateral_filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)

    # Edge detection
    edges = cv2.Canny(bilateral_filtered, 50, 150)

    # Dilate edges: increasing  dialation causes mergig of contours resulting in more detail
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask and draw largest contour
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [largest_contour], -20, 255, thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)

    # Apply mask to original image
    roi = cv2.bitwise_and(image, image, mask=mask)
    roi[mask == 0] = [255, 255, 255]

    return gray_image, bilateral_filtered, edges, dilated_edges, inverted_mask, roi

folders = ['easy', 'medium', 'hard']
base_dir = 'Dataset/Dataset'

# Loop through each folder and process images
for folder in folders:
    image_paths = [f"{base_dir}/input_images/{folder}/{folder}_{i}.jpg" for i in range(1, 4)]

    # Process each image
    for i, img_path in enumerate(image_paths):
        gray, bilateral, edges, dilated, inverted_mask, roi = create_flower_mask(img_path)

        plt.figure(figsize=(20, 10))

        steps = [gray, bilateral, edges, dilated, inverted_mask, roi]
        titles = ['Grayscale', 'Bilateral Filter', 'Edges', 'Dilated Edges', 'Inverted Mask', 'ROI']

        # Display each step
        for j in range(6):
            plt.subplot(2, 3, j+1)
            if j == 5:  # ROI in color
                plt.imshow(cv2.cvtColor(steps[j], cv2.COLOR_BGR2RGB))
            else:  # Other steps in grayscale
                plt.imshow(steps[j], cmap='gray')
            plt.title(titles[j])
            plt.axis('off')

        plt.suptitle(f'{folder.capitalize()} Image {i + 1}')
        plt.show()






# # Function to create a mask for the region of interest (ROI) which is the flower
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# def create_flower_mask(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Bilateral filter to reduce noise while keeping edges sharp
#     bilateral_filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)
#
#     # Canny edge detection
#     edges = cv2.Canny(bilateral_filtered, 50, 150)
#
#     # Dilate the edges to make them more pronounced and close gaps
#     dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
#
#     # Create an inverse mask where the flower is white and the background is black
#     # Find contours from the dilated edges
#     contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Assuming the largest contour is the flower
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # Create an empty mask
#     mask = np.zeros_like(gray_image)
#
#     # Fill the largest contour which is assumed to be the flower with white
#     cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#
#     # Invert the mask to get the flower in black and the background in white
#     inverted_mask = cv2.bitwise_not(mask)
#
#     # Use the mask to extract the ROI from the original image
#     roi = cv2.bitwise_and(image, image, mask=mask)
#
#     # Convert the background to white in the ROI image
#     roi[mask == 0] = [255, 255, 255]
#
#     return inverted_mask, roi
#
#
#
# # display
#
# folders = ['easy', 'medium', 'hard']
# base_dir = 'Dataset/Dataset'
#
# # Loop through each folder and process images
# for folder in folders:
#     image_paths = [f"{base_dir}/input_images/{folder}/{folder}_{i}.jpg" for i in range(1, 4)]
#
#     # Process each image to extract and display the ROI
#     for i, img_path in enumerate(image_paths):
#         inverted_mask, roi = create_flower_mask(img_path)
#
#         plt.figure(figsize=(10, 5))
#
#         plt.subplot(1, 2, 1)
#         plt.imshow(inverted_mask, cmap='gray')
#         plt.title('Inverted Mask')
#         plt.axis('off')
#
#         plt.subplot(1,2,2)
#         plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#         plt.title(f'ROI for {folder} Image {i + 1}')
#         plt.axis('off')
#         plt.show()

# Process each image and create the mask for the ROI (flower)
# for i in range(len(image_paths)):
#     inverted_mask, roi = create_flower_mask(image_paths[i])
#
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(inverted_mask, cmap='gray')
#     plt.title('Inverted Mask')
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#     plt.title('Region of Interest with White Background')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
#
# # Helper function to convert image to grayscale
# def convert_to_grayscale(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Helper function to convert image to black and white
# def convert_to_bw(img, threshold=127):
#     _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#     return binary_img
#
# # Function to calculate SSIM
# def calculate_ssim(image1, image2):
#     # Convert images to grayscale
#     image1_gray = convert_to_grayscale(image1)
#     image2_gray = convert_to_grayscale(image2)
#     # Compute SSIM between two images
#     return ssim(image1_gray, image2_gray)
#
# # Function to calculate IoU
# def calculate_iou(image1, image2):
#     # Convert images to binary (black and white)
#     image1_bw = convert_to_bw(image1)
#     image2_bw = convert_to_bw(image2)
#     # Calculate intersection and union
#     intersection = np.logical_and(image1_bw, image2_bw)
#     union = np.logical_or(image1_bw, image2_bw)
#     # Compute IoU score
#     return np.sum(intersection) / np.sum(union)
#
# # Function to find the ROI of the flower in the image
# def find_flower_roi(image):
#     # Convert to grayscale
#     gray_image = convert_to_grayscale(image)
#     # Apply Otsu's thresholding to get the binary image
#     _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # Find contours in the binary image
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Assuming the largest contour is the flower
#     flower_contour = max(contours, key=cv2.contourArea)
#     # Draw the contour on the image
#     contour_image = np.zeros_like(gray_image)
#     cv2.drawContours(contour_image, [flower_contour], -1, 255, thickness=cv2.FILLED)
#     # Create a mask for the contour
#     mask = contour_image.astype(bool)
#     # Create a copy of the original image
#     image_with_roi = image.copy()
#     # Apply the mask to the original image to get the ROI
#     image_with_roi[~mask] = 0
#     return image_with_roi
#
#
# # def find_flower_roi(image):
# #     # Convert to grayscale
# #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     # Apply Otsu's thresholding to get the binary image
# #     _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# #     # Find contours in the binary image
# #     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     # Assuming the largest contour is the flower
# #     flower_contour = max(contours, key=cv2.contourArea)
# #     # Draw the contour on the image
# #     contour_image = np.zeros_like(gray_image)
# #     cv2.drawContours(contour_image, [flower_contour], -1, 255, thickness=cv2.FILLED)
# #     # Create a mask for the contour
# #     mask = contour_image.astype(bool)
# #     # Create a copy of the original image
# #     image_with_roi = image.copy()
# #     # Apply the mask to the original image to get the ROI
# #     image_with_roi[~mask] = 0
# #     return image_with_roi
#
# # Set the base directory for your dataset
# base_dir = 'Dataset/Dataset'
# # Display the original and ROI of each flower in the images for the provided examples
# for folder in ['easy', 'medium', 'hard']:
#     image_files = [f"{base_dir}/input_images/{folder}/{folder}_{i}.jpg" for i in range(1, 4)]
#
#     fig, axs = plt.subplots(len(image_files), 2, figsize=(10, 15))
#
#     for i, image_file in enumerate(image_files):
#         # Read the image
#         image = cv2.imread(image_file)
#         # Validate if the image was loaded properly
#         if image is not None:
#             # Find the ROI containing the flower
#             flower_roi = find_flower_roi(image)
#
#             # Display the original image
#             axs[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#             axs[i, 0].set_title(f'Original {folder.capitalize()} {i+1}')
#             axs[i, 0].axis('off')
#
#             # Display the ROI
#             axs[i, 1].imshow(cv2.cvtColor(flower_roi, cv2.COLOR_BGR2RGB))
#             axs[i, 1].set_title(f'Flower ROI {folder.capitalize()} {i+1}')
#             axs[i, 1].axis('off')
#         else:
#             print(f"Image at {image_file} could not be loaded. Please check the path and try again.")
#
#     plt.tight_layout()
#     plt.show()
#
#
#
#
# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # Function to find the ROI and draw a bounding box around it
# # def find_flower_roi(image):
# #     # Convert to grayscale
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     # Apply a threshold
# #     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# #     # Find contours
# #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     # Assume the largest contour is the flower
# #     largest_contour = max(contours, key=cv2.contourArea)
# #     # Get the bounding box of the largest contour
# #     x, y, w, h = cv2.boundingRect(largest_contour)
# #     # Draw the bounding box on a copy of the original image
# #     roi_marked_image = image.copy()
# #     cv2.rectangle(roi_marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #     return roi_marked_image, (x, y, w, h)
# #
# # image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'
# # image = cv2.imread(image_path)
# #
# # # Find the ROI in the image and draw a bounding box around it
# # marked_image, roi_coords = find_flower_roi(image)
# #
# # # Display the original image and the image with the ROI marked
# # plt.figure(figsize=(12, 6))
# #
# # # Original image
# # plt.subplot(1, 2, 1)
# # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# # plt.title('Original Image')
# # plt.axis('off')
# #
# # # Image with ROI marked
# # plt.subplot(1, 2, 2)
# # plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
# # plt.title('Image with ROI Marked')
# # plt.axis('off')
# #
# # plt.show()
# #
#
#
#
# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from skimage.metrics import structural_similarity as ssim
# #
# #
# # # Helper function to convert image to black and white
# # def convert_BW(img, threshold=50):
# #     _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
# #     return binary_img
# #
# #
# # # Function to calculate SSIM
# # def calculate_ssim(image1, image2):
# #     similarity = ssim(image1, image2)
# #     return similarity
# #
# #
# # def calculate_iou(image1, image2):
# #     # Calculating intersection and union
# #     intersection = np.logical_and(image1, image2)
# #     union = np.logical_or(image1, image2)
# #
# #     # Computing the IoU
# #     iou_score = np.sum(intersection) / np.sum(union)
# #     return iou_score
# #
# #
# # def find_flower_roi(image):
# #     # Convert to grayscale
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #
# #     # Apply a threshold
# #     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# #
# #     # Find contours
# #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     # Assume the largest contour is the flower
# #     largest_contour = max(contours, key=cv2.contourArea)
# #
# #     # Get the bounding box of the largest contour
# #     x, y, w, h = cv2.boundingRect(largest_contour)
# #
# #     return (x, y, w, h), largest_contour
# #
# #
# # def process_image_enhanced_roi(image_path, ground_truth_path):
# #     # Read the image and ground truth
# #     image = cv2.imread(image_path)
# #     ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
# #
# #     # Find ROI coordinates
# #     roi_coords, _ = find_flower_roi(image)
# #     x, y, w, h = roi_coords
# #
# #     # Extract the ROI from the image and the ground truth
# #     roi_image = image[y:y+h, x:x+w]
# #     roi_ground_truth = ground_truth[y:y+h, x:x+w]
# #     roi_ground_truth_binary = convert_BW(roi_ground_truth)
# #
# #     # Apply Median Blur to reduce noise while keeping edges sharp
# #     median_blurred = cv2.medianBlur(roi_image, 15)
# #
# #     # Convert to grayscale
# #     gray_image = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2GRAY)
# #
# #     # Apply Otsu's thresholding after Gaussian filtering
# #     blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
# #     _, binary_thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #     inverted_binary_image = cv2.bitwise_not(binary_thresholded_image)
# #
# #     # Morphological operations to clean up the segmentation
# #     kernel = np.ones((5, 5), np.uint8)
# #     closing = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
# #     opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
# #
# #     # Calculate SSIM and IoU
# #     similarity = calculate_ssim(opening, roi_ground_truth_binary)
# #     iou = calculate_iou(opening, roi_ground_truth_binary)
# #
# #     # Prepare result dict to return
# #     result = {
# #         "original": roi_image,
# #         "gray": gray_image,
# #         "median_blurred": median_blurred,
# #         "binary_thresholded": inverted_binary_image,
# #         "morph_opening": opening,
# #         "ground_truth": roi_ground_truth_binary,
# #         "ssim": similarity,
# #         "iou": iou
# #     }
# #
# #     return result
# #
# #
# # # Example usage with a single pair of images
# # image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'  # Replace with your image path
# # ground_truth_path = 'Dataset/Dataset/ground_truths/easy/easy_1.png'  # Replace with your ground truth path
# #
# # results = process_image_enhanced_roi(image_path, ground_truth_path)
# #
# # titles = ['Original ROI', 'Grayscale ROI', 'Median Blurred ROI',
# #           'Inverted Binary ROI', 'Morphological Opening ROI', 'Ground Truth ROI']
# #
# # plt.figure(figsize=(12, 8))  # Adjust the size as needed
# # for j, (key, title) in enumerate(zip(['original', 'gray', 'median_blurred', 'binary_thresholded',
# #                                       'morph_opening', 'ground_truth'], titles)):
# #     plt.subplot(2, 3, j + 1)
# #     img = results[key]
# #     if img.ndim == 3:
# #         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #     else:
# #         plt.imshow(img, cmap='gray')
# #
# #     plt.title(f"{title}\nSSIM: {results['ssim']:.4f}" if key == 'morph_opening' else title)
# #     plt.axis('off')
# # plt.tight_layout()
# # plt.show()
# #
# # print(f"SSIM value for ROI: {results['ssim']}")
# # print(f"IOU value for ROI: {results['iou']}")
# #
# # # import cv2
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # #
# # #
# # # # Function to read an image
# # # def read_image(image_path):
# # #     return cv2.imread(image_path)#read image and return as multi d array
# # #
# # #
# # # # Function to preprocess the image
# # # # Preprocessing - Convert to a color space that may highlight the flowers better
# # # def preprocess_image(image):
# # #     # Convert to the HSV color space
# # #     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # #     # Use only the hue value for segmentation
# # #     hue_channel = hsv_image[:, :, 0]
# # #     return hue_channel
# # #
# # #
# # # # Segmentation - Use a method that can adapt to different flower colors
# # # def segment_image(image):
# # #     # Adaptive thresholding could be more effective than a fixed threshold
# # #     thresh_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# # #                                          cv2.THRESH_BINARY_INV, 11, 2)
# # #     return thresh_image
# # #
# # #
# # # # Postprocessing - Use closing to fill gaps and remove noise
# # # def postprocess_image(image):
# # #     kernel = np.ones((5, 5), np.uint8)
# # #     closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
# # #     return closing
# # #
# # # # eval segmentation
# # # def make_binary(image, threshold=128, is_ground_truth=False):
# # #     # If this is a ground truth image
# # #     if is_ground_truth:
# # #         # First, check if the image is already in grayscale
# # #         if len(image.shape) == 3:
# # #             # If not, convert to grayscale
# # #             gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # #         else:
# # #             gray_image = image
# # #
# # #         # Assuming the flower is white and the background is black
# # #         # We need to threshold in such a way that white becomes white (255) and black stays black (0)
# # #         # You might need to adjust the threshold value
# # #         # If the flower is darker than the background in the grayscale image, you should invert the threshold type
# # #         _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
# # #
# # #     else:
# # #         # For the segmented images, your logic seems correct; it's just for the ground truth that needs adjustment
# # #         # Process the segmented image
# # #         if len(image.shape) == 3:
# # #             # Assuming the flower is marked in red in the segmented image
# # #             image = image[:, :, 2]
# # #         _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
# # #
# # #     return binary_image
# # #
# # #
# # # # Modify the evaluate_segmentation function accordingly
# # # def evaluate_segmentation(segmented, ground_truth):
# # #     # Convert the ground truth and segmented images to binary masks
# # #     segmented_binary = make_binary(segmented)
# # #     ground_truth_binary = make_binary(ground_truth, is_ground_truth=True)
# # #
# # #     # Calculate IoU
# # #     intersection = np.logical_and(segmented_binary, ground_truth_binary)
# # #     union = np.logical_or(segmented_binary, ground_truth_binary)
# # #     iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
# # #     return iou_score
# # #
# # # # If the segmentation is inverted, invert it back
# # # def invert_colors(image):
# # #     return cv2.bitwise_not(image)
# # #
# # #
# # # # Use this function before evaluating if needed
# # # # Make sure the ground truth is on the same scale as the segmentation result
# # # def align_value_scale(ground_truth):
# # #     # Assuming ground_truth is binary with values 0 and 1
# # #     ground_truth_aligned = ground_truth * 255
# # #     return ground_truth_aligned
# # #
# # #
# # # # Main function where the pipeline is orchestrated
# # # def main():
# # #     image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'  # Replace with your image path
# # #     ground_truth_path = 'Dataset/Dataset/ground_truths/easy/easy_1.png'  # Replace with your ground truth path
# # #
# # #     # Read the image and ground truth
# # #     image = read_image(image_path)
# # #     ground_truth = read_image(ground_truth_path)
# # #
# # #     # Process the image
# # #     preprocessed = preprocess_image(image)
# # #     segmented = segment_image(preprocessed)
# # #     postprocessed = postprocess_image(segmented)
# # #
# # #     # Set up the matplotlib figure and axes, organized in a 2 x 2 grid
# # #     fig, axs = plt.subplots(4, 2, figsize=(10, 8))
# # #
# # #     # Show the original image
# # #     axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# # #     axs[0, 0].set_title('Original')
# # #     axs[0, 0].axis('off')
# # #
# # #     # Show the preprocessed image
# # #     axs[0, 1].imshow(preprocessed, cmap='gray')
# # #     axs[0, 1].set_title('Preprocessed')
# # #     axs[0, 1].axis('off')
# # #
# # #     # Show the segmented image
# # #     axs[1, 0].imshow(segmented, cmap='gray')
# # #     axs[1, 0].set_title('Segmented')
# # #     axs[1, 0].axis('off')
# # #
# # #     # Show the postprocessed image
# # #     axs[1, 1].imshow(postprocessed, cmap='gray')
# # #     axs[1, 1].set_title('Postprocessed')
# # #     axs[1, 1].axis('off')
# # #
# # #     axs[2, 0].imshow(ground_truth, cmap='gray')
# # #     axs[2, 0].set_title('Groundtrth')
# # #     axs[2, 0].axis('off')
# # #
# # #     # Display the figure
# # #
# # #     # Evaluate the result
# # #     ground_truth_binary = make_binary(ground_truth, threshold=128, is_ground_truth=True)
# # #     axs[3, 1].imshow(ground_truth_binary, cmap='gray')
# # #     axs[3, 1].set_title('GTbin')
# # #     axs[3, 1].axis('off')
# # #     # Convert the postprocessed image to a binary mask if necessary
# # #     postprocessed_binary = make_binary(postprocessed, threshold=128)
# # #
# # #     axs[3, 0].imshow(postprocessed_binary, cmap='gray')
# # #     axs[3, 0].set_title('PPbin')
# # #     axs[3, 0].axis('off')
# # #     plt.show()
# # #     # Evaluate the result
# # #     iou_score = evaluate_segmentation(postprocessed_binary, ground_truth_binary)
# # #     print(f"The Intersection over Union score is: {iou_score}")
# # #
# # #     # Show the result images
# # #     cv2.imshow('Original', image)
# # #     cv2.imshow('Segmented', segmented)
# # #     cv2.imshow('Postprocessed', postprocessed)
# # #     cv2.waitKey(0)
# # #     cv2.destroyAllWindows()
# # #
# # #
# # # if __name__ == '__main__':
# # #     main()
# #
# #
# #
# # #
# # # import cv2
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # #
# # # # Read the image
# # # image_path = 'Dataset/Dataset/input_images/easy/easy_1.jpg'
# # # image = cv2.imread(image_path)
# # #
# # # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # #
# # # # Apply contrast stretching to the grayscale image
# # # # Find the minimum and maximum pixel values
# # # minval = np.min(gray_image)
# # # maxval = np.max(gray_image)
# # #
# # # # Apply the contrast stretching formula
# # # contrast_stretched_image = ((gray_image - minval) / (maxval - minval)) * 255
# # # contrast_stretched_image = contrast_stretched_image.astype(np.uint8)
# # #
# # # # Apply binary thresholding to separate the flower from the background
# # # # We assume the flower is lighter than the background, so we use an inverse binary threshold
# # # _, binary_thresholded_image = cv2.threshold(contrast_stretched_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# # #
# # # # Display the original, contrast stretched, and binary thresholded images using matplotlib
# # # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# # #
# # # # Original Image
# # # axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# # # axs[0].set_title('Original Image')
# # # axs[0].axis('off')
# # #
# # # # Contrast Stretched Image
# # # axs[1].imshow(contrast_stretched_image, cmap='gray')
# # # axs[1].set_title('Contrast Stretched Image')
# # # axs[1].axis('off')
# # #
# # # # Binary Thresholded Image
# # # axs[2].imshow(binary_thresholded_image, cmap='gray')
# # # axs[2].set_title('Binary Thresholded Image')
# # # axs[2].axis('off')
# # #
# # # # Show the plot
# # # plt.show()
