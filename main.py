import numpy as np
import cv2
from skimage.measure import regionprops, label
import os

def convert_grayscale(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_bw

def inc_gain(img, gain_factor):
    scaled_image = img * gain_factor
    scaled_image[scaled_image > 255] = 255
    return scaled_image

def histogramEq(img):
    img = img.astype(np.uint8)
    if len(img.shape) > 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    eq_img = cv2.equalizeHist(gray_img)
    return eq_img


def otsu_threshold(img):
    if len(img.shape) > 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    if gray_img.dtype != 'uint8':
        gray_img = gray_img.astype('uint8')
    _, otsu_threshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return otsu_threshold


def gamma_Correction(img, g):
    gamma = g
    gamma_img = ((img / 255) ** gamma) * 255
    gamma_img = gamma_img.astype("uint8")
    return gamma_img


def erosion(binary_image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    return eroded_image


def dilation(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def extract_foreground(image):
    if image.dtype != 'uint8':
        image = image.astype('uint8')
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], np.uint8)
    rectangle = (1, 1, width, height)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.convertScaleAbs(image)
    cv2.grabCut(image, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)  # 5
    foreground_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    foreground = cv2.bitwise_and(image, image, mask=foreground_mask)
    return foreground

def morphology_close(image,kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing_image

def CCL_LSS(binary_image): # performs connected component labelling and largest segment selection
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    largest_label = max(regions, key=lambda region: region.area).label
    largest_black_blob_mask = labeled_image == largest_label
    largest_black_blob_image = np.zeros_like(
        binary_image)
    largest_black_blob_image[largest_black_blob_mask] = 1
    return largest_black_blob_image

def negation(img):
    if img.dtype != 'uint8':
        img = img.astype('uint8')
    negated_image = 255 - img
    return negated_image

# processing performed on ground truth:
# extract red foreground on ground truth images
# with just red foreground, the flower segmented was in two separate regions for some images
# slight dilation was performed to match the original image better

def process_ground_truth(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(grayscale_image.shape[0]):
        for j in range(grayscale_image.shape[1]):
            pixel_value = grayscale_image[i, j]
            if pixel_value == 38: # extract red foreground
                grayscale_image[i, j] = 1
            else:
                grayscale_image[i, j] = 0
    return CCL_LSS(dilation(grayscale_image, 7))

def calculate_overlap_score(ground_truth_mask, predicted_mask):
    intersection = np.logical_and(predicted_mask,ground_truth_mask)
    union = np.logical_or(predicted_mask,ground_truth_mask)
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    iou = intersection_area/ (union_area + 1e-6) #adding epsilon to avoid division by 0
    return iou

def mean_squared_error(ground_truth, final):
    squared_diff = (ground_truth - final) ** 2
    mse = np.mean(squared_diff)
    return mse

def calculate_miou(ground_truth, final):
    intersection = np.logical_and(final == 1, ground_truth == 1)
    union = np.logical_or(final == 1, ground_truth == 1)
    iou_class1 = np.sum(intersection) / np.sum(union)
    intersection_bg = np.logical_and(final == 0, ground_truth == 0)
    union_bg = np.logical_or(final == 0, ground_truth == 0)
    iou_class0 = np.sum(intersection_bg) / np.sum(union_bg)
    miou = (iou_class1 + iou_class0) / 2
    return miou

# creating output folders to store the final images, processed ground truths and images as they pass through the pipeline
current_directory = os.getcwd()

new_folder_path = os.path.join(current_directory, 'Output')
os.makedirs(new_folder_path, exist_ok=True)
file_names = ['easy', 'medium', 'hard']
for folder in file_names:
    folder_path = os.path.join(new_folder_path, folder)
    os.makedirs(folder_path, exist_ok=True)

new_folder_path = os.path.join(current_directory, 'processed_ground_truth')
os.makedirs(new_folder_path, exist_ok=True)
file_names = ['easy', 'medium', 'hard']
sub_folders = ['1', '2', '3']
for folder in file_names:
    folder_path = os.path.join(new_folder_path, folder)
    os.makedirs(folder_path, exist_ok=True)


main_folder = os.path.join(current_directory, 'Image Processing Pipeline')
os.makedirs(main_folder, exist_ok=True)
for folder_name in file_names:
    folder_path = os.path.join(main_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    for subfolder_name in sub_folders:
        subfolder_path = os.path.join(folder_path, folder_name + '_' + subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

# customizable function to apply all processing functions to image
def image_pipeline(img, ground,j,count):
    grayscale = convert_grayscale(img)
    gamma_img= gamma_Correction(grayscale,3.5)
    foreground_img = extract_foreground(gamma_img)
    negated_img = negation(foreground_img)
    hist_eq_img = histogramEq(negated_img)
    otsu_thresh_img = otsu_threshold(hist_eq_img)
    final_img = CCL_LSS(otsu_thresh_img)

    processed_groundt_img = process_ground_truth(ground)

    print("\nImage name: " + j + "_" + str(count))
    print("{:<25} {:<25}".format("Evaluation Metric Type", "Evaluation Metric Result"))
    print("{:<25} {:<25.5f}".format("mIoU (%)",
                                              calculate_miou(processed_groundt_img, final_img) * 100))
    print("{:<25} {:<25.5f}".format("Overlap Score (%)",
                                              calculate_overlap_score(processed_groundt_img, final_img) * 100))
    print("{:<25} {:<25.5f}".format("Mean Squared Error",
                                              mean_squared_error(processed_groundt_img, final_img)))

    # convert binary images to RGB format
    final_img = final_img*255
    processed_groundt_img = processed_groundt_img*255
    return final_img, processed_groundt_img, [grayscale, gamma_img, negated_img, foreground_img, hist_eq_img, otsu_thresh_img,final_img]

half_processed_file_names = ["1_grayscale", "2_gamma","3_foreground","4_negated", "5_histogram_equalisation", "6_otsu","7_ccl_lss"]
def execute_pipeline(image_pipeline):
    input_file_directory = ".\\Dataset\\input_images\\"
    ground_truth_directory = ".\\Dataset\\ground_truths\\"
    output_file_directory = ".\\Output\\"
    processed_ground_truth_file_directory = ".\\processed_ground_truth\\"
    pipeline_directory = ".\\Image Processing Pipeline\\"

    for j in file_names:

        input_directory = []
        ground_directory = []
        final_image_directory = []
        processed_ground_directory = []
        groundt_binary_directory = []

        input_images = []
        ground_images = []
        pipeline_image_list = []
        processed_ground_images = []
        groundt_binary_images = []

        # composing directories
        for num in sub_folders:
            input_directory.append(input_file_directory + j + "\\" + j + "_" + num + ".jpg")
            ground_directory.append(ground_truth_directory + j + "\\" + j + "_" + num + ".png")
            final_image_directory.append(output_file_directory + j + "\\" + j + "_" + num + ".jpg")
            processed_ground_directory.append(processed_ground_truth_file_directory + j + "\\" + j + "_" + num + ".jpg")

        # getting input images
        for directory in input_directory:
            input_images.append(cv2.imread(directory, cv2.IMREAD_COLOR))

        # getting ground truth images
        for directory in ground_directory:
            ground_images.append(cv2.imread(directory, cv2.IMREAD_COLOR))

        # getting final images
        count = 1
        for (input_img, ground_img, finalimgdir) in zip(input_images, ground_images, final_image_directory):
            fimage, ground_image, array = image_pipeline(input_img, ground_img, j, count)
            pipeline_image_list.append(array)
            processed_ground_images.append(ground_image)
            cv2.imwrite(finalimgdir, fimage)
            count += 1

        # storing the processed ground truths
        for (directory, img) in zip(processed_ground_directory,processed_ground_images):
            cv2.imwrite(directory,img)


        # storing the intermediate images from the pipeline
        for i in range(1,4):
            image_directory = pipeline_directory+j+"\\"+j+"_"+str(i)
            for (file_name,image) in zip(half_processed_file_names, pipeline_image_list[i-1]):
                cv2.imwrite(image_directory+"\\"+file_name+".jpg", image)

execute_pipeline(image_pipeline)
