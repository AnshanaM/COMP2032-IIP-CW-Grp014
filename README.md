<h1>COMP2032 Coursework</h1>

<b>Overview</b>

The main.py file contains a Python script for processing images using various image processing techniques and evaluating the results. 

<b>Requirements</b>

To run the main.py file, you need the following module packages installed:
- numpy
- opencv-python
- scikit-image
- os

You can install these modules using the Python packages option in Pycharm

**OR**

You can install these modules using pip:

<i>pip install numpy opencv-python scikit-image matplotlib</i>

<b>Prerequisites</b>

Before running the main.py file, ensure that you have the following:

- Input images: Images to be processed should be placed in the Dataset/input_images/ directory.
- Ground truth images: Ground truth images corresponding to the input images should be placed in the Dataset/ground_truths/ directory.
- Directory structure: The directory structure should be maintained as specified in the code to correctly read input and ground truth images.
- Output directories: Output directories will be created automatically to store processed images and intermediate results. The following are the output directories after After execution:
  - image-processing-pipeline - stores the images in their respective subfolders after each step of the pipeline
  - output_folder - stores the final processed images after all the steps in the pipeline
  - processed_ground_truth - stores the processed ground truth images

<b>Execution</b>

To execute the image processing pipeline, run the main.py file. The script will process the input images, apply various image processing techniques, and evaluate the results based on ground truth images. The evaluation will be printed on the console.

**Expected folder structure _before_ execution:**
- Dataset
  - ground_truths
    - easy
      - easy_1.png
      - easy_2.png
      - easy_3.png
    - medium
      - medium_1.png 
      - medium_2.png 
      - medium_3.png
    - hard
      - hard_1.png 
      - hard_2.png 
      - hard_3.png
  - input_images
    - easy
      - easy_1.jpg
      - easy_2.jpg
      - easy_3.jpg
    - medium
      - medium_1.jpg
      - medium_2.jpg
      - medium_3.jpg
    - hard
      - hard_1.jpg
      - hard_2.jpg
      - hard_3.jpg
- main.py
- README.md

**Expected folder structure _after_ execution:**
- Dataset
  - ground_truths
    - easy
      - easy_1.png
      - easy_2.png
      - easy_3.png
    - medium
      - medium_1.png 
      - medium_2.png 
      - medium_3.png
    - hard
      - hard_1.png 
      - hard_2.png 
      - hard_3.png
  - input_images
    - easy
      - easy_1.jpg
      - easy_2.jpg
      - easy_3.jpg
    - medium
      - medium_1.jpg
      - medium_2.jpg
      - medium_3.jpg
    - hard
      - hard_1.jpg
      - hard_2.jpg
      - hard_3.jpg
- Image Processing Pipeline
  - easy
    - easy_1
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
    - easy_2
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
    - easy_3
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
  - medium
    - medium_1
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
    - medium_2
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
    - medium_3
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
  - hard
    - hard_1
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
    - hard_2
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg
    - hard_3
      - 1_grayscale.jpg
      - 2_gamma.jpg
      - 3_foreground.jpg
      - 4_negated.jpg
      - 5_histogram_equalisation.jpg
      - 6_otsu.jpg
      - 7_ccl_lss.jpg 
- Output
  - easy
    - easy_1.jpg
    - easy_2.jpg
    - easy_3.jpg
  - medium
    - medium_1.jpg
    - medium_2.jpg
    - medium_3.jpg
  - hard
    - hard_1.jpg
    - hard_2.jpg
    - hard_3.jpg
- processed_ground_truth
  - easy
   - easy_1.jpg
   - easy_2.jpg
   - easy_3.jpg
  - medium
    - medium_1.jpg
    - medium_2.jpg
    - medium_3.jpg
  - hard
    - hard_1.jpg
    - hard_2.jpg
    - hard_3.jpg
- main.py
- README.md


<b>Output</b>

The processed images and intermediate results will be saved in the output_folder/ directory. The processed ground truth images will be saved in the processed_ground_truth/ directory. Additionally, intermediate results from the image processing pipeline will be saved in the image-processing-pipeline/ directory.

<b>Additional Notes</b>
The pipeline may take some time to execute depending on the size and number of input images

<b>Example Usage</b>
- This script can be run on **any** Python IDE or using Command Prompt with the following command in the directory where main.py and Dataset (this folder must have the ground truths and the input images) folder exists:
_python main.py_

<b>Acknowledgments</b>

The script uses various image processing techniques inspired by existing literature and tutorials.
This README documentation provides an overview of the main.py file, its requirements, execution steps, and output. Users can refer to this documentation to set up and run the image processing pipeline successfully.
