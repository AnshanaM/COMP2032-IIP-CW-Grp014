def convert_BW(img): # converts the image to grayscale
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_bw
    
def binary_threshold(img, threshold_num, max_num):
    _, bi_image = cv2.threshold(img, threshold_num, max_num, cv2.THRESH_BINARY)
    return bi_image

def contrast_stretch(img, min_num, max_num): # adjusts the image to the new minimum and maximum values
    contrast_img = cv2.normalize(img, None, min_num, max_num, norm_type=cv2.NORM_MINMAX)
    return contrast_img

def image_pipeline(img):# function to apply all processing functions to image
    bw_img = convert_BW(img)
    contrast_img=contrast_stretch(bw_img,0,255)
    final_img = binary_threshold(contrast_img,140,255)
    return final_img

def saveImage(img, img_name,img_difficulty,img_no):
    #save image to the image pipeline with the img_name


def display_final_images(image_pipeline): # using your imagepipeline() function, it will be applied on all images and displayed
    file_names = ["easy","medium","hard"]
    # getting all images file directories
    # C:\\Users\\Manoharan\\Desktop\\IP CW\\Dataset\\input_images\\easy\\easy_1.jpg
    
    default_file_directory = "C:\\Users\\Manoharan\\Desktop\\IP CW\\Dataset\\input_images\\"
    
    for j in file_names:
        
        first_file_directory = default_file_directory+j+"\\"+j+"_"+"1"+".jpg"
        sec_file_directory = default_file_directory+j+"\\"+j+"_"+"2"+".jpg"
        third_file_directory = default_file_directory+j+"\\"+j+"_"+"3"+".jpg"
            
        image1=cv2.imread(first_file_directory, cv2.IMREAD_COLOR)
        image2=cv2.imread(sec_file_directory, cv2.IMREAD_COLOR)
        image3=cv2.imread(third_file_directory, cv2.IMREAD_COLOR)
            
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(7, 6))
        ax1.imshow(image_pipeline(image1), cmap="gray")
        ax1.set_title(j+"_1")
        ax2.imshow(image_pipeline(image2), cmap="gray")
        ax2.set_title(j+"_2")
        ax3.imshow(image_pipeline(image3), cmap="gray")
        ax3.set_title(j+"_3")

        plt.tight_layout()
        plt.show()
                
    
    
display_final_images(image_pipeline)
