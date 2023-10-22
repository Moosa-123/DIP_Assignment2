

import cv2
import numpy as np

# Load the original image
image = cv2.imread("E:\7th Sem\DIP\Assignment2\data\1-3.jpg")


# Define the region of interest (ROI) coordinates and dimensions
region_x = 310  
region_y = 40   


roi_width = 160  
roi_height = 140  

# Extract the ROI as a template
template = image[region_y:region_y + roi_height, region_x:region_x + roi_width]

# Define the lower region

lower_region_x = 0  
lower_region_y = 200  


lower_roi_width = image.shape[1]  
lower_roi_height = image.shape[0] - lower_region_y  


#Code to crop the lower region of the image

lower_region = image[lower_region_y:lower_region_y + lower_roi_height, lower_region_x:lower_region_x + lower_roi_width]


# Using CV's match funtion to check if two templates match
check_match_found = cv2.matchTemplate(lower_region, template, cv2.TM_CCOEFF_NORMED)

#Amount of threshold which will show if match or not
min_thres = 0.95


# Matches above the threshold assigned
Boxes_match = np.where(check_match_found >= min_thres)

# Reverse the coordinates
Boxes_match = list(zip(*Boxes_match[::-1]))  

##IF no threshold no match then print
if not Boxes_match:
    print("No Box to be match found")
#IF match found
else:
    # Draw rectangles around the matched areas on the original image
    for location in Boxes_match:
        
        top_left = (location[0] + lower_region_x, location[1] + lower_region_y)
        template_height, template_width, _ = template.shape
        
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        
        
    #Showing the image with a green outline outside the boxes which match
    cv2.imshow('Images with match are shown as', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:




