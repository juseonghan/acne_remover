import numpy as np
import cv2 as cv
import argparse


# importing required packages
import cv2
import numpy as np
 
def pop_acne(img, points):
    # first point is top left pt
    # bottom left is bottom right pt
    # goal is to get gradient values and interpolate them in the ROI
    # coordinate system is centered on top left of image, with +x in right and +y to down

    firstpt = points[0] # top left
    secondpt = points[1] # bottom right
    top_left = (firstpt[1], firstpt[0])
    bottom_right = (secondpt[1], secondpt[0])
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    img = cv.transpose(img) 
    
    print("Top Left:", top_left)
    print("Top Right:", top_right)
    print("Bottom Left:", bottom_left)
    print("Bottom_Right:", bottom_right)

    num_rows = bottom_left[1] - top_left[1] 
    num_columns = top_right[0] - top_left[0]

    # first generate horizontal gradients
    img_patch_hor = np.zeros((num_rows, num_columns, 3))

    for row in range(num_rows):
        img_patch_hor[row, :, 0] = linear_interpolation(img[row + top_left[1], top_left[0], 0], img[row+top_left[1], top_right[0], 0], num_columns)
        img_patch_hor[row, :, 1] = linear_interpolation(img[row + top_left[1], top_left[0], 1], img[row+top_left[1], top_right[0], 1], num_columns)
        img_patch_hor[row, :, 2] = linear_interpolation(img[row + top_left[1], top_left[0], 2], img[row+top_left[1], top_right[0], 2], num_columns)

    # second generate vertical gradients
    img_patch_ver = np.zeros((num_rows, num_columns, 3))

    for col in range(num_columns):
        img_patch_ver[:, col, 0] = linear_interpolation(img[top_left[1], col + top_left[0], 0], img[bottom_left[1], col + top_left[0], 0], num_rows)
        img_patch_ver[:, col, 1] = linear_interpolation(img[top_left[1], col + top_left[0], 1], img[bottom_left[1], col + top_left[0], 1], num_rows)
        img_patch_ver[:, col, 2] = linear_interpolation(img[top_left[1], col + top_left[0], 2], img[bottom_left[1], col + top_left[0], 2], num_rows)

    # # now we average the two images together
    final_patch = (img_patch_ver + img_patch_hor) / 2
    # print(final_patch)
    img[top_left[1]:bottom_left[1], top_left[0]:top_right[0],:] = final_patch
    final_img = cv.transpose(img)
    cv.imwrite('final_2.jpg', final_img)

def linear_interpolation(start, end, num_pts):
    ## returns a numpy array of length num_pts that starts with start linear interpolates until end
    res = np.zeros(num_pts)
    
    for i in range(num_pts):
        res[i] = int(int(start) + (int(end) - int(start))/(num_pts - 1) * i)

    return res

# mouse call back function
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
       
        points.append((x,y))
        # cv2.circle(img,(x,y), 4, (0, 255, 0), -1)

        if len(points) == 2:
            pop_acne(img, points)
            return
        cv2.imshow('image', img)

img = cv.imread('two.jpg')
points = []

cv2.imshow('image',img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# st = 101
# end = 15
# num_pts = 10
# res = linear_interpolation(st, end, num_pts)
# print(res)