import numpy as np
import cv2 as cv
import time 


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
    # top_left = points[0]
    # bottom_right = points[1]
    # top_right = (top_left[0], bottom_right[1])
    # bottom_left = (bottom_right[0], top_left[1])
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    img = cv.transpose(img) 
    
    print("Top Left:", top_left)
    print("Top Right:", top_right)
    print("Bottom Left:", bottom_left)
    print("Bottom_Right:", bottom_right)

    # # first just make the box black
    # num_rows = bottom_left[1] - top_left[1]
    # num_cols = top_right[0] - top_left[0]
    # img_patch = np.zeros((num_rows, num_cols, 3))
    # print("image patch shape:", img_patch.shape)
    # print("image shape:", img.shape)
    # img[top_left[1]:bottom_left[1], top_left[0]:top_right[0],:] = img_patch
    # # img[top_left[0]+1:bottom_left[0], top_left[1]+1:top_right[1],:] = img_patch

    # img = cv.transpose(img)
    # cv.imwrite('final.jpg', img)

    # first do horizontal, ie get gradient from left edge to right edge
    # hor_gradient is a list that contains difference between right edge and left edge for each row 
    # specifically, we are doing right - left subtraction 
    num_rows = bottom_left[1] - top_left[1] 
    print(f"There are {num_rows} rows")
    hor_gradient = []
    for row in range(num_rows): 
        grad_b = int(img[top_right[1] + row, top_right[0], 0]) - int(img[top_left[1] + row, top_left[0], 0])
        grad_g = int(img[top_right[1] + row, top_right[0], 1]) - int(img[top_left[1] + row, top_left[0], 1])
        grad_r = int(img[top_right[1] + row, top_right[0], 2]) - int(img[top_left[1] + row, top_left[0], 2])
        grads = np.array([grad_b, grad_g, grad_r])
        hor_gradient.append(grads)

    print("df/dx vector:", hor_gradient)
    # now do vertical, ie get gradient from top edge to bottom edge
    # ver_gradient is a list that contains difference between top and bottom edge for each column
    # specifically, we are doing top - bottom subtraction. 
    num_columns = top_right[0] - top_left[0]
    print(f"There are {num_columns} columns")
    ver_gradient = []
    for col in range(num_columns):
        grad_b = int(img[top_left[1], top_left[0] + col, 0]) - int(img[bottom_left[1], bottom_left[0]+ col, 0])
        grad_g = int(img[top_left[1], top_left[0] + col, 1]) - int(img[bottom_left[1], bottom_left[0]+ col, 1])
        grad_r = int(img[top_left[1], top_left[0] + col, 2]) - int(img[bottom_left[1], bottom_left[0]+ col, 2])
        grads = np.array([grad_b, grad_g, grad_r])
        ver_gradient.append(grads)
    
    print("df/dy vector:", ver_gradient)

    # we make an img_patch that finds interpolated values using hor_gradient
    img_patch_hor = np.zeros((num_rows, num_columns, 3))
    for row in range(num_rows):
        start_val = img[top_left[1] + row, top_left[0], :]
        for col in range(num_columns):
            img_patch_hor[row, col, 0] = int(start_val[0]) + col * int(hor_gradient[row][0])/num_columns
            img_patch_hor[row, col, 1] = int(start_val[1]) + col * int(hor_gradient[row][1])/num_columns
            img_patch_hor[row, col, 2] = int(start_val[2]) + col * int(hor_gradient[row][2])/num_columns

    # now we make img_patch that finds interpolated values using ver_gradient
    img_patch_ver = np.zeros((num_rows, num_columns, 3))
    for col in range(num_columns):
        start_val = img[bottom_left[1], bottom_left[0] + col, :]
        for row in range(num_rows):
            img_patch_ver[row, col, 0] = int(start_val[0]) + row * int(ver_gradient[col][0])/num_rows
            img_patch_ver[row, col, 1] = int(start_val[1]) + row * int(ver_gradient[col][1])/num_rows
            img_patch_ver[row, col, 2] = int(start_val[2]) + row * int(ver_gradient[col][2])/num_rows

    # now we average the two images together
    final_patch = (img_patch_ver + img_patch_hor) / 2
    # print(final_patch)
    img[top_left[1]:bottom_left[1], top_left[0]:top_right[0],:] = final_patch
    final_img = cv.transpose(img)
    cv.imwrite('final.jpg', final_img)


# mouse call back function
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
       
        points.append((x,y))
        # cv2.circle(img,(x,y), 4, (0, 255, 0), -1)

        if len(points) == 2:
            pop_acne(img, points)
            return
        cv2.imshow('image', img)

img = cv.imread('sample.jpg')
points = []

cv2.imshow('image',img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()