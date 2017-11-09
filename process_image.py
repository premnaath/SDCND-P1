import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2


def process_image(image):

    # Transform to grey-scale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply guassian blurr
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur_gray, 150, 200)

    # Mask using cv2.fillPoly.
    # Here two maskings are done.
    # Mask 1 - Performed with vertices to select valid Region Of Interest, where the lanes occur.
    # Mask 2 - Performed on the Mask 1, in between the lane lines. It was found during testing that some
    # bright parts may be considered as lane participants. Those will be filtered in this mask.
    mask = np.zeros_like(edges)
    mask_color = 255
    ignore_mask_color = 0

    imshape = image.shape
    # A quadrilateral
    vertices = np.array([[(140, imshape[0]), (430, 325), (520, 325), (900, imshape[0])]], dtype=np.int32)
    # A triangle
    vertices_ignore = np.array([[(250, imshape[0]), (475, 375), (700, imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, mask_color)
    cv2.fillPoly(mask, vertices_ignore, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Pre-process the line segments.
    # Here lines with are almost vertical/horizontal are filtered away.
    lines_preprocessed = []
    kMinPixelThreshold = 2
    kDistanceThreshold = 15

    for linepp in lines:
        for x1, y1, x2, y2 in linepp:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            delta_x = x2 - x1
            delta_y = y2 - y1
            if abs(delta_x) > kMinPixelThreshold | abs(delta_y) > kMinPixelThreshold:
                lines_preprocessed.append(linepp)

    lines_pp = np.array(lines_preprocessed)

    # Classify lines based on it's slope
    length_l = []
    length_r = []
    lefts = []
    rights = []

    mean_separation_x = 475 # Mean in x-direction of the considered Region mask 1.

    # Iterate over the output "lines" and draw lines on a blank image
    # Classify the lines to be belonging to left or right lane.
    # This classification is carried out with two parameters.
    # Parameter 1 - The slope. The slope of the left lane will be negative and vice-versa for the left lane.
    # Parameter 2 - The partition which the line belongs to on the Resion mask 1. Eg: left lanes belong to the
    # left side of the quadrilateral.
    for line in lines_pp:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            length = math.hypot((x2 - x1), (y2 - y1))
            if (slope < 0) & (x1 < mean_separation_x) & (x2 < mean_separation_x):
                length_l.append(length)
                lefts.append([[length, x1, y1, x2, y2]])
            elif (slope > 0) & (x1 > mean_separation_x) & (x2 > mean_separation_x):
                length_r.append(length)
                rights.append([[length, x1, y1, x2, y2]])

    length_left = np.array(length_l)
    length_right = np.array(length_r)

    # Remove outliers near the solid line
    # During testing small white patches were seen closer to the side of the solid lanes (often this was the casr).
    # Since their presence directly affects the linear line fit parameters, they are removed here.
    solid_mean_length = 100
    to_remove_left = []
    to_remove_right = []

    # If left lane is solid lane, find the lines which are of short distances and closer to it.
    if np.mean(length_left) > solid_mean_length:
        for element in lefts:
            for line_length, x1, y1, x2, y2 in element:
                if line_length <= kDistanceThreshold:
                    to_remove_left.append(element)
    # vice-versa for right lane.
    elif np.mean(length_right) > solid_mean_length:
        for element in rights:
            for line_length, x1, y1, x2, y2 in element:
                if line_length <= kDistanceThreshold:
                    to_remove_right.append(element)

    # Remove the invalid lines.
    lefts = [e for e in lefts if e not in to_remove_left]
    rights = [e for e in rights if e not in to_remove_right]

    # Prepare data for Linear Regression.
    x_l = []
    y_l = []
    x_r = []
    y_r = []

    for l in lefts:
        for length, x1, y1, x2, y2 in l:
            x_l.append(x1)
            x_l.append(x2)
            y_l.append(y1)
            y_l.append(y2)

    x_left = np.array(x_l)
    y_left = np.array(y_l)

    for r in rights:
        for length, x1, y1, x2, y2 in r:
            x_r.append(x1)
            x_r.append(x2)
            y_r.append(y1)
            y_r.append(y2)

    x_right = np.array(x_r)
    y_right = np.array(y_r)

    # Merge all lines into one and perform 1-st order linear regression upon it.
    # Left
    if x_left is not None:
        if y_left is not None:
            param_left = np.polyfit(x_left, y_left, 1)
    # Right
    if x_right is not None:
        if y_right is not None:
            param_right = np.polyfit(x_right, y_right, 1)

    # Use returned fit parameters to extrapolate the  lane lines.
    # There may be more of the lane visible. This is evaluated now by region masking with a triangle over
    # the Region mask 1.
    mask_extend = np.zeros_like(edges)
    mask_extend_color = 255
    # Define vertex of the triangle.
    left_vertex = (np.max(x_left) - 15, np.min(y_left))
    right_vertex = (np.min(x_right) + 15, np.min(y_right))
    mid_point = int((left_vertex[0] + right_vertex[0]) / 2)
    height = int((left_vertex[1] + right_vertex[1]) / 2) - 30
    # Fill the triangle with maxmum intensity.
    vertices_extend = np.array([[left_vertex, (mid_point, height), right_vertex]], dtype=np.int32)
    cv2.fillPoly(mask_extend, vertices_extend, mask_extend_color)
    extended_edges = cv2.bitwise_and(edges, mask_extend)

    # Draw Hough Lines on the extended edges.
    lines_extended = cv2.HoughLinesP(extended_edges, rho, theta, 5, np.array([]), 5, 2)

    # Check if the above lines correspond to any of the lanes with the help of already computed slope.
    # Define extrapolation values as default
    default_y_extrap_value = 325
    extrap_left_y = default_y_extrap_value
    extrap_right_y = default_y_extrap_value
    # Slope threshold for the line to be a viable participant to any of the lane.
    delta_slope_threshold = 0.01

    left_extend = []
    right_extend = []
    x_extend_left = []
    y_extend_left = []
    x_extend_right = []
    y_extend_right = []
    # Classify lines belonging to left or right only using the slope.
    if lines_extended is not None:
        for each_line in lines_extended:
            for x1, y1, x2, y2 in each_line:
                slope = (y2 - y1) / (x2 - x1)
                if abs(param_left[0] - slope) < delta_slope_threshold:
                    left_extend.append(each_line)
                    x_extend_left.append(x1)
                    x_extend_left.append(x2)
                    y_extend_left.append(y1)
                    y_extend_left.append(y2)

                elif abs(param_right[0] - slope) < delta_slope_threshold:
                    right_extend.append(each_line)
                    x_extend_right.append(x1)
                    x_extend_right.append(x2)
                    y_extend_right.append(y1)
                    y_extend_right.append(y2)

    x_extend_left = np.array(x_extend_left)
    y_extend_left = np.array(y_extend_left)

    x_extend_right = np.array(x_extend_right)
    y_extend_right = np.array(y_extend_right)

    # Compute linear fit parameters on the classified lines.
    # If the slope of a line segment is within the defined threshold, then that line can become a possible extension.
    if not x_extend_left.size == 0:
        if not y_extend_left.size == 0:
            extended_param_left = np.polyfit(x_extend_left, y_extend_left, 1)
            if abs(extended_param_left[0] - param_left[0]) < delta_slope_threshold:
                possible_extension_y_left = np.mean(y_extend_left)
                if possible_extension_y_left < extrap_left_y:
                    # Here the extension considered can be very noisy, hence only 50% of it is considered.
                    extrap_left_y = int((possible_extension_y_left + default_y_extrap_value)/2)
    # Same for the right side.
    if not x_extend_right.size == 0:
        if not y_extend_right.size == 0:
            extended_param_right = np.polyfit(x_extend_right, y_extend_right, 1)
            if abs(extended_param_right[0] - param_right[0]) < delta_slope_threshold:
                possible_extension_y_right = np.mean(y_extend_right)
                if possible_extension_y_right < extrap_right_y:
                    extrap_right_y = int((possible_extension_y_right + default_y_extrap_value)/2)

    # A new extrapolated y-value might be available now.
    # Create y-points
    y_spaced_left = np.array([extrap_left_y, imshape[0]])
    y_spaced_right = np.array([extrap_right_y, imshape[0]])
    x_spaced_l = []
    x_spaced_r = []

    # Compute the x-points.
    for y_left in y_spaced_left:
        x_l = int((y_left - param_left[1]) / param_left[0])
        x_spaced_l.append(x_l)

    x_spaced_left = np.array(x_spaced_l)

    for y_right in y_spaced_right:
        y_l = int((y_right - param_right[1]) / param_right[0])
        x_spaced_r.append(y_l)

    x_spaced_right = np.array(x_spaced_r)

    # Draw the raw lines on the image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # Draw the extrapolated lines on the image
    lane_image = np.copy(image) * 0  # creating a blank to draw lines on
    cv2.line(lane_image, (x_spaced_left[0], y_spaced_left[0]), (x_spaced_left[1], y_spaced_left[1]), (255, 0, 0), 5)
    cv2.line(lane_image, (x_spaced_right[0], y_spaced_right[0]), (x_spaced_right[1], y_spaced_right[1]), (255, 0, 0), 5)
    lanes_full = cv2.addWeighted(image, 0.8, lane_image, 1, 0)

    return lanes_full
