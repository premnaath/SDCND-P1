# Finding Lane Lines on the Road


## Project report
#### By Premnaath Sukumar

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
---


### Reflection


### 1. Implemented pipeline

The implemented pipeline consists of 10 main steps listed as follows,
1. Convert to Grayscale.
2. Gaussian blur to remove noise.
3. Canny to find edges.
4. Region Of Interest mask on edges.
5. Hough line segments on edges.
6. Pre-process line segments.
7. Classify lane lines.
8. Filter short lines in left and right lane candidates.
9. Linear line fitting.
10. Find extension of lanes. 
11. Plot them.

#### 1. Convert to grayscale

The 3-channel RGB image has a lot of information. In order for us to begin with processing and make it easier in a way, the image is converted to grayscale. Now the information contained in the image is streamlined to blacks and whites. Since our focus here is to find lanes, it can be observed that no information is lost in this conversion.

#### 2. Gaussian blur

There can be a lot of noise in the image even after grayscale conversion. Gaussian blur removes such noises and makes it easy for the upcoming steps.

[Result1]: ./doc/_1_blur_gray.jpg "Grayscale"

#### 3. Canny to find edges

Canny edge detection(as introduced in Lesson 1) is used next in the pipeline. This function is already available as a helper function from the OpenCV library and it takes the image, low threshold and high threshold(thresholds for intensity) as input. Edges are drawn on areas in the image where a change in intensity is detected(low-to-high or high-to-low). It must be noted that Canny function outputs the edges of every object on the image. Only part of the detected edges are lanes Canny's output has to be reduced.

[Result2]: ./doc/_2_edges.jpg "Edges"

#### 4. Region Of Interest(ROI) mask

Masking is nothing but setting the color intensity to a different value. The lanes generally reside in the same location on the image(more or less). Hence a polygon can be drawn on the image which can represent the ROI. Here a quadrilateral is selected and all pixels inside it are set to the masking color value.

This is accomplished with the helper functions,
- cv2.fillPoly() - Fills a polygon defined by vertices with the intensity value.
- cv2.bitwise_and() - Does a intersection of images.

Here a second polygon is also selected to remove certain unwanted lane participants. This is a triangle defined with its vertices. But the purpose of this ROI is to mask-out edge information(on contrary to above mentioned quadrilateral mask). It was noted during testing that there may be bright patches on the road(possibly spillings of lane paints) between the lanes. The presence of these will affect the lane fitting and hence these were removed.

[Result3]: ./doc/_3_masked_edges.jpg "Masked edges"

#### 5. Hough line segments

The output of ROI masks will still be edges(as output from Canny) but, constrained to a specific area defined by the polygons. Lines can be drawn on the edges with the help of helper function cv2.HoughLinesP(). Mainly based on the parameters minimum length of line(pixel), minimum gap(pixel) between the lines and minimum number of votes on a (\rho, \theta) grid the properties of the output line segments are defined.

#### Description of parameters used

- rho = 1
	- The distance resolution of the Hough grid is set to 1 pixel in order to capture lines more accurately.
- theta = np.pi / 180
	- Angular resolution is set to one degree for the same reason as for rho.
- threshold = 5
	- Minimum number of votes for intersection set to 'five' to cpature all intersecting lines.
- min_line_length = 5
	- Minimum line length set to five pixels to detect lines in the FoV of the camera.
- max_line_gap = 5
	- Maximum gap between the lines also set to five pixels to detect most of the lines.
- line_image = np.copy(image) * 0
	- This is the blank image where the lines are drawn on.

[Result4]: ./doc/_4_lines_edges.jpg "Hough lines"

#### 6. Pre-process line segments

For the considered parameters of cv2.HoughLinesP above, the output will be a lot of lines spanning in all directions. Hence a pre-processing of these lines in necessary where, some non-lane participants will be removed.

It can be inferred from the camera image that the lane lines are not straight in the perspective of the camera.If the start and end points of the lines are closer, then it might mean that the lines are vertical or horizontal. These lines are bad candidates for linear fitting. Hence these are filtered away.

A threshold of 2 pixel is set. If the difference in x and y candidates of the start and end point of the lines are below the set threshold, then they are filtered away in this step.

#### 7. Classify lane lines

The available output from the above step is a collection of lines. They don't yet point to the left and right lanes separately. This step performs the classification of the available lines to left or right lane.

This classification is based on 2 features of the lines,
- Slope - The left lane has a negative slope and the right lane has a positive slope.
- Position in ROI - If the lines belong to the left side of the quadrilateral then they mostly are candidates of left lane and vice-versa.

Each of the line segments are evaluated for their slope and their start and end points belonging to a side of the quadrilateral. Here the quadrilateral is split in to equal halves. The lines with a negative slope and belonging to the left side of the quadrilateral are candidates for left lane. The vice-versa is true for the right lanes.

#### 8. Filter short lines in left and right lane candidates

After the candidates are classified to left and right lane, it was found that we can't proceed to the linear fitting method. There may be a number of short lines besides the lanes(with approximately the same slope). Since in this method a relatively short threshold for min_line_length is set for cv2.HoughLinesP(), these lines will be picked up and will be possible candidates for the linear fitting.

The presence of these lines may be mainly because of spilled over lane paint, old temporary lane lines that are not completely wiped off or many more. During testing it was noted that these short lines always occur on the side beyond the shoulder of the road but still can qualify as lane candidates. In any case these short lines must be filtered away so that our parameters for linear fit is more trustable.

\note There can also be possiblilities of such short lines occuring where there are no lane shoulders. But in this pipeline, the presence of such lines are only considered and filtered on the side with lane shoulder. This can be mentioned as one of the shortcoming of the pipeline.

#### Method

The lane shoulder may exist on the left or right side of the road. Firstly, the shoulder's presence is detected by averaging the length of the line segments. This average(set as 100 pixels) is at least twice as much as the average on the side with dotted lines. So setting a high threshold on the line length to be detected can confidently detect the sides with lane shoulder.

Once the lane shoulder side is found, those lines with length much smaller than the average(threshold set as 15 pixels) can be just removed from the lane candidates.

#### 9. Linear line fitting

The available left and right lane candidates have been tuned and filtered in all the methods stated above. Now we can go ahead and model a linear fit on the available candaidates.

The method np.polyfit() is used for estimating the parameters of a first order polynomial. The estimated linear fit(Y = mX + b) parameters now are the slope and y-intercept.

#### 10. Find extension of lanes

Now we have a model of the left and right lane detected inside the ROI we have set. It can be the case the the lanes extend for a longer distance. The existance of this extension to lanes is evaluated by defining a triangle ROI on top of the quadrilateral ROI(Section 4).

#### Method

This method consists of 3 steps,
- Defining a triangle ROI mask to detect edges
- Drawing Hough lines
- Classifying them as candidates to left or right lane
- Evaluating the extension

#### Defining a triangle ROI

The same method mentioned in Section 4 is used here except for a different polygon, a triangle in this case. This triangle is drawn with its base overlaying on top of the quadrialteral used in ROI mask 1, but extended 15 pixels more in the left and right direction. The height of this triangle is set to 30 pixels. We fill this triangle with a cv2.fillPoly() method and do an intersection with Canny edges using cv2.bitwise_and(). The output of this procedure will be edges.

[Result5]: ./doc/_6_extended_edges.jpg "Extended edges"

#### Draw Hough lines

We draw Hough lines on the outcome of the section above with cv2.HoughLinesP() using almost the same parameters used in Section 4 except for,

- max_line_gap = 2
	- A small gap paremeter is selected as edges are more closer together at farther distance.

#### Classify Hough lines

Next we classify candidates for left and right lanes with the use of its slope. If the computed slope of the line is almost equal to the slope of the left lane(computed from Section 9), with a small difference threshold of 0.01, then the extension line segment is classified as left lane extension candidate. The vice-versa holds for the right lane.

#### Evaluating extension

After classification, a first order polynomial is modelled using numpy.polyfit(). If the slope of this model is very close to the slope of any of the lanes, then those extended line segments can be used for extending the lanes.

[Result6]: ./doc/_7_filtered_lines_edges.jpg "Filtered lines"

Extending is done by averaging all the y-positions of the extended line segment. This method introduces some noise on the extension and hence only half the average of y-positions is used.

Using this y-position and the computed slope from Section 9, the x-positions of the lanes are calculated.

#### 11. Plot them

A blank copy of the image is made to write the extended lanes on. Then this image with lane lines are weighted against the input image itself using the method cv2.addWeighted().

If you'd like to include images to show how the pipeline works, here is how to include an image: 

[Result7]: ./doc/_8_lanes_full.jpg "Full lane"


### 2. Potential shortcomings with current pipeline

Some shortcomings of the current pipeline are stated,

1. This method lacks robustness of detectiong lanes other than straight lanes. Shadows and different color of the roads might also affect robustness.

2. There is not tracking or any feedback loop for correcting the estimated parameters.

3. Filtering of unwanted short lines(mentioned in Section 8) can only be performed if lane shoulders are detected.


### 3. Possible improvements to pipeline

Seated below are some of the potential improvements of the pipeline,

1. If the Region of Interest masking is made dynamic, the robustness can be improved.

2. Use of methods other than Canny() and HoughLinesP() also can improve robustness.


### 4. Other suggestions for the course

It would be very handy if a video for setting up different development environments were available.


### 5. Satisfaction of project criterias

Lane finding pipeline,

- Criteria 1 is satisfied by the submitted interactive python file P1.ipynb. Annotated videos are also uploaded for reference.

- Criteria 2 is satisfied by Reflection Sections 1 - 9.

- Criteria 3 is satisfied mainly by Reflection Section 10.

- Reflection criteria is satisfied by everything submitted.
