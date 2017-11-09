from moviepy.editor import VideoFileClip
from process_image import process_image

white_output = 'solidYellowLeft.mp4'
#solidWhiteRight.mp4
#solidYellowLeft.mp4
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds

#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(1.95,2.1)
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
