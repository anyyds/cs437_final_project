# cs437_final_project

To run the project, first install the dependencies:

pip install opencv-python numpy torch ultralytics

Note: if there are warnings or errors about compatibility, it may be due to incompatible versions of the required libraries. My machine uses the following versions:
opencv-python 4.11.0.86
torch 2.7.0
ultralytics 8.3.135
numpy 2.2.5
Python 3.12.10

Then, move the input video to the same directory as the project. Run python start_detector.py to start the detection process. 

You will be asked to enter a reference object length and select 2 points for the reference object. Select 2 points in the video so that the scale is defined. 

Then, you will be asked to select 4 perspective transformation points. Selet the points (in this order: top left, top right, bottom right, bottom left) that contain the road where vehicles are to be detected. 
