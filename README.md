# object-detection-raspberry-pi


## 1) [v1.x] image_processing_obj_detection **(Run on Raspberry Pi)**<br />
This script detects movement based on the concept of background subtraction of new frames from the webcam live feed with a base frame(That does not contain the foreign object). If movement is detected, the script will start recording x seconds of video from the live feed. As an alert, the script will then send a direct message to a Discord(A messaging application) channel along with the recorded video. That is basically it :D, the concept is simple enough and is able to fufill my requirement.

Downside of this script however is that the bounding box is not accurate as compared to object detection via deep learning apporach. It is also quite susceptible to noises caused by shadows or change in lighting(Partially resolve by performing erosion and setting a minimum connected components size)

Sample output video snippet:<br />
![image](https://user-images.githubusercontent.com/43441027/119250512-24832480-bbd3-11eb-9107-7aa9a9500c3d.png)

Sample discord alert message:<br />
![Screenshot 2021-05-22 171420](https://user-images.githubusercontent.com/43441027/119250539-47153d80-bbd3-11eb-9d1f-9d5b13152eee.jpg)

<br />
<br />

## 2) [v1.x] opencv_dnn_deep_learning_obj_detection **(Run on Desktop)**<br />
It contains three mode which are **video**, **image**, and **webcam**. This script uses pre-trained YOLO(You only look once) deep learning models to perform object detection. It uses OpenCV's dnn module to load and run the model(Not configured to run on GPU). However, this approach is not optimized to run on raspberry pi and also its live webcam feed(not even YOLOv3-tiny) as it struggles to even obtain 1fps. For live feed, TensorFlow Lite framework is much more promising as tflite model are more optimized to run on lightweight devices such as raspberry pi.
