# raspberry-pi-movement-detection

This script detects movement based on the concept of background subtraction of new frames from the webcam live feed with a base frame(That does not contain the foreign object). If movement is detected, the script will start recording x seconds of video from the live feed. As an alert, the script will then send a direct message to a Discord(A messaging application) channel along with the recorded video. That is basically it :D, the concept is simple enough and is able to fufill my requirement.

Downside of this script however is that the bounding box is not accurate as compared to object detection via deep learning apporach. It is also quite susceptible to noises caused by shadows or change in lighting(Partially resolve by performing erosion and setting a minimum connected components size)

Sample output video snippet:
![image](https://user-images.githubusercontent.com/43441027/119250512-24832480-bbd3-11eb-9107-7aa9a9500c3d.png)

Sample discord alert message:
![Screenshot 2021-05-22 171420](https://user-images.githubusercontent.com/43441027/119250539-47153d80-bbd3-11eb-9d1f-9d5b13152eee.jpg)



