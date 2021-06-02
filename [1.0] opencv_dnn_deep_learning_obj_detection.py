import numpy as np
import cv2
import time
import argparse

window_name = "Object Detection using OpenCV DNN Module"

parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=int, help=".weights file for the model")
parser.add_argument("--config", type=str, help=".cfg file for the model")
parser.add_argument("--className", type=str,
                    help="list of classes for predicted by model")
parser.add_argument("--mode", type=int,
                    help="Mode 1 for video, 2 for webcam, and 3 for image")
parser.add_argument("--vidFileName", type=str,
                    help="Filename for the video file to be tested")
parser.add_argument("--imgFileName", type=str,
                    help="Filename for the image file to be tested")

args = parser.parse_args()

model_weight_file = args.weight if args.weight != None else 'sample_yolo_model_and_weight/yolov3-tiny.weights'
model_config_file = args.config if args.config != None else 'sample_yolo_model_and_weight/yolov3-tiny.cfg'
class_file = args.className if args.className != None else 'sample_yolo_model_and_weight/coco.names'
MODE = args.mode if args.mode != None else 1  # default video mode
vid_file_name = args.vidFileName if args.vidFileName != None else 'sample_video_and_image/tokyo-shibuya.mp4'
img_file_name = args.imgFileName if args.imgFileName != None else 'sample_video_and_image/First-time-in-Tokyo.jpg'

# For yolov3
# input shape is 416x416
# Order needs to be in R -> G -> B order
# Makes prediction at 3 scale (Makes it more robust to detect smaller object)
# Bounding box is x,y,w,h format (x,y refers to center of object of interest)
net = cv2.dnn.readNet(model_weight_file, model_config_file)
classes = []

with open(class_file, 'r') as f:
    classes = f.read().splitlines()


# Mode 1 for video, 2 for webcam, and 3 for image
if (MODE == 1):
    cap = cv2.VideoCapture(vid_file_name)
elif (MODE == 2):
    cap = cv2.VideoCapture(0)
elif (MODE == 3):
    img = cv2.imread(img_file_name)  # for image


# https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0


while True:

    # Initialization placed in loop to reset after every frame
    b_boxes = []
    confidences = []
    class_label_ids = []
    THRESHOLD = 0.6

    if(MODE == 1 or MODE == 2):
        _, img = cap.read()

    height, width, _ = img.shape

    # single function to perform the necessary pre processing ;D
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    # to get names of layers with unconnected output
    output_layers_names = net.getUnconnectedOutLayersNames()
    # run forward pass to compute output of layer with name "output name"
    layersOutputs = net.forward(output_layers_names)

    for output in layersOutputs:  # extract info from the layers output, 3 different
        # output contains all the bounding boxes for each output layers
        # 3 layersOutputs from the prediction made on the 3 different scale of image (Because of YoloV3 architecture to make the alog more robust)
        # extract info for each output, contains: bounding box pos(4 para),box confidence level(1 para), x classes probabilities(x para)
        for detection in output:

            # print(len(detection))
            scores = detection[5:]  # the probabilities values of ALL classes
            class_label_id = np.argmax(scores)  # return index
            # the highest probability value
            confidence = scores[class_label_id]
            if confidence > THRESHOLD:
                # multiply to scale back to original because it was rescaled
                b_center_x = int(detection[0] * width)
                b_center_y = int(detection[1] * height)
                b_w = int(detection[2] * width)
                b_h = int(detection[3] * height)

                # get upper left coordinates of the bounding boxes
                b_x = int(b_center_x - b_w/2)
                b_y = int(b_center_y - b_h/2)

                # append the boxes position info
                b_boxes.append([b_x, b_y, b_w, b_h])
                confidences.append(float(confidence))
                class_label_ids.append(class_label_id)

    # Non maximum suppression(NMS) algorithm to get rid of overlapping boxes for a single object
    # Pseudo code of NMS
    # 1. Select box with highest confidence
    # 2. Compare intersection over union(IOU)
    # 3. Remove bounding boxes with overlap (More than x%)
    # 4. Move to next highest confidence box
    # 5. Repeat 2-4

    #print("Number of bounding boxes before NMS: ",len(b_boxes))
    indexes = cv2.dnn.NMSBoxes(b_boxes, confidences, THRESHOLD, 0.4)
    #print("Number of bounding boxes after NMS: ",len(indexes))
    # print(np.sort(indexes.flatten()))

    # Display the result of the detection
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(b_boxes), 3))  # 3 channel

    # https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
    new_frame_time = time.time()
    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = int(1/(new_frame_time-prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(img, "FPS", (20, 50), font, 2, (0, 255, 0), 2)
    cv2.putText(img, str(fps), (100, 50), font, 2, (0, 255, 0), 2)

    if len(indexes) > 0:
        for i in indexes.flatten():  # loop through indexes that has not been removed by NMS only

            b_x, b_y, b_w, b_h = b_boxes[i]
            label = str(classes[class_label_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (b_x, b_y), (b_x+b_w, b_y+b_h),
                          color, 5)  # 2 for thickness
            cv2.putText(img, label + " " + confidence,
                        (b_x, b_y-10), font, 2, (255, 255, 255), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)
    cv2.imshow(window_name, img)

    if(MODE == 1 or MODE == 2):
        key = cv2.waitKey(1)
        if key == 27:  # esc key to exit
            break
    elif(MODE == 3):
        key = cv2.waitKey(0)
        if key == 27:  # esc key to exit
            break
        break


if (MODE == 1 or MODE == 2):
    cap.release()  # For video
cv2.destroyAllWindows()
