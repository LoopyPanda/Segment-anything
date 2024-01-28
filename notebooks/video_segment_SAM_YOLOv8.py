# Import necessary packages
import os  # os will handle file operations
from io import BytesIO  # Dealing with binary data

import PIL  # It will provide functionalities to work with Images
from PIL import Image

HOME = os.getcwd()  # Get current working directory
print(HOME)

import cv2  # It will use OpenCV library and provide image and video processing functionalities
import time  # Can calculate Program running time
import numpy as np
import uuid  # It will generate universally unique identifiers.
import tempfile  # It will help to create temporary file

import torch
print("Torch version is: ", torch.__version__)  # to check current version
import torchvision
print("TorchVision version is: ", torchvision.__version__)
import torchaudio
print("TorchAudio version is: ", torchaudio.__version__)

# Load YOLOv8
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# Instantiate YOLOv8 model
model = YOLO(f'{HOME}/yolov8n.pt')  # Here we are using normal YOLO version 8 for detection task

colors = np.random.randint(0, 256, size=(len(model.names), 3))  # Generate random colors for visualization

print(model.names)  # Print the class name associated with the model

'''
- We came to know about the range of class_ids after running this code for a specific class_id,
namely we chose [4] : airplane. 
- Once we knew the range we opted to use all of them so that our model can detect any object 
from any input video.
'''
chosen_class_ids = list(range(80))


### SAM dependencies
# It will construct the full path to the checkpoint file by joining the directory and file

CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

"""
After accessing the 'sam_vit_h_4b8939' model now we will import necessary modules 
and start the SAM model.
"""

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Start SAM model
DEVICE = torch.device('cpu')  # If you are using GPU please change 'cpu' to 'CUDA'
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)  # It will predict mask from sam_model_registry "vit_h" where h stand for huge model

'''
- Labelbox’s Python SDK provides services in data labeling, and data management,
while supporting varied data types including text, images and videos.
- It's also gives you easy methods to create projects and datasets,
and upload labels to masks in the video.
- Because of these services we can make our process faster.
'''
import labelbox as lb
import labelbox.types as lb_types

# Create a Labelbox API key for your account to create a dataset by following the instructions here:
# https://docs.labelbox.com/reference/create-api-key
# Client key is a part of API/SDK which is a unique identifier that authenticates our requests.
# Then, fill it in here API_KEY field

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHBuNms1cDkwcWIzMDcxbDZlNGgyaTB5Iiwib3JnYW5pemF0aW9uSWQiOiJjbHBuNms1b3EwcWIyMDcxbGJyNzVmYjB4IiwiYXBpS2V5SWQiOiJjbHBuNnFtdW4wZDk2MDcwc2hqMTgzanVoIiwic2VjcmV0IjoiYjE2NGRiYzI3ZDgzMmU2NTM5N2EwYmNhYmVjYTNkYTkiLCJpYXQiOjE3MDE0NjkwNDYsImV4cCI6MjMzMjYyMTA0Nn0.1KYmpegeM1uw82PROAOh36R6iKuO8hDuGPr0S8-xBTM"
client = lb.Client(API_KEY)

'''
Now, we have imported and installed and added all the required libraries and models
let us define some functions which will help to reduce complexity.
'''

# Helper Functions

'''
get_color function take RGB color where color[0]= Red, Color[1]= Green, color[2]= Blue
and that converts into integers in the range of 0 to 255.
'''
def get_color(color):
  return (int(color[0]), int(color[1]), int(color[2]))

'''
get_video_dimensions function returns the height and width of the video frame using OpenCV
'''
def get_video_dimensions(input_cap):
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  return height, width

'''
get_output_video_writer function gives an output video and makes sure it has same dimensions as input video
'''
def get_output_video_writer(input_cap, output_path):
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Define the output video file
  output_codec = cv2.VideoWriter_fourcc(*"mp4v")  # Video format is mp4
  output_video = cv2.VideoWriter(output_path, output_codec, fps, (width, height))

  return output_video

'''
visualize_detection function will visualize bounding boxes, confidence scores (probabilities)
and classes around object detection through YOLOv8. 
'''
# Visualize a video frame with bounding boxes, classes and confidence scores
def visualize_detections(frame, boxes, conf_thresholds, class_ids):
    frame_copy = np.copy(frame)
    for idx in range(len(boxes)):
        class_id = int(class_ids[idx])
        conf = float(conf_thresholds[idx])
        x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
        if class_id in chosen_class_ids:
            color = colors[class_id]   # will give color to bounding box
            label = f"{model.names[class_id]}: {conf:.2f}"  # will give object name and confidence score
        else:
            # If the class is not in chosen_class_ids, set it to 'unknown'
            color = (0, 0, 0)  # Black color for 'unknown' class
            label = f"Unknown: {conf:.2f}"
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), get_color(color), 2)
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_color(color), 2)
    return frame_copy
# "unknown" label would probability not be used because we also need to make some changes.

'''
add_color_to_mask function will add color to mask which come frame SAM model
'''
def add_color_to_mask(mask, color):
  next_mask = mask.astype(np.uint8)
  next_mask = np.expand_dims(next_mask, 0).repeat(3, axis=0)
  next_mask = np.moveaxis(next_mask, 0, -1)
  return next_mask * color

'''
merge_masks_colored function merges colored masks (coming from SAM) from class_ids and iterates through class_ids
and their masks and combining them into a colored mask. 
'''
def merge_masks_colored(masks, class_ids):
  filtered_class_ids = []
  filtered_masks = []
  for idx, cid in enumerate(class_ids):
    if int(cid) in chosen_class_ids:
      filtered_class_ids.append(cid)
      filtered_masks.append(masks[idx])

  merged_with_colors = add_color_to_mask(filtered_masks[0][0], get_color(colors[int(filtered_class_ids[0])])).astype(np.uint8)

  if len(filtered_masks) == 1:
    return merged_with_colors

  for i in range(1, len(filtered_masks)):
      if i < len(filtered_class_ids):
        curr_mask_with_colors = add_color_to_mask(filtered_masks[i][0], get_color(colors[int(filtered_class_ids[i])]))
        merged_with_colors = np.bitwise_or(merged_with_colors, curr_mask_with_colors)

  return merged_with_colors.astype(np.uint8)

'''
get_instance_uri function will convert a numpy array into data that labelbox understands.
This requires authentication to labelbox SDK where we use the aforementioned API key.
'''
def get_instance_uri(client, global_key, array):
    """ Reads a numpy array into a temp Labelbox data row to-be-uploaded to Labelbox
    Args:
        client        :   Required (lb.Client) - Labelbox Client object
        global_key    :   Required (str) - Data row global key
        array         :   Required (np.ndarray) - NumPy ndarray representation of an image
    Returns:
        Temp Labelbox data row to-be-uploaded to Labelbox as row data
    """
    image_as_pil = PIL.Image.fromarray(array)  # Convert PIL image to PNG file bytes
    image_as_bytes = BytesIO()
    image_as_pil.save(image_as_bytes, format='PNG' if array.ndim == 3 else 'mp4')  # modified with an if statement to include images also
    content_type = "image/jpeg" if array.ndim == 3 else "video/mp4"  # modified with an if statement to include images also
    file_extension = 'jpg' if array.ndim == 3 else 'mp4'  # modified with an if statement to include images also
    image_as_bytes = image_as_bytes.getvalue()
    # Convert PNG file bytes to a temporary Labelbox URL
    # URL includes all the encoded data content which is listed below.
    url = client.upload_data(
    content=image_as_bytes,
    filename=f"{global_key}.{file_extension}",  # modified as per file_extension
    content_type=content_type,
    sign=True
    )
    # Return the URL
    return url

'''
get_local_instance_uri function takes the array and converts into temporary file 
'''
def get_local_instance_uri(array):
    # Convert array to PIL image
    image_as_pil = PIL.Image.fromarray(array)

    with tempfile.NamedTemporaryFile(suffix='.png', dir="/content", delete=False) as temp_file:
      image_as_pil.save(temp_file)
      file_name = temp_file.name

    # Return the URL
    return file_name

"""
create_mask_frame function will create path for each frame 
"""
def create_mask_frame(frame_num, instance_uri):
  return lb_types.MaskFrame(index=frame_num, instance_uri=instance_uri)

"""
From a list of class_ids, create_mask_instances function will remove duplicat class_ids,
and return a list of mask instances
"""
def create_mask_instances(class_ids):
  instances = []
  for cid in list(set(class_ids)): # get unique class ids
    if int(cid) in chosen_class_ids:
      color = get_color(colors[int(cid)])
      name = model.names[int(cid)]
      instances.append(lb_types.MaskInstance(color_rgb=color, name=name))
  return instances

"""
create_video_mask_annotation function creates an annotation object specifically for masks
"""
def create_video_mask_annotation(frames, instance):
  return lb_types.VideoMaskAnnotation(
        frames=frames,
        instances=[instance]
    )

# Here we use uuid to make sure our global_key is unique everytime we run the program
"""
Although here we introduced unique identification.
This creates a uuid for each row_data as we are using Labelbox.
The uuid helps to identify different instances or projects.
"""

unique_identifier = str(uuid.uuid4())[:8]  # we take only 8 characters
# global_key = f"{os.path.basename('video/skateboarding')}_{unique_identifier}"
#global_key = f"{os.path.basename('video/aeroplane')}_{unique_identifier}"
global_key = f"{os.path.basename('video/Highway')}_{unique_identifier}"
#global_key = f"{os.path.basename('video/House_objects')}_{unique_identifier}"

# Creating a dictionary containing below key-value pairs
asset = {
    #"row_data": 'video/aeroplane.mp4',
    "row_data": 'video/Highway.mp4',
    # "row_data": 'video/skateboarding.mp4',
    #"row_data": 'video/House_objects.mp4',
    "global_key": global_key,
    "media_type": "VIDEO"
}

"""
We will use labelbox SDK to create a dataset,
and upload a video as row_data to the dataset 'yolo-sam-video-masks-dataset'
"""
dataset = client.create_dataset(name="yolo-sam-video-masks-dataset")
task = dataset.create_data_rows([asset])
task.wait_till_done()

print(f"Errors: {task.errors}")  # To ensure everything works properly
print(f"Failed data rows: {task.failed_data_rows}")  # To ensure there were no failed data rows

"""
As we noticed that running this on CPU takes a lot of time, we decided to time it.
We also keep an end timer at the end and print the time difference to note
how much time it takes to run the program.

"""

start_time = time.time()  # Starts timer from the moment of the start of object detection

# Run YOLOv8 and SAM per-frame

#cap = cv2.VideoCapture('video/aeroplane.mp4')
cap = cv2.VideoCapture('video/Highway.mp4')
#cap = cv2.VideoCapture('video/skateboarding.mp4')
#cap = cv2.VideoCapture('video/House_objects.mp4')

unique_class_ids = set()  # Using a set makes sure that class-ids are unique

"""
We will now process frames running them through YOLOv8 for object detection.
It then saves each frame as an image file and collects the detected class-ids.
"""

"""
As a first step, we will try to read each frame from the input video and store it as an image file.
We do this to get an overview of each input frame on which bounding boxes and masks would be applied.
Since it is for our viewing purpose, we restrict ourselves to just 10 input frames. 
"""
mask_frames = []  # Initializing a list for frames with mask

# Loop through the frames of the video
frame_num = 1
max_frames = 10  # Maximum input frames to be written as an image for testing purpose, to see video progress

# ret indicates if the frame was successfully read

while cap.isOpened() and frame_num <= max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Frames', frame)
    cv2.imwrite(f'images/frame_{frame_num}.jpg', frame)  # Save the image with the specified filename

    # Run frame through YOLOv8 and get class ids predicted
    detections = model.predict(frame, conf=0.7)
    for cid in detections[0].boxes.cls:
        unique_class_ids.add(int(cid))
    frame_num += 1

cap.release()  # Ensures that the video capture object is released after wards to free memory usage

unique_class_ids  # prints the unique class ids

# Run YOLOv8 and then SAM on each frame, and write visualization videos to disk

#cap = cv2.VideoCapture('video/aeroplane.mp4')
# cap = cv2.VideoCapture('video/skateboarding.mp4')
cap = cv2.VideoCapture('video/Highway.mp4')
#cap = cv2.VideoCapture('video/House_objects.mp4')


# output_video_boxes = get_output_video_writer(cap, "content/skateboarding_boxes.mp4")
# output_video_masks = get_output_video_writer(cap, "content/skateboarding_masks.mp4")
# output_video_boxes = get_output_video_writer(cap, "auto_content3/Highway_boxes.mp4")
# output_video_masks = get_output_video_writer(cap, "auto_content3/Highway_masks.mp4")
output_video = get_output_video_writer(cap, "auto_content4/Highway_output.mp4")# Creates an output video
# output_video = get_output_video_writer(cap, "auto_content5/House_objects.mp4")

"""
Starting from 1st frame, below 'while' loop processes frames from video, 
performs object detection using YOLOv8, generates masks using SAM model, 
then creates an output video which include detections, masks and bounding boxes.
"""

mask_frames = []

# Loop through the frames of the video
frame_num = 1

while cap.isOpened():
    if frame_num % 30 == 0 or frame_num == 1:  # choosing 30 since usually a video has fps=30
        print("Processing frames", frame_num, "-", frame_num + 29)
    # ret_boxes and ret_masks check if the frames are read.
    # If the frames were read, frame_boxes and frame_masks contain the actual frames as NumPy arrays.
    """
    We have modified the code to make sure that the frames are used in pairs as below.
    That is, by doing this, the detection of objects and generation of masks happen on the same time step.
    """
    ret_boxes, frame_boxes = cap.read()
    ret_masks, frame_masks = cap.read()

    # Break the loop if video ends
    if not ret_boxes or not ret_masks:
        break

    # Run frame through YOLOv8 to get detections
    detections = model.predict(frame_boxes, conf=0.7)  # frame is a numpy array

    """
    Below we visualize detections from video frames (from frame_boxes),
    and write them to an output video. These detections should come from frame_boxes and not from 'frame'.
    This is also a part where we update the code to make sure our output with masks and boxes 
    is running together with the input video.
    """

    frame_with_detections = visualize_detections(frame_boxes,
                                                 detections[0].boxes.cpu().xyxy,
                                                 detections[0].boxes.cpu().conf,
                                                 detections[0].boxes.cpu().cls)
    #output_video_boxes.write(frame_with_detections)
    output_video.write(frame_with_detections)

    # Run frame and detections through SAM to get masks and transforming bounding boxes

    transformed_boxes = mask_predictor.transform.apply_boxes_torch(detections[0].boxes.xyxy,list(get_video_dimensions(cap)))
    if len(transformed_boxes) == 0:
        print("No boxes found on frame", frame_num)
        #output_video_masks.write(frame)
        output_video.write(frame)
        frame_num += 1
        continue
    mask_predictor.set_image(frame_boxes)

    # Predict masks based on transformed boxes

    masks, scores, logits = mask_predictor.predict_torch(
        boxes=transformed_boxes,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )
    masks = np.array(masks.cpu())

    # If masks are found, then merge colored masks

    if masks is None or len(masks) == 0:
        print("No masks found on frame", frame_num)
        #output_video_masks.write(frame)
        output_video.write(frame)
        frame_num += 1
        continue
    merged_colored_mask = merge_masks_colored(masks, detections[0].boxes.cls)

    # Save images which contain both boxes and masks
    #cv2.imwrite(f'images/Highway_frames/combined_frame_{frame_num}.jpg', combined_frame)
    """
    Below we join detections with colored masks and save those images into a file for each frame.
    Then we join all those images together and write them into an output video.
    """
    # Output_video_masks.write(image_combined)
    """
    For fairness, we adjust the confidence score and keep this cut-off at 0.7.
    This makes sure that the detection happens only when the model detects 
    the object for a class-id with probability > 0.7.
    """
    image_with_automask = cv2.addWeighted(frame_with_detections, 0.7, merged_colored_mask, 0.7, 0)
    cv2.imwrite(f'images/Highway/image_with_automask{frame_num}.jpg', image_with_automask)
    #cv2.imwrite(f'images/House_objects/image_with_automask{frame_num}.jpg', image_with_automask)
    output_video.write(image_with_automask)

    # Create video mask annotation for upload to Labelbox
    instance_uri = get_instance_uri(client, global_key, merged_colored_mask)
    mask_frame = create_mask_frame(frame_num, instance_uri)
    mask_frames.append(mask_frame)

    frame_num += 1

    # For the purposes of the video, only look at the first 30 frames,
    # so that we get 1 seconds of video output
    if frame_num > 30:
      break


cap.release()  # release object from video capture to free the video file for other purposes
# output_video_boxes.release()  # release object containing boxes
# output_video_masks.release()  # release object containing masks
output_video.release()  # release object containing boxes and masks
cv2.destroyAllWindows()  # close any open OpenCV window for proper execution

# End timer
end_time = time.time()

# Check the total time spent on detection
total_time = end_time - start_time

print(f"Total time: {total_time} seconds")