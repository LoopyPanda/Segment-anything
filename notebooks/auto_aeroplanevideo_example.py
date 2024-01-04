import os
from io import BytesIO

import PIL
import matplotlib.pyplot as plt
HOME = os.getcwd()
print(HOME)
import sys # sys module provides access to system-specific parameters and functions in Python.
import cv2
import time
import numpy as np
import uuid
import tempfile
from PIL import Image
#Load YOLOv8
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# Instantiate YOLOv8 model
model = YOLO(f'{HOME}/yolov8n.pt')
colors = np.random.randint(0, 256, size=(len(model.names), 3))

print(model.names)

# Specify which classes. The rest of classes will be filtered out.
# chosen_class_ids = [0] # person
# chosen_class_ids = [4] # aeroplane
chosen_class_ids = list(range(80)) # since Yolo dataset has 80 different objects for detection


# SAM dependency

CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

# Import packages
import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
import torchaudio
print(torchaudio.__version__)
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Start SAM model
DEVICE = torch.device('cpu')
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

#Labelbox’s Python SDK gives you easy methods to create ontologies, projects and datasets, and upload masks to a video.
import labelbox as lb
import labelbox.types as lb_types

# Create a Labelbox API key for your account by following the instructions here:
# https://docs.labelbox.com/reference/create-api-key
# Then, fill it in here
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHBuNms1cDkwcWIzMDcxbDZlNGgyaTB5Iiwib3JnYW5pemF0aW9uSWQiOiJjbHBuNms1b3EwcWIyMDcxbGJyNzVmYjB4IiwiYXBpS2V5SWQiOiJjbHBuNnFtdW4wZDk2MDcwc2hqMTgzanVoIiwic2VjcmV0IjoiYjE2NGRiYzI3ZDgzMmU2NTM5N2EwYmNhYmVjYTNkYTkiLCJpYXQiOjE3MDE0NjkwNDYsImV4cCI6MjMzMjYyMTA0Nn0.1KYmpegeM1uw82PROAOh36R6iKuO8hDuGPr0S8-xBTM"
client = lb.Client(API_KEY)

# Helper Function
# Cast color to ints
def get_color(color):
  return (int(color[0]), int(color[1]), int(color[2]))

# Get video dimensions
def get_video_dimensions(input_cap):
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  return height, width

# Get output video writer with same dimensions and fps as input video
def get_output_video_writer(input_cap, output_path):
  # Get the video's properties (width, height, FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Define the output video file
  output_codec = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
  output_video = cv2.VideoWriter(output_path, output_codec, fps, (width, height))

  return output_video

# Visualize a video frame with bounding boxes, classes and confidence scores
def visualize_detections(frame, boxes, conf_thresholds, class_ids):
    frame_copy = np.copy(frame)
    for idx in range(len(boxes)):
        class_id = int(class_ids[idx])
        conf = float(conf_thresholds[idx])
        x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
        color = colors[class_id]
        label = f"{model.names[class_id]}: {conf:.2f}"
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), get_color(color), 2)
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_color(color), 2)
    return frame_copy

def add_color_to_mask(mask, color):
  next_mask = mask.astype(np.uint8)
  next_mask = np.expand_dims(next_mask, 0).repeat(3, axis=0)
  next_mask = np.moveaxis(next_mask, 0, -1)
  return next_mask * color

# Merge masks into a single, multi-colored mask
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

def get_instance_uri(client, global_key, array):
    """ Reads a numpy array into a temp Labelbox data row to-be-uploaded to Labelbox
    Args:
        client        :   Required (lb.Client) - Labelbox Client object
        global_key    :   Required (str) - Data row global key
        array         :   Required (np.ndarray) - NumPy ndarray representation of an image
    Returns:
        Temp Labelbox data row to-be-uploaded to Labelbox as row data
    """
    # Convert array to PIL image

    image_as_pil = PIL.Image.fromarray(array)
    # Convert PIL image to PNG file bytes
    image_as_bytes = BytesIO()
    image_as_pil.save(image_as_bytes, format='PNG' if array.ndim == 3 else 'mp4')
    content_type = "image/jpeg" if array.ndim == 3 else "video/mp4"
    file_extension = 'jpg' if array.ndim == 3 else 'mp4'
    image_as_bytes = image_as_bytes.getvalue()
    # Convert PNG file bytes to a temporary Labelbox URL
    url = client.upload_data(
    content=image_as_bytes,
    filename=f"{global_key}.{file_extension}",
    content_type=content_type,
    sign=True
    )
    # Return the URL
    return url

def get_local_instance_uri(array):
    # Convert array to PIL image
    image_as_pil = PIL.Image.fromarray(array)

    with tempfile.NamedTemporaryFile(suffix='.png', dir="/content", delete=False) as temp_file:
      image_as_pil.save(temp_file)
      file_name = temp_file.name

    # Return the URL
    return file_name

def create_mask_frame(frame_num, instance_uri):
  return lb_types.MaskFrame(index=frame_num, instance_uri=instance_uri)

def create_mask_instances(class_ids):
  instances = []
  for cid in list(set(class_ids)): # get unique class ids
    if int(cid) in chosen_class_ids:
      color = get_color(colors[int(cid)])
      name = model.names[int(cid)]
      instances.append(lb_types.MaskInstance(color_rgb=color, name=name))
  return instances

def create_video_mask_annotation(frames, instance):
  return lb_types.VideoMaskAnnotation(
        frames=frames,
        instances=[instance]
    )

#Labelbox SetUp
# Everytime it creates a new dataset to avoid duplication of data
# read more here: https://docs.labelbox.com/reference/data-row-global-keys

unique_identifier = str(uuid.uuid4())[:8]
# global_key = f"{os.path.basename('video/skateboarding')}_{unique_identifier}"
#global_key = f"{os.path.basename('video/aeroplane')}_{unique_identifier}"
global_key = f"{os.path.basename('video/Highway')}_{unique_identifier}"

asset = {
    #"row_data": 'video/aeroplane.mp4',
    "row_data": 'video/Highway.mp4',
    # "row_data": 'video/skateboarding.mp4',
    "global_key": global_key,
    "media_type": "VIDEO"
}

dataset = client.create_dataset(name="yolo-sam-video-masks-dataset")
task = dataset.create_data_rows([asset])
task.wait_till_done()

print(f"Errors: {task.errors}")
print(f"Failed data rows: {task.failed_data_rows}")

# Start timer
start_time = time.time()

#Run YOLOv8 and SAM per-frame
#cap = cv2.VideoCapture('video/aeroplane.mp4')
cap = cv2.VideoCapture('video/Highway.mp4')
# cap = cv2.VideoCapture('video/skateboarding.mp4')

# This will contain the resulting mask predictions for upload to Labelbox
unique_class_ids = set()

mask_frames = []

# Loop through the frames of the video
frame_num = 1
max_frames = 10

while cap.isOpened() and frame_num <= max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Frames', frame)
    cv2.imwrite(f'images/frame_{frame_num}.jpg', frame)  # Save the image with the specified filename

    # Run frame through YOLOv8 and get class ids predicted
    detections = model.predict(frame, conf=0.7)  # frame is a numpy array
    for cid in detections[0].boxes.cls:
        unique_class_ids.add(int(cid))
    frame_num += 1

cap.release()

unique_class_ids

# Run YOLOv8 and then SAM on each frame, and write visualization videos to disk
# You can download /content/skateboarding_boxes.mp4 and /content/skateboarding_masks.mp4
# to visualize the results
#cap = cv2.VideoCapture('video/aeroplane.mp4')
# cap = cv2.VideoCapture('video/skateboarding.mp4')
cap = cv2.VideoCapture('video/Highway.mp4')

# output_video_boxes = get_output_video_writer(cap, "content/skateboarding_boxes.mp4")
# output_video_masks = get_output_video_writer(cap, "content/skateboarding_masks.mp4")
output_video_boxes = get_output_video_writer(cap, "auto_content3/Highway_boxes.mp4")
output_video_masks = get_output_video_writer(cap, "auto_content3/Highway_masks.mp4")
mask_frames = []

# Loop through the frames of the video
frame_num = 1
while cap.isOpened():
    if frame_num % 30 == 0 or frame_num == 1:
        print("Processing frames", frame_num, "-", frame_num + 29)
    ret, frame = cap.read()
    if not ret:
        break
    # print("Maximum frames are:", mask_frames)

    # Run frame through YOLOv8 to get detections
    detections = model.predict(frame, conf=0.7)  # frame is a numpy array

    # Write detections to output video
    frame_with_detections = visualize_detections(frame,
                                                 detections[0].boxes.cpu().xyxy,
                                                 detections[0].boxes.cpu().conf,
                                                 detections[0].boxes.cpu().cls)
    output_video_boxes.write(frame_with_detections)

    # Run frame and detections through SAM to get masks
    transformed_boxes = mask_predictor.transform.apply_boxes_torch(detections[0].boxes.xyxy,list(get_video_dimensions(cap)))
    if len(transformed_boxes) == 0:
        print("No boxes found on frame", frame_num)
        output_video_masks.write(frame)
        frame_num += 1
        continue
    mask_predictor.set_image(frame)

    masks, scores, logits = mask_predictor.predict_torch(
        boxes=transformed_boxes,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )
    masks = np.array(masks.cpu())
    if masks is None or len(masks) == 0:
        print("No masks found on frame", frame_num)
        output_video_masks.write(frame)
        frame_num += 1
        continue
    merged_colored_mask = merge_masks_colored(masks, detections[0].boxes.cls)

    # Write masks to output video
    image_combined = cv2.addWeighted(frame, 0.7, merged_colored_mask, 0.7, 0)
    output_video_masks.write(image_combined)

    # Create video mask annotation for upload to Labelbox
    instance_uri = get_instance_uri(client, global_key, merged_colored_mask)
    mask_frame = create_mask_frame(frame_num, instance_uri)
    mask_frames.append(mask_frame)

    frame_num += 1

    # For the purposes of the airplane video, only look at the first 30 frames
    if frame_num > 60:
       break

cap.release()
output_video_boxes.release()
output_video_masks.release()
cv2.destroyAllWindows()

# End timer
end_time = time.time()

# Check the total time spent on detection
total_time = end_time - start_time

print(f"Total time: {total_time} seconds")