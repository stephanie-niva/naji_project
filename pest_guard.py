# import paho.mqtt.client as mqtt
# import cv2
# import numpy as np
# import time
# from picamera2 import Picamera2
# import os
# import json

# THINGSBOARD_HOST = 'YOUR_THINGSBOARD_IP_OR_HOSTNAME'
# ACCESS_TOKEN = 'RASPBERRY_PI_DEMO_TOKEN'

# # Initialize Picamera2
# camera = Picamera2()
# camera.configure(camera.create_still_configuration())

# # Path to your dataset folder
# dataset_path = "/home/naji/naji_project/Miner leaves/"

# # Helper Functions
# def preprocess_image(image):
#     """Preprocess image by converting to grayscale and applying Gaussian blur."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def extract_disease_patches(image):
#     """Extract disease patches based on color and hue."""
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # Define the hue range for diseased patches (adjust these values if needed)
#     lower_hue = np.array([0, 100, 100])  # Lower hue threshold
#     upper_hue = np.array([20, 255, 255])  # Upper hue threshold
#     # Create a mask for diseased areas
#     disease_mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
#     # Extract diseased regions
#     disease_patches = cv2.bitwise_and(image, image, mask=disease_mask)
#     return disease_patches, disease_mask

# def extract_features_from_disease_patches(image):
#     """Extract color features from diseased patches."""
#     disease_patches, mask = extract_disease_patches(image)
#     # Calculate color histogram for the diseased patches
#     if cv2.countNonZero(mask) == 0:
#         return None  # No diseased patches detected
#     hist = cv2.calcHist([disease_patches], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist = cv2.normalize(hist, hist).flatten()
#     return hist

# def match_features(hist1, hist2):
#     """Match color histograms using correlation."""
#     return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# def find_best_match(captured_image):
#     """Compare the captured image with each image in the dataset and find the best match."""
#     best_confidence = 0
#     best_match_filename = None
#     best_match_image = None
#     # Extract features from the captured image
#     hist1 = extract_features_from_disease_patches(captured_image)
#     if hist1 is None:
#         return None, 0, None  # No diseased patches detected
#     for dataset_image_filename in os.listdir(dataset_path):
#         dataset_image_path = os.path.join(dataset_path, dataset_image_filename)
#         dataset_image = cv2.imread(dataset_image_path)
#         if dataset_image is None:
#             continue
#         hist2 = extract_features_from_disease_patches(dataset_image)
#         if hist2 is None:
#             continue
#         # Match the histograms
#         confidence = match_features(hist1, hist2)
#         # Keep track of the highest confidence match
#         if confidence > best_confidence:
#             best_confidence = confidence
#             best_match_filename = dataset_image_filename
#             best_match_image = dataset_image
#     return best_match_filename, best_confidence, best_match_image

# def capture_and_process(threshold=0.8):
#     """Capture an image, process it for leaf disease detection, and send telemetry data to ThingsBoard."""
#     # Capture image
#     filename = "leaf_image.jpg"
#     camera.start()
#     time.sleep(2)  # Wait for camera to adjust
#     camera.capture_file(filename)
#     camera.stop()

#     # Read captured image
#     captured_image = cv2.imread(filename)
#     # Check for diseased patches and match
#     best_match, confidence, match_image = find_best_match(captured_image)

#     # Decide based on confidence
#     if confidence > threshold:
#         print(f"Leaf is affected! Match found with: {best_match}, Confidence: {confidence}")
#         status = "affected"
#     else:
#         print(f"Leaf is healthy or not matching any affected dataset image. Confidence: {confidence}")
#         status = "healthy"

#     # Send telemetry data to ThingsBoard
#     telemetry_data = json.dumps({"leaf_status": status, "confidence": confidence})
#     client.publish('v1/devices/me/telemetry', telemetry_data, 1)

#     # Discard the image after processing
#     print("Processing done. Discarding image...")
#     return True

# # MQTT callbacks
# def on_connect(client, userdata, flags, rc):
#     print('Connected with result code ' + str(rc))
#     # Subscribing to receive RPC requests
#     client.subscribe('v1/devices/me/rpc/request/+')

# def on_message(client, userdata, msg):
#     print('Topic: ' + msg.topic + '\nMessage: ' + str(msg.payload))

# # Initialize MQTT client
# client = mqtt.Client()
# client.on_connect = on_connect
# client.on_message = on_message
# client.username_pw_set(ACCESS_TOKEN)
# client.connect(THINGSBOARD_HOST, 1883, 60)

# # Main loop
# try:
#     client.loop_start()
#     while True:
#         capture_and_process()
#         time.sleep(300)  # Wait for 5 minutes before capturing the next image
# except KeyboardInterrupt:
#     print("Exiting...")
#     client.loop_stop()
#     client.disconnect()

import paho.mqtt.client as mqtt
import cv2
import numpy as np
import time
from datetime import datetime
from picamera2 import Picamera2
import os
import json


THINGSBOARD_HOST = 'thingsboard.cloud'
ACCESS_TOKEN = 'A6IoE3wPZFZs05YQsHyJ'
# Initialize Picamera2
camera = Picamera2()
camera.configure(camera.create_still_configuration())
# Path to your dataset folder
dataset_path = "/path/to/your/affected_leaves_dataset/"  # Update this path
# Helper Functions
def preprocess_image(image):
    """Preprocess image by converting to grayscale and applying Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
def extract_disease_patches(image):
    """Extract disease patches based on color and hue."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the hue range for diseased patches (you may need to adjust these values)
    lower_hue = np.array([0, 100, 100])  # Lower hue threshold (e.g., brownish or yellowish)
    upper_hue = np.array([20, 255, 255])  # Upper hue threshold
    # Create a mask for diseased areas
    disease_mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
    # Use the mask to extract diseased regions
    disease_patches = cv2.bitwise_and(image, image, mask=disease_mask)
    return disease_patches, disease_mask
def extract_features_from_disease_patches(image):
    """Extract color features from diseased patches."""
    disease_patches, mask = extract_disease_patches(image)
    # Calculate color histogram for the diseased patches
    if cv2.countNonZero(mask) == 0:
        return None  # No diseased patches detected
    hist = cv2.calcHist([disease_patches], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
def match_features(hist1, hist2):
    """Match color histograms using correlation."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
def find_best_match(captured_image):
    """Compare the captured image with each image in the dataset and find the best match."""
    best_confidence = 0
    best_match_filename = None
    best_match_image = None
    # Extract features from the captured image
    hist1 = extract_features_from_disease_patches(captured_image)
    if hist1 is None:
        return None, 0, None  # No diseased patches detected
    for dataset_image_filename in os.listdir(dataset_path):
        dataset_image_path = os.path.join(dataset_path, dataset_image_filename)
        dataset_image = cv2.imread(dataset_image_path)
        if dataset_image is None:
            continue
        hist2 = extract_features_from_disease_patches(dataset_image)
        if hist2 is None:
            continue
        # Match the histograms
        confidence = match_features(hist1, hist2)
        # Keep track of the highest confidence match
        if confidence > best_confidence:
            best_confidence = confidence
            best_match_filename = dataset_image_filename
            best_match_image = dataset_image
    return best_match_filename, best_confidence, best_match_image
def visualize_matches(captured_image, match_image):
    """Visualize the matching keypoints between the two images."""
    orb = cv2.ORB_create()
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(captured_image, None)
    kp2, des2 = orb.detectAndCompute(match_image, None)
    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw matches
    matched_image = cv2.drawMatches(captured_image, kp1, match_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Display the matched image
    cv2.imshow('Matches', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def capture_and_process(threshold=0.8):
    filename = "leaf_image.jpg"
    camera.start()
    time.sleep(2)  # Wait for camera to adjust
    camera.capture_file(filename)
    camera.stop()
    captured_image = cv2.imread(filename)
    best_match, confidence, match_image = find_best_match(captured_image)
    if confidence > threshold:
        print(f"Leaf is affected! Match found with: {best_match}, Confidence: {confidence}")
        status = "affected"
    else:
        print(f"Leaf is healthy or not matching any affected dataset image. Confidence: {confidence}")
        status = "healthy"
    # Get the current date
    detection_date = datetime.now().strftime('%Y-%m-%d')
    # Include detection date in the telemetry data
    telemetry_data = json.dumps({
        "leaf_status": status,
        "confidence_score": confidence,
        "detection_date": detection_date
    })
    client.publish('v1/devices/me/telemetry', telemetry_data, 1)
    print("Processing done. Discarding image...")
    return True
# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print('Connected with result code ' + str(rc))
    client.subscribe('v1/devices/me/rpc/request/+')
def on_message(client, userdata, msg):
    print('Topic: ' + msg.topic + '\nMessage: ' + str(msg.payload))
# Initialize MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883, 60)
# Main loop
try:
    client.loop_start()
    while True:
        capture_and_process()
        time.sleep(300)  # Wait for 5 minutes before capturing the next image
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()






