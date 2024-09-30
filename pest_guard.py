import cv2
import numpy as np
import time
from picamera2 import Picamera2
import os

# Initialize Picamera2
camera = Picamera2()
camera.configure(camera.create_still_configuration())

# Path to your dataset folder
dataset_path = "/home/naji/naji_project/Miner leaves/"

# Helper Functions
def preprocess_image(image):
    """Preprocess image by converting to grayscale and applying Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def extract_orb_features(image):
    """Extract ORB keypoints and descriptors from the image."""
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """Match ORB descriptors between the captured image and a dataset image."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance (similarity)
    return matches

def calculate_confidence(matches):
    """Calculate the confidence score based on the number and quality of matches."""
    if not matches:
        return 0
    avg_distance = np.mean([m.distance for m in matches])
    confidence = max(1 - avg_distance / 100, 0)  # Confidence score between 0 and 1
    return confidence * 100  # Convert to percentage

def find_best_match(captured_image):
    """Compare the captured image with each image in the dataset and find the best match."""
    best_confidence = 0
    best_match_filename = None

    # Preprocess the captured image
    preprocessed = preprocess_image(captured_image)
    
    # Extract ORB features from the captured image
    keypoints1, descriptors1 = extract_orb_features(preprocessed)

    # Loop through each image in the dataset
    for dataset_image_filename in os.listdir(dataset_path):
        dataset_image_path = os.path.join(dataset_path, dataset_image_filename)
        dataset_image = cv2.imread(dataset_image_path)
        if dataset_image is None:
            continue

        # Preprocess dataset image and extract ORB features
        preprocessed_dataset_image = preprocess_image(dataset_image)
        keypoints2, descriptors2 = extract_orb_features(preprocessed_dataset_image)

        # Match features between the captured image and the dataset image
        matches = match_features(descriptors1, descriptors2)
        confidence = calculate_confidence(matches)

        # Keep track of the highest confidence match
        if confidence > best_confidence:
            best_confidence = confidence
            best_match_filename = dataset_image_filename

    return best_match_filename, best_confidence

def capture_and_process(threshold=80):
    """
    Capture an image, process it for leaf disease detection via ORB matching, then discard it.
    """
    # Capture image
    filename = "leaf_image.jpg"
    camera.start()
    time.sleep(2)  # Wait for camera to adjust
    camera.capture_file(filename)
    camera.stop()

    # Read captured image
    captured_image = cv2.imread(filename)

    # Compare the captured image with the dataset
    best_match, confidence = find_best_match(captured_image)

    # Decision based on confidence
    if confidence > threshold:
        print(f"Leaf is affected! Match found with: {best_match}, Confidence: {confidence}%")
    else:
        print(f"Leaf is healthy or not matching any affected dataset image. Confidence: {confidence}%")

    # Discard the image after processing
    print("Processing done. Discarding image...")
    return True

# Main Loop
while True:
    capture_and_process()
    time.sleep(300)  # Wait for 5 minutes before capturing the next image
