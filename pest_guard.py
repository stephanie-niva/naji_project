import cv2
import os
import numpy as np
import time
from picamera2 import Picamera2, Preview

def extract_features(image_path):
    """Extract features from a given image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def compare_images(query_image_path, dataset_folder_path):
    # Load the query image
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Could not load query image: {query_image_path}")
        return
    
    # Extract features from the query image
    orb = cv2.ORB_create()
    query_keypoints, query_descriptors = orb.detectAndCompute(query_img, None)
    print(f"Query image keypoints: {len(query_keypoints)}")
    
    # Initialize variables for confidence scores
    affected_count = 0
    confidence_threshold = 0.5  # Set a threshold for confidence score

    # Loop through each image in the dataset folder
    for image_file in os.listdir(dataset_folder_path):
        dataset_image_path = os.path.join(dataset_folder_path, image_file)
        dataset_keypoints, dataset_descriptors = extract_features(dataset_image_path)
        
        if dataset_descriptors is None:
            continue  # Skip if the dataset image could not be loaded

        # Create a Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(query_descriptors, dataset_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate the average distance of the matches
        if matches:
            average_distance = np.mean([m.distance for m in matches])
            max_distance = 100  # Adjust this based on your data
            confidence_score = 1 - (average_distance / max_distance)
            confidence_score = max(0, min(confidence_score, 1))  # Clamp to [0, 1]

            # Determine if the leaf is affected based on confidence score
            affected_status = "Affected" if confidence_score > confidence_threshold else "Not Affected"
            if affected_status == "Affected":
                affected_count += 1

            print(f"Matches with {image_file} - Confidence Score: {confidence_score:.2f} - {affected_status}")
    
    print(f'Number of affected leaves: {affected_count} out of {len(os.listdir(dataset_folder_path))}')

def capture_and_compare(interval_seconds, dataset_folder_path, num_images):
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    try:
        for i in range(num_images):
            # Capture image
            image_path = f"captured_image_{i}.jpg"
            picam2.capture_file(image_path)
            print(f"Captured image {i + 1}/{num_images}")

            # Compare with the dataset
            compare_images(image_path, dataset_folder_path)

            # Wait for the next interval
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("Image capture stopped by user.")
    finally:
        picam2.stop()

if __name__ == "__main__":
    # Parameters for image capture
    interval_seconds = 10  # Time interval in seconds
    dataset_folder_path = '/path/to/your/dataset'  # Replace with your dataset folder path
    num_images = 5  # Number of images to capture

    # Start capturing and comparing
    capture_and_compare(interval_seconds, dataset_folder_path, num_images)
