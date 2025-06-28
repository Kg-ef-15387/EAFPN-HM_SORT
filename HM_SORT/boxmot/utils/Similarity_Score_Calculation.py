import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import os

def preprocess_image(img):
    img_resized = cv2.resize(img, (500, 500))
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    img_equalized = cv2.equalizeHist(img_gray)
    return img_equalized

def calculate_similarity_score(img1, img2):
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    img1_gray = cv2.resize(img1_gray, (500, 500))
    img2_gray = cv2.resize(img2_gray, (500, 500))

    similarity_score = ssim(img1_gray, img2_gray)

    similarity_score = (similarity_score + 1) / 2.0
    return similarity_score

def find_similar_ridges(img1, img2):
    img1_gray = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_gray = cv2.resize(img1_gray, (500, 500))
    img2_gray = cv2.resize(img2_gray, (500, 500))

    result = match_template(img1_gray, img2_gray)
    y, x = np.unravel_index(np.argmax(result), result.shape)

    img1_points = cv2.goodFeaturesToTrack(img1_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    img2_points = cv2.goodFeaturesToTrack(img2_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    img2_matched_points = []
    for point1 in img1_points:
        x1, y1 = point1.ravel()
        distances = euclidean_distances(point1, img2_points.reshape(-1, 2))
        min_distance_idx = np.argmin(distances)
        matched_point = img2_points[min_distance_idx]
        img2_matched_points.append(matched_point)

    return img1_points, img2_matched_points

def segment_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw the segmented regions
    segmented_mask = np.zeros_like(img)

    # Draw the contours on the segmented mask
    cv2.drawContours(segmented_mask, contours, -1, (0, 0, 255), 2)

    # Combine the original image with the segmented mask
    segmented_img = cv2.addWeighted(img, 0.7, segmented_mask, 0.3, 0)

    return segmented_img

def visualize_images(img1, img2, similarity_score, segmented_img1, segmented_img2, img1_points, img2_matched_points):
    height = min(img1.shape[0], img2.shape[0])

    img1_display = cv2.resize(img1, (int(height * img1.shape[1] / img1.shape[0]), height))
    img2_display = cv2.resize(img2, (int(height * img2.shape[1] / img2.shape[0]), height))

    # Draw circles on the concatenated image to show matched points
    concatenated = np.hstack((img1_display, img2_display))
    for point1, point2 in zip(img1_points, img2_matched_points):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        x2 += img1_display.shape[1]  # Shift the x-coordinate for the second image
        cv2.circle(concatenated, (int(x1), int(y1)), 5, (0, 255, 0), -1)  # Green circle for img1
        cv2.circle(concatenated, (int(x2), int(y2)), 5, (0, 255, 0), -1)  # Green circle for img2

    text = f"Similarity score: {similarity_score:.4f}"
    cv2.putText(concatenated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(f"Images and Segmented Images", concatenated)
    cv2.imshow(f"Segmented Image 1", segmented_img1)
    cv2.imshow(f"Segmented Image 2", segmented_img2)
    cv2.waitKey(0)

# Load the original images
def read_tif_files(folder_path):

    img_path = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(file_path)
                img_path.append(image)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return img_path

def SSIM_Cost(detection_img, track_img):
    ssim_cost = np.zeros((len(detection_img), len(track_img)))
    detection_preprocess = [preprocess_image(i) for i in detection_img]
    track_preprocess = [preprocess_image(i) for i in track_img]
    for i in range(len(detection_img)):
        for j in range(len(track_img)):
            ssim_cost[i][j] = calculate_similarity_score(detection_preprocess[i], track_preprocess[j])

    return ssim_cost
