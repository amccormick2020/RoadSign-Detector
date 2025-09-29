### Current Iteration of Model ###

### How to Use Model
### 1. Upload .png of roadsign to colab file storage
### 2. change image_path variable to be the name of the roadsign file
# Update the image path to your image
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# Ensure Tesseract is properly set up
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update if necessary

def detect_road_sign(image):
    original_height, original_width = image.shape[:2]

    # Convert to BGR if the input is in RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # Resize the image for better processing
    image_resized = cv2.resize(image, (600, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edged = cv2.Canny(blurred, 50, 150)

    # Convert to HSV
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image for colors
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean up masks with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_white = cv2.dilate(mask_white, kernel, iterations=1)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)

    # Detect Stop Signs and No Parking Signs
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 500:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity > 0.7:  # Circular shapes (for No Parking signs)
                roi = image_resized[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary_roi = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
                custom_config = r'--psm 6'
                detected_text = pytesseract.image_to_string(binary_roi, config=custom_config)
                if 'P' in detected_text or "NO PARKING" in detected_text.upper():
                    cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(image_resized, "No Parking Sign", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                elif len(approx) == 8 and 0.8 < aspect_ratio < 1.2:
                    cv2.drawContours(image_resized, [approx], 0, (0, 0, 255), 3)
                    cv2.putText(image_resized, "Stop Sign", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)



    # Detect Speed Limit Signs
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_white:
        area = cv2.contourArea(contour)
        if area > 1000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if len(approx) == 4 and 0.5 < aspect_ratio < 1.5:
                cv2.drawContours(image_resized, [approx], 0, (255, 0, 0), 3)
                cv2.putText(image_resized, "Speed Limit Sign", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Detect Yield Signs
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 1.5:
                    cv2.drawContours(image_resized, [approx], 0, (0, 255, 255), 3)
                    cv2.putText(image_resized, "Yield Sign", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Detect Railroad Crossing Signs
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # List to store bounding boxes for merging
    bounding_boxes = []

    for contour in contours_yellow:
        # Filter smaller contours to avoid noise
        area = cv2.contourArea(contour)
        if area > 500:  # Lowered the area threshold for detecting smaller or dim regions
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is roughly circular
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.7 < aspect_ratio < 1.3:  # Allow a broader range for aspect ratio
                # Check the circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.6:  # Lowered the circularity threshold for imperfect circles
                    bounding_boxes.append((x, y, w, h))

    # Merge bounding boxes into a single region
    if bounding_boxes:
        x_min = min([x for x, y, w, h in bounding_boxes])
        y_min = min([y for x, y, w, h in bounding_boxes])
        x_max = max([x + w for x, y, w, h in bounding_boxes])
        y_max = max([y + h for x, y, w, h in bounding_boxes])

        # Extract the merged bounding box
        roi = image_resized[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_roi = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)  # Lowered the binary threshold

        # Detect lines or 'X' shape in the ROI
        edges = cv2.Canny(binary_roi, 30, 120)  # Adjusted edge detection thresholds
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=10)

        if lines is not None and len(lines) >= 4:  # Expect at least 4 lines for 'X'
            cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(image_resized, "Railroad Crossing", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGBA2BGR)
    image_resized = cv2.resize(image_resized, (original_width, original_height))

    return image_resized

# Streamlit GUI
st.title("Road Sign Detector")
st.write("By Austin McCormick")

uploaded_file = st.file_uploader("Choose a road sign image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert to numpy array

    # Process the image using the detection function
    processed_image = detect_road_sign(image_np)

    # Display the original and processed images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(processed_image, caption="Processed Image", use_container_width=True)

