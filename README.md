# RoadSign-Detector
# Deployed online at the following link
https://roadsign-detector.streamlit.app/

Detects and classifies stop signs, railroad crossing signs, no parking signs, speed limit signs, and yield signs from images. Uses a heuristic approach to classification through extracted features from images utilizing sketch recognition techniques rather than a deep learning model. This allows for exponentially faster computations

- Utilizes OpenCV for image preprocessing and feature extraction
- Image sizes are normalized
- Erosion and dilation used to reduce noise
- Color spaces are extracted
- Thresholding is performed to segment sign regions
- Canny edge detection and contour detection is used to extract number of edges on potential signs
- Tesseract used to further extract text from images
- Simple frontend built using Streamlit