import cv2
import time

# Define motion detection parameters
min_area = 5000  # minimum area of motion contour
threshold = 50   # threshold for detecting motion

# Initialize video capture object
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

# Wait for camera to warm up
time.sleep(2)

# Initialize variables for motion detection
first_frame = None
motion_detected = False

while True:
    # Read frame from video capture object
    ret, frame = vc.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Store first frame for motion detection
    if first_frame is None:
        first_frame = gray
        continue
    
    # Compute absolute difference between current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours in thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if motion has been detected
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        motion_detected = True
        break
    
    # If motion has been detected, capture image
    if motion_detected:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"motion_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Motion detected! Image saved as {filename}")
    
    # Display video feed in window
    cv2.imshow("preview", frame)
    
    # Wait for key press event
    key = cv2.waitKey(1) & 0xFF
    
    # If 'q' key is pressed, break from loop
    if key == ord("q"):
        break

# Release video capture object and close all windows
vc.release()
cv2.destroyAllWindows()
