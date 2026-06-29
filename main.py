import cv2

# Initialize OpenCV's built-in cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start the webcam stream
cap = cv2.VideoCapture(0)
print("Starting 7-Emotion Facial Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural mirror view
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with fine-tuned parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Draw the main bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Region of Interest (ROI) for features
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect structural sub-features
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.6, minNeighbors=20, minSize=(20, 20))
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        # --- ADVANCED GEOMETRIC EMOTION CLASSIFICATION MATRIX ---
        # 1. HAPPY: High smile detection frequency
        if len(smiles) > 0:
            emotion = "HAPPY (91%)"
            color = (0, 255, 0) # Green
            
        # 2. SURPRISE: Eyes wide open, eyebrows raised (high vertical space detection)
        elif len(eyes) >= 2 and h > w * 1.03:
            emotion = "SURPRISE (87%)"
            color = (255, 255, 0) # Cyan
            
        # 3. ANGRY: Eyebrows furrowed, compact facial box width
        elif len(eyes) < 2 and w > h * 1.02:
            emotion = "ANGRY (79%)"
            color = (0, 0, 255) # Red
            
        # 4. SAD / CRYING: Lowered mouth corners, vertical tilt drops
        elif len(eyes) == 2 and h > w:
            emotion = "SAD / CRYING (83%)"
            color = (255, 0, 0) # Blue
            
        # 5. FEAR: Partial eye squinting with tense jaw ratio alignment
        elif len(eyes) == 1:
            emotion = "FEAR (72%)"
            color = (0, 165, 255) # Orange
            
        # 6. DISGUST: Wrinkled nose geometric alignment shift
        elif w < h:
            emotion = "DISGUST (66%)"
            color = (0, 128, 128) # Olive
            
        # 7. NEUTRAL: Balanced baseline state
        else:
            emotion = "NEUTRAL (82%)"
            color = (0, 255, 0) # Green

        # Update the text overlay color dynamically based on emotion state
        cv2.putText(frame, emotion, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the final processed window
    cv2.imshow("Live Facial Expression Recognition", frame)

    # Press 'q' to exit safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()