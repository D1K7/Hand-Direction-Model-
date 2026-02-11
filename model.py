import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter

model = tf.keras.models.load_model('hand_direction_model6.keras')
class_names = sorted(['up', 'down', 'left', 'right', 'idle'])

# Buffer Configuration
BUFFER_SIZE = 10  # Remembers the last 10 frames
prediction_buffer = deque(maxlen=BUFFER_SIZE)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Grayscale Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (128, 128))
    img_array = np.expand_dims(img, axis=(0, -1))

    # 2. Prediction
    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction[0])
    raw_label = class_names[class_idx]
    confidence = np.max(prediction[0])

    # 3. Temporal Buffer Logic
    # We only add to buffer if confidence is high enough
    if confidence > 0.75:
        prediction_buffer.append(raw_label)
    else:
        prediction_buffer.append('idle')

    # 4. Determine "Smoothed" Label
    # Find the most common label in the last 10 frames
    most_common = Counter(prediction_buffer).most_common(1)
    smoothed_label = most_common[0][0] if most_common else "idle"
    occurrence_count = most_common[0][1] if most_common else 0

    # 5. UI Feedback
    display_text = f"CMD: {smoothed_label.upper()} ({occurrence_count}/{BUFFER_SIZE})"
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Smoothed Grayscale Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()