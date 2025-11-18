import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = img.astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))[0]

    label = np.argmax(pred)

    cv2.putText(frame, f"Gesture: {label}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()