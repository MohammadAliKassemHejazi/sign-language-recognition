import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from inference_utils import estimate_label
from performance_utils import report_computational_cost, plot_distribution

# Load your trained model
model_path = "../model/sign.h5"  
model = load_model(model_path)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Start webcam
cap = cv2.VideoCapture(0)
inference_times = []

print("📷 Starting webcam inference. Press ESC to stop...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    predicted_label, inf_time = estimate_label(frame, model)
    inference_times.append(inf_time)

    cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Model Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

avg = report_computational_cost(inference_times)
plot_distribution(inference_times, avg)

