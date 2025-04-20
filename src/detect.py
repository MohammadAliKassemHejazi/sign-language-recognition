import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load your trained model
model_path = "../model/sign.h5"  
model = load_model(model_path)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Class label mapping
class_mapping = {i: str(i) for i in range(10)}

# Prepare webcam
cap = cv2.VideoCapture(0)

# Timing storage
inference_times = []

print("📷 Starting webcam inference. Press ESC to stop...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to fit model input
    frame_resized = cv2.resize(frame, (100, 100))

    # Preprocess
    img_array = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0

    # Inference timing
    start_time = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.perf_counter()

    # Record timing
    inference_times.append((end_time - start_time) * 1000)  # ms

    # Prediction
    predicted_class = np.argmax(predictions)
    predicted_label = class_mapping[predicted_class]

    # Annotate and display
    cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Model Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

# === Report computational cost ===
inference_times = np.array(inference_times)
avg_time = np.mean(inference_times)
fps = 1000 / avg_time

print("\n📊 Classification Timing Stats:")
print(f"Frames Processed: {len(inference_times)}")
print(f"Average Inference Time: {avg_time:.2f} ms")
print(f"Min Time: {np.min(inference_times):.2f} ms")
print(f"Max Time: {np.max(inference_times):.2f} ms")
print(f"Standard Deviation: {np.std(inference_times):.2f} ms")
print(f"Estimated FPS: {fps:.1f}")

# === Optional: Plot distribution ===
plt.figure(figsize=(10, 4))
plt.hist(inference_times, bins=20, color="skyblue", edgecolor="black")
plt.axvline(avg_time, color="red", linestyle="--", label=f"Avg: {avg_time:.2f} ms")
plt.title("Inference Time Distribution per Frame")
plt.xlabel("Inference Time (ms)")
plt.ylabel("Number of Frames")
plt.legend()
plt.tight_layout()
plt.show()
