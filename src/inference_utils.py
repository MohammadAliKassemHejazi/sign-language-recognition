import cv2
import numpy as np
import time

class_mapping = {i: str(i) for i in range(10)}

def estimate_label(frame, model):
    frame_resized = cv2.resize(frame, (100, 100))
    img_array = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0
    start_time = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.perf_counter()
    predicted_class = np.argmax(predictions)
    predicted_label = class_mapping[predicted_class]
    inference_time = (end_time - start_time) * 1000
    return predicted_label, inference_time
