
# ✋ Sign Language Digit Recognition

This project is a real-time Sign Language Digit Recognition system that identifies hand gestures for numbers 0 through 9 using deep learning and transfer learning with the VGG16 model. It uses OpenCV for live webcam input and TensorFlow/Keras for model training and inference.

---

## 📌 Features

- ✅ Real-time digit recognition via webcam  
- ✅ Transfer learning using pre-trained VGG16  
- ✅ Clean and modular code (data preprocessing, training, prediction)  
- ✅ Image normalization and one-hot encoding  
- ✅ Accuracy/loss visualization  
- ✅ Easily extendable to more gestures or alphabets  

---

## 🧠 Technologies Used

- Python 3  
- TensorFlow / Keras  
- OpenCV  
- Matplotlib  
- Scikit-learn  
- Pre-trained VGG16 (ImageNet)  
- Google Colab (for training)  

---

## 📁 Dataset

- **Name:** Sign Language for Numbers  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers)  
- **License:** CC0: Public Domain  
- **Content:** Folders containing hand gesture images for digits 0–9  

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```


2. Install Required Packages
```bash
pip install -r requirements.txt
```
3. Prepare the Dataset

Download the dataset manually from Kaggle

Extract the dataset and place it in the root project folder (e.g., /SignLanguageforNumbers/)


4. Train the Model (Optional if using pretrained)
```bash
python train_model.py
```
5. Run Real-Time Prediction
```bash
python realtime_prediction.py
```

🧪 Model Overview

Base model: VGG16 (pretrained, top layers removed)

Custom layers: Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(10, Softmax)

Loss Function: Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy


📈 Model Performance

Training Accuracy: 93%+

Validation Accuracy: 95%+

Model generalizes well to unseen images

Robust under moderate lighting conditions


🖼 Sample Output

Real-time webcam view with predicted digit overlay

Frame processed and prediction updated every 1 frame

Simple, clean UI with cv2.putText() used for displaying output


📌 Future Improvements

Expand to full ASL alphabet and dynamic gestures

Improve performance in low-light conditions

Add gesture smoothing for continuous predictions

Deploy to mobile using TensorFlow Lite


🙋‍♂️ Author
Mohamad Ali Hijazi
📍 Based in Hungary



