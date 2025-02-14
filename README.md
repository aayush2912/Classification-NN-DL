# DeepWear - Advanced Deep Learning for Fashion Recognition

## Project Overview
A deep learning project that classifies fashion items into 10 different categories using transfer learning with the Xception model. The project demonstrates the progressive development of a robust image classification system through various optimization techniques.

## Dataset
- **Classes:** 10 clothing categories
  - T-shirt (928 items)
  - Long Sleeve (576 items)
  - Pants (559 items)
  - Shirt (345 items)
  - Shoes (297 items)
  - Dress (288 items)
  - Shorts (257 items)
  - Outwear (246 items)
  - Hat (149 items)
  - Skirt (136 items)
- **Split:** 60-20-20 (Train-Validation-Test)
- **Image Dimensions:** Initially 150x150, later upgraded to 299x299

## Development Process
### 1. Initial Model Setup
- **Base Model:** Xception (pre-trained on ImageNet)
- **Input Size:** 150x150x3
- **Architecture:**
  - Frozen Xception base
  - GlobalAveragePooling2D
  - Dense output layer (10 classes)

### 2. Learning Rate Optimization
- **Tested rates:** [0.0001, 0.001, 0.01, 0.1]
- **Best performing:** 0.001
- **Metric:** Validation accuracy

### 3. Model Architecture Enhancement
- **Added inner dense layer**
- **Experimented with layer sizes:** [10, 100, 1000]
- **Final architecture:**
  ```python
  Input → Xception → GlobalAveragePooling2D → Dense(100) → Dropout → Dense(10)
  ```

### 4. Regularization Implementation
- **Dropout rates tested:** [0.0, 0.2, 0.5, 0.8]
- **Optimal dropout rate:** 0.2
- **Added model checkpointing**

### 5. Data Augmentation
Implemented various augmentations:
- Rotation range
- Width/height shifts
- Shear range
- Zoom range
- Horizontal flips

### 6. Final Model Upgrade
- **Increased input size** to 299x299
- **Reduced learning rate** to 0.0005
- **Maintained dropout rate** at 0.2
- **Added comprehensive checkpointing**

## Results
- **Final Test Accuracy:** 90.05%
- **Test Loss:** 0.2711
- **Best performing model:** `xception_v4_1_31_0.906.h5`
- **Consistent performance across validation and test sets**
- **Successfully distinguishes between similar clothing items (e.g., pants vs shorts)**

## Technical Implementation Details
```python
# Model Configuration
input_size = 299
learning_rate = 0.0005
inner_layer_size = 100
dropout_rate = 0.2

# Data Preprocessing
- Image size: 299x299
- Batch size: 32
- Preprocessing: Xception's preprocess_input
```

## Key Learnings
- **Transfer learning effectiveness** for fashion classification
- **Importance of methodical hyperparameter tuning**
- **Balance between model capacity and regularization**
- **Progressive model development approach**
- **Impact of input size on model performance**

## Requirements
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Python 3.9

## Usage
```python
# Load and prepare image
img = load_img(path, target_size=(299, 299))
x = preprocess_input(np.array([np.array(img)]))

# Make prediction
pred = model.predict(x)
probabilities = tf.nn.softmax(pred[0])
```

This project demonstrates the successful application of transfer learning and various deep learning optimization techniques to achieve high accuracy in fashion image classification.
