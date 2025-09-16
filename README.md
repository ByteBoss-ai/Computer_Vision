# MNIST Digit Classification

## Overview
This project is a beginner-friendly implementation of a **handwritten digit classifier** using the MNIST dataset. The model is a **fully connected neural network** built with Keras and trained to classify digits from 0 to 9. 

The focus of this project is on:
- Dataset understanding
- Data preprocessing
- Model building and training
- Evaluating results using accuracy and confusion matrix
- Visualizing predictions

---

## Dataset
- **Dataset Used:** MNIST (handwritten digits)  
- **Training Samples:** 60,000 images  
- **Test Samples:** 10,000 images  
- **Image Size:** 28×28 pixels, grayscale  
- **Number of Classes:** 10 (digits 0–9)  

---

## Workflow
1. **Data Exploration:** Visualized examples of each digit to understand patterns.  
2. **Preprocessing:** 
   - Normalized pixel values to [0,1]  
   - Flattened images to 1D vectors  
   - Converted labels to one-hot encoding  
3. **Model Building:**  
   - Two hidden layers with 128 neurons each  
   - Dropout to prevent overfitting  
   - Output layer with 10 neurons and softmax activation  
4. **Training:** Model trained on 60,000 images for 10 epochs with batch size 512.  
5. **Evaluation:** Tested on 10,000 unseen images and calculated accuracy.  
6. **Visualization:** Displayed single predictions and a confusion matrix for overall performance.

---

## Results
- **Test Accuracy:** ~0.948 
- **Confusion Matrix:** Most predictions are correct; few misclassifications occur on similar digits.  
- **Single Image Prediction:** The model can correctly predict individual digits with high confidence.  

---

## Observations
- Simple fully connected networks perform well on MNIST, a small grayscale dataset.  
- Dropout improves generalization and reduces overfitting.  
- Misclassifications mainly happen for digits that look similar (e.g., 3 vs 5).  

---

## Tools & Libraries
- Python 3  
- Keras / TensorFlow  
- NumPy, Matplotlib, Seaborn  
- Scikit-learn  

---

## Future Work
- Use **Convolutional Neural Networks (CNNs)** for higher accuracy.  
- Apply **data augmentation** to improve generalization.  
- Analyze **misclassified samples** to understand model weaknesses.

