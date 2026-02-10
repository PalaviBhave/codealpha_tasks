\# Handwritten Character Recognition using Neural Networks



A machine learning project that recognizes handwritten digits (0-9) using a Multi-Layer Perceptron Neural Network trained on the MNIST dataset.



\## 📊 Project Overview



\- \*\*Task\*\*: Handwritten Character Recognition

\- \*\*Dataset\*\*: MNIST (70,000 handwritten digits)

\- \*\*Algorithm\*\*: Multi-Layer Perceptron (MLP) Neural Network

\- \*\*Accuracy\*\*: ~95-97%

\- \*\*Framework\*\*: scikit-learn



\## 🎯 Features



\- ✅ Automatic MNIST dataset download

\- ✅ Neural network with 2 hidden layers

\- ✅ Comprehensive visualizations (confusion matrix, accuracy charts)

\- ✅ Model training and evaluation

\- ✅ Prediction confidence scores

\- ✅ Model persistence (save/load)



\## 🏗️ Model Architecture

```

Input Layer:    784 neurons (28×28 pixels)

&nbsp;                  ↓

Hidden Layer 1: 128 neurons (ReLU activation)

&nbsp;                  ↓

Hidden Layer 2:  64 neurons (ReLU activation)

&nbsp;                  ↓

Output Layer:    10 neurons (Softmax)

```



\*\*Total Parameters\*\*: ~106,000



\## 📈 Results



| Metric | Value |

|--------|-------|

| Training Accuracy | ~97% |

| Test Accuracy | ~95-96% |

| Training Time | 2-5 minutes |

| Model Size | ~850 KB |



\## 🚀 Installation



\### Prerequisites

```bash

pip install scikit-learn numpy matplotlib seaborn

```



\### Clone the Repository

```bash

git clone https://github.com/YOUR\_USERNAME/handwritten-recognition.git

cd handwritten-recognition

```



\## 💻 Usage



Run the main script:

```bash

python handwritten\_recognition\_windows.py

```



The script will:

1\. Download MNIST dataset automatically

2\. Train the neural network

3\. Generate visualizations

4\. Save the trained model

5\. Create an `output/` folder with all results



\## 📁 Output Files



After running, you'll find these files in the `output/` folder:



| File | Description |

|------|-------------|

| `sample\_digits.png` | Sample handwritten digits from dataset |

| `training\_loss.png` | Training loss curve |

| `confusion\_matrix.png` | Model performance matrix |

| `sample\_predictions.png` | Prediction examples with confidence |

| `per\_digit\_accuracy.png` | Accuracy breakdown per digit |

| `model.pkl` | Trained model for reuse |



\## 🔮 Using the Trained Model

```python

import pickle

import numpy as np



\# Load the trained model

with open('output/model.pkl', 'rb') as f:

&nbsp;   model = pickle.load(f)



\# Prepare your image (28×28 pixels, grayscale, normalized)

image = your\_image / 255.0

image = image.flatten().reshape(1, -1)



\# Make prediction

prediction = model.predict(image)\[0]

probabilities = model.predict\_proba(image)\[0]

confidence = probabilities\[prediction] \* 100



print(f"Predicted Digit: {prediction}")

print(f"Confidence: {confidence:.2f}%")

```



\## 🛠️ Technical Details



\### Technologies Used

\- \*\*Python 3.x\*\*

\- \*\*scikit-learn\*\*: Neural network implementation

\- \*\*NumPy\*\*: Numerical computations

\- \*\*Matplotlib\*\*: Visualizations

\- \*\*Seaborn\*\*: Statistical plots



\### Key Components

\- Multi-Layer Perceptron (MLP) classifier

\- Adam optimizer

\- ReLU activation (hidden layers)

\- Softmax activation (output layer)

\- Early stopping to prevent overfitting

\- Cross-entropy loss function



\## 📊 Sample Results



\### Confusion Matrix

Shows how well the model predicts each digit (0-9)



\### Per-Digit Accuracy

Breakdown of accuracy for each individual digit



\### Sample Predictions

Visual examples of correct and incorrect predictions with confidence scores



\## 🎓 Learning Outcomes



This project demonstrates:

\- Neural network architecture design

\- Image classification techniques

\- Model training and evaluation

\- Data preprocessing and normalization

\- Performance metrics analysis

\- Model persistence and deployment



\## ⚙️ Customization



You can adjust these parameters in the code:

```python

\# Training size (line ~80)

sample\_size = 10000  # Increase to 60000 for full dataset



\# Model architecture (line ~100)

hidden\_layer\_sizes=(128, 64)  # Try (256, 128) for more capacity



\# Training epochs (line ~106)

max\_iter=20  # Increase for longer training

```



\## 🐛 Troubleshooting



\*\*Issue\*\*: Slow dataset download

\- \*\*Solution\*\*: First run takes longer (~70MB download). Be patient!



\*\*Issue\*\*: Low accuracy

\- \*\*Solution\*\*: Increase `sample\_size` and `max\_iter` in the code



\*\*Issue\*\*: Out of memory

\- \*\*Solution\*\*: Reduce `sample\_size` to 5000



\## 📝 License



This project is free to use for educational purposes.



\## 👨‍💻 Author

**PALAVI BHAVE**

- GitHub: [@PalaviBhave](https://github.com/PalaviBhave)

\## 🙏 Acknowledgments



\- MNIST dataset by Yann LeCun

\- scikit-learn library

\- Python community



\## 📧 Contact



For questions or suggestions, feel free to open an issue!



---



⭐ \*\*Star this repo\*\* if you found it helpful!

