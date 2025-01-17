
# TensorFlow Dataset Train and Test

This repository contains scripts and instructions for training and testing a machine learning model using TensorFlow. The model is trained on a dataset of cat and rabbit images. 

## Prerequisites

Ensure you have the following versions installed:

- Python 3.12
- TensorFlow 2.17.0

## Getting Started

### 1. Clone or Download the Repository

Clone the repository to your local machine using:

```bash
git clone https://github.com/ammarproficient/tensorflow_dataset_train_test.git
```

Or download the repository as a ZIP file and extract it.

### 2. Setup

Make sure you have the correct versions of Python and TensorFlow installed. You can create a virtual environment and install the required packages:

```bash
# Create a virtual environment (optional)
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

# Install TensorFlow
pip install tensorflow==2.17.0
```

### 3. Train the Model

Run the `train.py` script to train the model. This script will create a trained model file with the `.h5` extension.

```bash
python train.py
```

### 4. Test the Model

Run the `test.py` script to evaluate the model on the test dataset. You can replace or add different images to the `test` folder to check the accuracy of the model on new images.

```bash
python test.py
```

### 5. Customization

You can modify the `train.py` script to adjust parameters and improve the model's performance. Experiment with different values to achieve better results.

## Files

- `train.py`: Script to train the model.
- `test.py`: Script to test the model.
- `model.h5`: The trained model file (created after running `train.py`).
- `dataset/`: Directory containing training and test datasets.

## Project Directory

```
project-root/
│
├── train.py
├── test.py
├── dataset/
│   ├── train-cat-rabbit/
│   │   ├── cat/
│   │   └── rabbit/
│   └── test-images/
│   │   ├── cat/
│   │   └── rabbit/
|   └── val-cat-rabbit/
│   │   ├── cat/
│   │   └── rabbit/
└── model.h5
```

## Notes

- Ensure your images are properly labeled and placed in the correct folders (`cat` and `rabbit`).
- The model’s performance may vary based on the dataset and parameters used. Fine-tuning and additional experimentation may be required.


## Contact

For any questions or issues, please contact [ammarproficient@gmail.com](mailto:your-email@example.com).

