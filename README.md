# Healthcare IDS CNN-LSTM-XAI

A Secure and Explainable LSTM-Based Intrusion Detection Framework for Early Breach Detection and Prevention in Healthcare Systems

## Abstract
Healthcare institutions increasingly rely on Electronic Patient Record (EPR) systems to store and manage sensitive patient information, making them attractive targets for cyberattacks and unauthorized access. This project presents a hybrid CNN-LSTM framework with Grad-CAM++ and Score-CAM explainability for detecting malicious activities in healthcare systems.


## Repository Structure
- Main.py - main execution script for preprocessing, training, evaluation, and explainability output
- graph.py - visualization and plotting utilities
- Output/ - generated metrics and explainability heatmaps
- Dataset/ - unzip the dataset from the goolgle drive link 
- Instructions.docx - project instructions and documentation

## Dataset Setup
The dataset is intentionally not pushed to GitHub.

1. Download the dataset from the Google Drive link:
   - https://drive.google.com/file/d/1xoCubhd4R5oMSa0qz9ULtW8TVcqjLvq3/view?usp=drive_link
2. Extract the files locally.
3. Place the extracted dataset folder inside this project folder as:
   - Dataset/

## How to Run

1. Run the main script:
   - python Main.py
2. The outputs will be generated in the Output folder, including:
   - metrics.json
   - explainability_heatmaps.npz
   - plots and evaluation results

## Install Dependencies

It's recommended to use a virtual environment. From the project root:

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also install packages individually:

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap lime
```

## How to Unzip and Use the Dataset
If the downloaded dataset is provided as a ZIP file:
1. Right-click the ZIP file and choose Extract All.
2. Extract it into the project folder.
3. Ensure the folder name is Dataset.
4. If the dataset files are inside a nested folder, move or rename them so the project can read them from Dataset/.

## Keywords
Electronic Patient Record (EPR); Intrusion Detection; Convolutional Neural Network (CNN); Long Short-Term Memory (LSTM); Explainable Artificial Intelligence (XAI); Grad-CAM++; Score-CAM; Healthcare Cybersecurity.

