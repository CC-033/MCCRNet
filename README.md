# MCCRNet: Multimodal Classroom Climate Recognition Network

This repository contains the official code and data access information for the work: Multimodal Classroom Climate Recognition

## 📦 1. Data Access and Setup (Data Access)

### Dataset Name and Acquisition

* **Dataset Name:** [Multimodal Classroom Climate Dataset(MCCD)]
**📢 Acquisition Instructions:**
1.  **Download Link:** The raw dataset files (video, audio, etc.) can be downloaded from the following location:
    > **[[Dataset Download Link](https://osf.io/fw28j/overview)]**
2. **Usage:** Researchers must commit to using it solely for research purposes and strictly protect participant privacy.
3. **Data Splits:** Reference splits are provided with the dataset link, but you may customize the partitions according to your own needs.

## ⚙️ 2. Environment and Installation

### 2.1 Dependencies

* Python Version: [Python 3.8+]

### 2.2 Package Installation
Install all required dependencies using `pip`:
'''bash
pip install -r requirements.txt
'''

### 🚀 **3. Model Execution and Reproduction**
#### **3.1 Pre-trained Checkpoints**
The model weights used to report the results are provided within the /pre_trained_models directory of this repository.

#### **3.2 Training the Model**
Start the training process

'''bash
python main.py --visual_encoder_type xx --acoustic_encoder_type --use_xx_fusion --save_name xx
'''
