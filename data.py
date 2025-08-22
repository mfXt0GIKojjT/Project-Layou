# Dataset Information

This project involves four publicly available biomedical and clinical datasets.
Since each dataset requires application or download through official channels, **this repository does not provide raw data directly**.
Please follow the links below to apply for or download the datasets.

---

## 1. MIMIC-III Dataset
- **Description**: A large-scale ICU clinical database containing de-identified health records of over 60,000 ICU admissions. Includes vital signs, medications, lab tests, and diagnostic codes.
- **Usage**: Widely used for machine learning research in medical informatics (e.g., outcome prediction, clinical decision support).
- **Access**: [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)

---

## 2. NHANES Dataset
- **Full Name**: National Health and Nutrition Examination Survey
- **Description**: Conducted by the U.S. CDC to assess the health and nutritional status of adults and children in the U.S. Includes demographics, dietary information, physical examinations, and some imaging (e.g., X-rays).
- **Note**: Does not include human body keypoint annotations or pose estimation data.
- **Access**: [https://wwwn.cdc.gov/nchs/nhanes/Default.aspx](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx)

---

## 3. UK Biobank Dataset
- **Description**: A large biomedical database with genetic, lifestyle, and health data from 500,000 UK participants. Includes imaging data such as MRI and ultrasound, and some functional task recordings.
- **Note**: Does not include large-scale 3D motion capture or synchronized multi-view video for pose estimation.
- **Access**: [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/)

---

## 4. DEAP Dataset
- **Full Name**: Dataset for Emotion Analysis using Physiological signals
- **Description**: A multimodal dataset for emotion research, including EEG recordings, facial videos, and self-reported emotional responses while watching music videos.
- **Note**: Not a sports or human pose dataset; unrelated to the LSP dataset.
- **Access**: [https://www.eecs.qmul.ac.uk/mmv/datasets/deap/](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

---

## 📌 Usage Instructions
1. Apply for and download the datasets from the official links above.
2. Place the raw data into `data/raw/`.
3. Run `data/preprocess.py` to preprocess and format the data for model training.
