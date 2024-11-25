# Credit Card Fraud Detection

## Overview

This repository contains an advanced solution for detecting credit card fraud using machine learning techniques. By analyzing anonymized transaction data from European cardholders in 2023, we developed a robust fraud detection system with exceptional performance metrics, including a near-perfect **accuracy of 99.99%** and an **AUC-ROC score of 0.999**.

The project leverages state-of-the-art algorithms and a comprehensive exploratory data analysis (EDA) pipeline to distinguish legitimate transactions from fraudulent ones. It aims to enhance financial security and assist institutions in reducing fraud-related losses.

For more details, check the [High-Level Design Document](./reports/Credit_Card_Fraud_Detection_HLD.pdf).

---

## Key Features

- **Extensive Dataset**: Over **550,000 anonymized credit card transaction records** from 2023.
- **Cutting-Edge Models**: Machine learning algorithms including Logistic Regression, Random Forest, and Gradient Boosting.
- **High Accuracy**: Achieved an accuracy of **99.99%** with a focus on minimizing false positives.
- **Balanced Training**: Leveraged techniques like SMOTE to handle class imbalance effectively.
- **Real-Time Potential**: Designed for real-world deployment with seamless integration into existing systems.

---

## Project Structure

```
.
├── data/                      # Dataset folder
├── notebooks/                 # Jupyter Notebooks for EDA and model building
├── reports/                   # Documentation and reports
│   ├── Credit_Card_Fraud_Detection_HLD.pdf
├── src/                       # Source code for model training and evaluation
│   ├── preprocess.py          # Data preprocessing scripts
│   ├── model_training.py      # Machine learning pipeline
│   └── evaluation.py          # Evaluation metrics and reporting
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependencies
├── README.md                  # Project overview
└── LICENSE                    # Licensing details
```

---

## Results

### Performance Metrics:

- **Accuracy**: 99.99%
- **AUC-ROC**: 0.999
- **Precision-Recall AUC**: 0.999
- **Confusion Matrix**:
  - **False Positives**: 86
  - **False Negatives**: 20

### Model Insights:

- **Important Features**: Key contributors to fraud detection include anonymized features (e.g., V10, V12, and V14).
- **Precision-Recall Curve**: Demonstrates a near-perfect balance between fraud detection and minimizing false alarms.

---

## Tools and Technologies

- **Programming Language**: Python
- **Machine Learning Libraries**:
  - Scikit-learn
  - TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook, VS Code
- **Cloud**: Google Colab/AWS

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Preprocess the dataset:
   ```bash
   python src/preprocess.py
   ```
4. Train the model:
   ```bash
   python src/model_training.py
   ```
5. Evaluate the results:
   ```bash
   python src/evaluation.py
   ```

---

## High-Level Design Document

For an in-depth understanding of the project, refer to the [High-Level Design Document](./reports/Credit_Card_Fraud_Detection_HLD.pdf).

---

## Future Work

- **Incorporate Time-Based Patterns**: Explore time-series models to capture temporal fraud patterns.
- **Advanced Models**: Test ensemble methods like XGBoost or LightGBM for enhanced performance.
- **Real-World Deployment**: Build an API for real-time fraud detection.

---

## 📝 Contact

**Md Fahimul Kabir Chowdhury**
**Email**: [info@tech2etc.com](mailto:info@tech2etc.com)
**LinkedIn**: [Md Fahimul kabir Chowdhury](https://bd.linkedin.com/in/fahimulkabirchowdhury)

Feel free to reach out for collaborations, job opportunities, or any queries regarding this project!

---

## License

This project is licensed under the [MIT License](./LICENCE).

---

## Acknowledgments

- Data Source: [Kaggle - Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- Tools and Techniques: Thanks to Scikit-learn, TensorFlow, and Matplotlib for their robust ecosystems.
