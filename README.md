
# Credit-card-fraud-detection-federated-learning

Credit card fraud is one of the fastest-growing financial crimes globally, causing billions of dollars in losses every year. Traditional fraud detection systems face a critical challenge known as the Data Silo Paradox — machine learning models need large diverse datasets to detect fraud accurately, but financial institutions are legally prohibited from sharing sensitive customer transaction data due to GDPR and data confidentiality regulations.

This project presents a complete simulation of Federated Learning to solve this paradox. Instead of sharing raw customer data, three simulated banks — HDFC, SBI, and ICICI — each train a fraud detection model privately on their own isolated data and share only the learned model weights with a central server. The central server combines these weights using the Federated Averaging (FedAvg) algorithm to produce one powerful Global Model that benefits from the fraud knowledge of all three banks combined, without any raw transaction data ever leaving any institution.

The project uses the ULB Credit Card Fraud Detection dataset from Kaggle containing 284,807 real credit card transactions made by European cardholders, of which only 492 are fraudulent representing 0.173 percent of total transactions. The dataset is partitioned into three equal bank simulations. Each bank independently applies SMOTE to handle the severe class imbalance and trains a local Logistic Regression model. The FedAvg algorithm then aggregates the model weights to create the Global Federated Model.

The Global Model achieved an F1-Score of 0.1417, representing a 37 percent improvement over the best individual bank model, and correctly identified 88 out of 98 fraud cases in the test set. An interactive Streamlit dashboard with five tabs provides complete visualisation of the pipeline including EDA, confusion matrices, ROC curves, and model comparison charts.

Tech Stack: Python, Scikit-learn, Imbalanced-learn, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Streamlit, Google Colab

Dataset: ULB Credit Card Fraud Detection — Kaggle


This project proves that financial institutions can collaboratively improve fraud detection performance without violating customer privacy or data protection regulations. Zero raw customer data is shared at any stage. Fully GDPR compliant.

## Acknowledgements

 - [Research Paper](https://link.springer.com/article/10.1007/s44230-022-00004-0)
 
 - [Data Set](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

## How to Run 

   1. Open any browser like Google Chrome or Safari. 

   2. In search engine search for Google Colab and open it. 

   3. In Google Colab open the .pynb file provided.

   4. Run all cells one by one.

   5. Upload Dataset in cell 3 and run it.

   6. After that run all other cells to get the result. 

## Result table 

<img width="452" height="277" alt="Screenshot 2026-07-04 at 12 17 52 PM" src="https://github.com/user-attachments/assets/2760ec24-fee9-4bb3-ab46-b5000d6993ad" />
