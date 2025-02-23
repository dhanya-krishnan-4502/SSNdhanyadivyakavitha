---

# **SSNdhanyadivyakavitha at MEDIQA-Sum 2023**
### **Medical Dialogue Summarization using Linear Support Vector Classification**

## **Overview**
This project was developed as part of the **ImageCLEFmed MEDIQA-Sum 2023** competition, which aims at **medical dialogue summarization**. The focus of this research is to categorize and summarize **doctor-patient conversations** into relevant **section headers** using **machine learning techniques**. 

We implemented multiple models, including
- **Linear Support Vector Classification (SVC)** with TF-IDF for **text classification**.
- **Classification and Regression Trees (CART) with Logistic Regression**.
- **Convolutional Neural Networks (CNNs)** for deeper analysis.

Among these, the **Linear SVC model** showed the best test performance with a score of **0.72**, ranking **10th** in the competition.

---

## **Features**
The project includes multiple functionalities to process and classify medical dialogues:
- **Dialogue to Topic Classification** (**Subtask A**): Assigning snippets of **doctor-patient** conversations to predefined medical **section headers**.
- **Preprocessing Pipelines**: Includes **stopword removal**, **lemmatization**, and **TF-IDF vectorization** for improving model performance.
- **Model Implementations**:
  - **Linear SVC with TF-IDF** (best-performing model).
  - **CART with Logistic Regression**.
  - **CNN-based classification** (experimental).
- **Data Balancing**: Uses **Random Oversampling** to address **class imbalance** in training data.

---

## **Dataset**
The dataset consists of **doctor-patient conversations**, where each snippet corresponds to a **section header** (e.g., **Assessment, Diagnosis, Medications, Plan**). 

- **Training set**: **1201** dialogue snippets.
- **Validation set**: **100** dialogues.
- **Test set**: **200** dialogues.
- **20 section headers** are used for classification.

---

## **Installation & Setup**
### **1️⃣ Change File Permissions**
Before running scripts, modify their permissions:
```sh
chmod +x install.sh
chmod +x activate.sh
chmod +x decode_taskA_run1.sh
chmod +x decode_taskA_run2.sh
```

### **2️⃣ Install Required Packages**
Run the installation script:
```sh
./install.sh
```

### **3️⃣ Activate Virtual Environment**
To activate the environment created:
```sh
source activate.sh
```

---

## **Execution Instructions**
### **Run 1 - Linear SVC Model**
To execute **Run 1** (Linear SVC with TF-IDF):
```sh
./decode_taskA_run1.sh path/to/TaskA-TestSet.csv
```
- Uses **TF-IDF feature extraction**.
- Applies **Random Oversampling** for class balancing.
- Achieved **0.72 test accuracy** in **MEDIQA-Sum 2023**.

### **Run 2 - CART with Logistic Regression**
To execute **Run 2** (CART with Logistic Regression):
```sh
./decode_taskA_run2.sh path/to/TaskA-TestSet.csv
```
- Uses **CountVectorizer** for text processing.
- Applies **Lemmatization** for better text representation.
- Implements a **Decision Tree-based approach**.

---

## **Model Implementations**
### **🔹 Run 1 - Linear SVC Model**
**File**: `run1.py`

**Steps**:
1. **Preprocess text**: Remove stopwords, punctuation, digits.
2. **TF-IDF transformation**.
3. **Apply Linear Support Vector Classifier (SVC)**.
4. **Use RandomOversampler** to handle class imbalance.
5. **Predict section headers for test dialogues**.

🔹 **Performance**:
- **Training Accuracy**: 0.99
- **Validation Accuracy**: 0.69
- **Test Accuracy**: 0.72 (Ranked 10th in competition)

---

### **🔹 Run 2 - CART with Logistic Regression**
**File**: `run2.py`

**Steps**:
1. **Apply Lemmatization** using `nltk.WordNetLemmatizer()`.
2. **Use CountVectorizer** to transform text into numerical format.
3. **Train Classification and Regression Trees (CART) with Logistic Regression**.
4. **Predict section headers for test dialogues**.

🔹 **Performance**:
- **Training Accuracy**: 1.0
- **Validation Accuracy**: 0.66
- **Test Accuracy**: 0.69 (Ranked 15th in competition)

---

## **System Requirements**
To run this project, ensure you have:
- **Python 3.7+**
- Required libraries:
  ```sh
  pip install -r requirements.txt
  ```
- **Hardware**:
  - **Intel Core i5** (or equivalent).
  - **8GB RAM**.
  - **SSD recommended** for faster processing.

---

## **Future Improvements**
- **Improve validation accuracy** to prevent overfitting.
- **Explore advanced deep learning models** like Transformers.
- **Implement additional data augmentation techniques** to enhance learning.
- **Fine-tune hyperparameters** to improve performance.

---

## **Workshop Paper**
This project is documented in the **MEDIQA-Sum 2023** workshop paper:

📄 **Title**: SSNdhanyadivyakavitha at MEDIQA-Sum 2023: **Medical Dialogue Summarization using Linear Support Vector Classification Technique**  
📍 **Conference**: **CLEF 2023**, Thessaloniki, Greece  
🔗 **Reference**: [paper-127.pdf](paper-127.pdf)

The paper discusses:
- **Dataset details & preprocessing techniques**.
- **Comparative performance of different models**.
- **Limitations and future scope**.

---

## **Contributors**
👩‍💻 **Dhanya Krishnan**  
👩‍💻 **Divya Srinivasan**  
👩‍💻 **Kavitha Srinivasan**  
🏛 **Sri Sivasubramaniya Nadar College of Engineering, India**

---

## **License**
This project is licensed under the **Creative Commons License Attribution 4.0 (CC BY 4.0)**.

---

## **Contact**
For any queries, feel free to reach out:
📧 **Email**: dhanya2010402@ssn.edu.in

---

