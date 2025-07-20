# 🫀 Cardiac Disease Prediction App

Full-stack multi-model web app to predict if a user has cardiac/heart disease or not.

---

## 📝 Overview

This repository contains a web application designed to predict the likelihood of cardiac disease based on various health parameters. It leverages machine learning models to provide an intuitive and accessible tool for preliminary risk assessment.

---

## 🚀 Features

- **User-Friendly Interface**: Simple and intuitive web interface for inputting health data.
- **Machine Learning Powered**: Utilizes a trained machine learning model to predict cardiac disease risk.
- **Real-time Predictions**: Get instant predictions based on the provided inputs.
- **Data Visualization** *(Optional)*: May include basic visualizations of input data or prediction confidence.
- **Responsive Design**: Accessible on various devices (desktop, tablet, mobile).

---

## 🛠️ Technologies Used

**Backend:**
- Python
- Flask *(or Streamlit, depending on implementation)*
- Scikit-learn
- Pandas
- NumPy

**Frontend:**
- HTML5
- CSS3 *(with Tailwind CSS)*
- JavaScript

**Deployment:**
- *(Potentially Heroku, AWS, etc.)*

---

POV:

<img width="1919" height="912" alt="Screenshot 2025-07-20 060158" src="https://github.com/user-attachments/assets/50f7db40-7910-4f58-85bb-adda7f79f21c" />
<img width="1356" height="742" alt="Screenshot 2025-07-20 060303" src="https://github.com/user-attachments/assets/1f8e9c9f-9070-424b-aef8-2c8e5ab57f1f" />
<img width="1920" height="1080" alt="Screenshot (196)" src="https://github.com/user-attachments/assets/aefb0b70-29bd-4275-9a6c-3a47080ec0f9" />

---

## ⚙️ Installation

To set up and run this project locally:

```bash
# Clone the repository
git clone https://github.com/aarizkhanshaikh/cardiac-disease-prediction-app.git
cd cardiac-disease-prediction-app

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Usage

To run the app:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

1. Enter the required health parameters in the form.
2. Click the **Predict** button.
3. View your cardiac disease risk assessment.

---

## 📊 Dataset

This project uses the **UCI Heart Disease Dataset** (or a similar dataset), which includes:

* `age`
* `sex`
* `cp` (chest pain type)
* `trestbps` (resting blood pressure)
* `chol` (serum cholesterol)
* `fbs` (fasting blood sugar)
* `restecg` (resting ECG results)
* `thalach` (max heart rate)
* `exang` (exercise-induced angina)
* `oldpeak` (ST depression)
* `slope`, `ca`, `thal`, and `target`

Dataset Info:

<img width="950" height="818" alt="Screenshot 2025-07-20 033800" src="https://github.com/user-attachments/assets/cc68e611-4506-4b0f-a242-beafead6d033" />
<img width="758" height="593" alt="Screenshot 2025-07-20 033711" src="https://github.com/user-attachments/assets/d95ad165-8e54-44dd-a862-454f50d1f728" />
<img width="758" height="462" alt="Screenshot 2025-07-20 033856" src="https://github.com/user-attachments/assets/75311029-9b69-455d-909c-efa8c13bf619" />
<img width="762" height="467" alt="Screenshot 2025-07-20 033848" src="https://github.com/user-attachments/assets/83a48c92-cfa2-4e05-8aae-6d84b805a157" />


---

## 🧠 Machine Learning Model

The application uses a trained ML model like:

* Logistic Regression
* Support Vector Machine
* Random Forest Classifier

Trained using the `heart_disease_data.csv` dataset.
Training details can be found in:
📁 `notebooks/model_training.ipynb`

---

## 🤝 Contributing

Contributions are welcome! 🚀

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git commit -m 'Add new feature'

# Push to your branch
git push origin feature/your-feature-name

# Create a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## 📧 Contact

**Aariz Khan Shaikh**
GitHub: [aarizkhanshaikh](https://github.com/aarizkhanshaikh)
