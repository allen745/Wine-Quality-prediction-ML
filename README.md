# Wine-Quality-prediction-ML
# 🍷 Wine Quality Prediction

A Machine Learning project that uses **Random Forest Classification** to predict the quality of red wine based on physicochemical properties.

---

## 📌 Project Overview

This project builds a classification model to predict whether a red wine is **good quality** or **bad quality** based on 11 chemical features. It demonstrates a complete ML pipeline including data loading, EDA, feature engineering, model training, and prediction.

---

## 📂 Dataset

- **Source:** [Kaggle — Red Wine Quality (UCI)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **File:** `winequality-red.csv`
- **Samples:** 1,599
- **Features:** 11
- **Target:** `quality` (converted to binary: 1 = Good, 0 = Bad)
- **Missing Values:** None ✅

**Features:**

| Feature | Description |
|---------|-------------|
| fixed acidity | Amount of fixed acids |
| volatile acidity | Amount of volatile acids |
| citric acid | Amount of citric acid |
| residual sugar | Sugar remaining after fermentation |
| chlorides | Salt content |
| free sulfur dioxide | Free form of SO₂ |
| total sulfur dioxide | Total SO₂ |
| density | Density of wine |
| pH | Acidity level |
| sulphates | Sulphate content |
| alcohol | Alcohol percentage |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Programming Language |
| NumPy | Numerical computations |
| Pandas | Data loading and analysis |
| Scikit-learn | ML model, train-test split, accuracy evaluation |
| Random Forest | Classification model |

---

## ⚙️ Project Workflow

1. **Data Loading** → Load red wine CSV using Pandas
2. **EDA** → Shape, statistics, null check
3. **Label Binarization** → Convert quality score to Good (1) / Bad (0)
4. **Feature/Target Split** → X (11 features) and Y (quality)
5. **Train-Test Split** → 80% training / 20% testing
6. **Model Training** → Random Forest Classifier
7. **Evaluation** → Accuracy on test data
8. **Prediction** → Predict quality of new wine sample

---

## 📊 Model Performance

| Dataset | Accuracy |
|---------|----------|
| Testing | **93.4%** |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/allen745/Wine-Quality-Prediction-ML.git
cd Wine-Quality-Prediction-ML
```

**2. Install dependencies**
```bash
pip install numpy pandas scikit-learn
```

**3. Run the script**
```bash
python "Wine Quality prediction.py"
```

---

## 🔍 Sample Prediction

```python
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.80, 10.5)
# Output: IT IS GOOD QUALITY WINE
```

---

## 📁 Repository Structure

```
Wine-Quality-Prediction-ML/
│
├── Wine Quality prediction.py   # Main ML script
├── winequality-red.csv          # Dataset
└── README.md                    # Project documentation
```

---

## 👨‍💻 Author

**Allen Christian** || **Patent Holder**
- 🎓 AI & Data Science Student — A.D. Patel Institute of Technology
- 💼 [LinkedIn](https://www.linkedin.com/in/allen-christian-708545409)
- 🌐 [Portfolio](https://allen745.github.io)
- 🐙 [GitHub](https://github.com/allen745)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
