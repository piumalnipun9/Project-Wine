# Wine Quality Prediction System

This repository contains the code and resources for the **COM 763 Advanced Machine Learning** Portfolio Task. It implements a complete end-to-end machine learning pipeline to predict wine quality based on physicochemical properties, utilizing a Random Forest Classifier.

## Project Structure
* `app.py`: The Streamlit web application.
* `src/wine_quality_analysis.py`: The main Python script that handles data loading, preprocessing, model training, evaluation, and saving the artifacts.
* `models/`: Directory where the trained model (`random_forest_model.joblib`) and scaler (`scaler.joblib`) are saved.
* `data/`: Contains the `WineQT.csv` dataset.
* `report_task1/`: Contains the LaTeX source and figures for Portfolio Task 1.
* `report_task2.tex`: The LaTeX source for Portfolio Task 2 (Model Card).
* `requirements.txt`: Python dependencies required to run the project.

---

## 1. Local Setup

Before running the scripts or the application, you need to install the required Python dependencies.

1. **Clone the repository** (or navigate to your project directory):
   ```bash
   cd Proect-Wine
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. How to Train the Model

The model training process is fully automated in the `wine_quality_analysis.py` script. Running this script will process the data, train both the Logistic Regression and Random Forest models, generate all evaluation figures, and save the best model.

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the analysis script:
   ```bash
   python wine_quality_analysis.py
   ```

**What this does:**
* Loads data from `../data/WineQT.csv`.
* Scales features and splits data.
* Performs Hyperparameter tuning via GridSearchCV for the Random Forest.
* Saves evaluation plots (Confusion Matrices, ROC curves, Feature Importance) to `../report_task1/figures/`.
* Saves the `scaler.joblib` and `random_forest_model.joblib` into the `../models/` directory.

---

## 3. How to Run the App Locally

Once the models are trained and saved in the `models/` folder, you can start the Streamlit web interface.

1. Ensure you are in the root directory of the project (`Proect-Wine/`).
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. A browser window will automatically open (usually at `http://localhost:8501`) displaying the interactive Wine Quality Predictor.

---

## 4. How to Host on Streamlit Community Cloud

Hosting the app on Streamlit Community Cloud is free and allows anyone to access your model via a public URL.

### Step A: Push code to GitHub
Make sure your complete project is pushed to a public GitHub repository. You must include:
* `app.py`
* `requirements.txt`
* `models/` (containing the `.joblib` files)
* `data/`

To push your latest changes:
```bash
git add .
git commit -m "Prepare repository for Streamlit deployment"
git push origin main
```

### Step B: Deploy on Streamlit
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account.
2. Click the **"New app"** button.
3. In the "Deploy an app" form, select:
   * **Repository:** Your repository (e.g., `piumalnipun9/Project-Wine`)
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **"Deploy!"**

Streamlit will automatically install the packages listed in `requirements.txt` and launch your app. You will receive a public URL (e.g., `https://your-app-name.streamlit.app`) which you can share and include in your assignment reports!
