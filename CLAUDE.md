# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a Student Depression ML project, a machine learning application that predicts depression risk in students based on various factors. The project includes both a GUI and CLI interface for working with the model.

## Key Components

1. **Main Application**: `src/depression_ml_gui.py` - Contains two main classes:
   - `DepressionMLProject`: Handles the core ML functionality (data loading, preprocessing, model training, evaluation)
   - `DepressionMLGUI`: Implements the tkinter-based GUI interface

2. **Data**: Located in the `data/` directory
   - `student_depression_dataset.csv`: The main dataset for training models

3. **Outputs**: Directory structure for storing model outputs
   - `models/`: For saved ML models
   - `reports/`: For generated reports
   - `visualizations/`: For plots and visualizations

## Development Commands

### Setup Environment

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run the application (will prompt for GUI or CLI mode)
python src/depression_ml_gui.py

# Run in GUI mode directly
python src/depression_ml_gui.py
# When prompted, enter "gui"

# Run in CLI mode directly
python src/depression_ml_gui.py
# When prompted, enter "cli"
```

## Machine Learning Pipeline

The application implements a complete ML pipeline:

1. **Data Loading**: Loads student data from CSV
2. **Preprocessing**:
   - Handles missing values with median imputation
   - Encodes categorical variables
   - Creates engineered features (Pressure_Score, Satisfaction_Score, Stress_Index)
   - Splits data into training and testing sets
   - Applies SMOTE for class balancing
   - Scales features

3. **Model Training**: Trains multiple models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVM
   - Neural Network (MLP)

4. **Evaluation**: Computes various metrics:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Cross-validation scores
   - Confusion matrices
   - Feature importance

5. **Prediction**: Uses the best model to predict depression risk for new inputs

## GUI Features

The GUI interface has four main tabs:
1. **Data Loading**: For loading and preprocessing data
2. **Model Training**: For selecting and training models
3. **Results**: For visualizing model performance
4. **Make Prediction**: For using the trained model to predict depression risk based on user inputs