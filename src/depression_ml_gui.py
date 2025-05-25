# Import necessary libraries for data manipulation and analysis
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For enhanced statistical visualizations
# Import machine learning model selection and evaluation tools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # For splitting data and model evaluation
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For feature scaling and encoding categorical variables
# Import machine learning classification models
from sklearn.linear_model import LogisticRegression  # Linear classifier with logistic function
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Tree-based ensemble methods
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron neural network
# Import metrics for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer  # For handling missing values
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages

# Import libraries for GUI implementation
import tkinter as tk  # Main GUI toolkit
from tkinter import ttk, filedialog, messagebox  # Advanced widgets and dialogs
import threading  # For running model training in background thread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding matplotlib plots in tkinter

class DepressionMLProject:
    """Main class that handles all the machine learning functionality for the student depression prediction project"""
    def __init__(self):
        """Initialize the project with empty attributes for data and models"""
        self.df = None  # Will hold the original dataset
        self.X_train = None  # Will hold training features
        self.X_test = None  # Will hold testing features
        self.y_train = None  # Will hold training target
        self.y_test = None  # Will hold testing target
        self.models = {}  # Dictionary to store all trained models
        self.results = {}  # Dictionary to store all model evaluation results
        self.scaler = StandardScaler()  # Initialize scaler for feature normalization
        self.label_encoders = {}  # Dictionary to store label encoders for categorical variables
        
    def load_data(self, filepath):
        """Load and perform initial data exploration on the dataset
        
        Args:
            filepath: Path to the CSV dataset file
            
        Returns:
            pandas DataFrame: The loaded dataset
        """
        # Load the dataset from CSV file
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print("\nDataset Info:")
        print(self.df.info())  # Display basic info about the dataset columns
        print("\nTarget Distribution:")
        print(self.df['Depression'].value_counts())  # Show class distribution of target variable
        return self.df
    
    def preprocess_data(self):
        """Comprehensive data preprocessing including handling missing values, encoding categorical variables,
        and creating engineered features
        
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        # Create a copy of the dataframe to avoid modifying the original
        df_processed = self.df.copy()
        
        # Define numeric columns that may need imputation
        numeric_columns = ['Age', 'CGPA', 'Academic Pressure', 'Work Pressure', 
                          'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours']
        
        # Impute missing values in numeric columns using median imputation
        for col in numeric_columns:
            if col in df_processed.columns:
                imputer = SimpleImputer(strategy='median')  # Use median to handle outliers better than mean
                df_processed[col] = imputer.fit_transform(df_processed[[col]])  # Apply imputation
        
        # Define categorical columns that need encoding
        categorical_columns = ['Gender', 'City', 'Profession', 'Sleep Duration', 
                              'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                              'Financial Stress', 'Family History of Mental Illness']
        
        # Encode categorical variables using Label Encoding (convert to numerical values)
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()  # Initialize a label encoder for each categorical variable
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))  # Transform categorical values to numbers
                self.label_encoders[col] = le  # Store encoder for later use in prediction
        
        # Feature engineering: Create interaction features to capture relationships between variables
        # Combine academic and work pressure into a total pressure score
        df_processed['Pressure_Score'] = df_processed['Academic Pressure'] + df_processed['Work Pressure']
        # Combine study and job satisfaction into a total satisfaction score
        df_processed['Satisfaction_Score'] = df_processed['Study Satisfaction'] + df_processed['Job Satisfaction']
        # Create a comprehensive stress index from academic, work pressure and financial stress
        df_processed['Stress_Index'] = (df_processed['Academic Pressure'] + 
                                       df_processed['Work Pressure'] + 
                                       df_processed['Financial Stress']) / 3
        
        # Separate features (X) and target variable (y)
        X = df_processed.drop(['Depression', 'id'], axis=1)  # Remove target and ID columns
        y = df_processed['Depression']  # Extract target variable
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, use_smote=True):
        """Split data into training and testing sets and apply feature scaling
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing (default 0.2 = 20%)
            use_smote: Whether to apply SMOTE for class balancing (default True)
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        # Split data into training and testing sets with stratification to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply SMOTE (Synthetic Minority Over-sampling Technique) if requested and if class imbalance exists
        # SMOTE creates synthetic samples of the minority class to balance class distribution
        if use_smote and y_train.value_counts().min() / y_train.value_counts().max() < 0.8:  # If imbalance ratio < 0.8
            smote = SMOTE(random_state=42)  # Initialize SMOTE with fixed random seed for reproducibility
            X_train, y_train = smote.fit_resample(X_train, y_train)  # Apply SMOTE to create balanced dataset
            print("SMOTE applied to balance classes")
        
        # Scale features to have zero mean and unit variance (standardization)
        # This is important for models like SVM, Neural Networks, and Logistic Regression
        X_train_scaled = self.scaler.fit_transform(X_train)  # Fit scaler on training data and transform
        X_test_scaled = self.scaler.transform(X_test)  # Transform test data using the fitted scaler
        
        # Store the processed data as instance variables
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize all ML models with optimal parameters
        
        This method creates instances of five different machine learning models:
        1. Logistic Regression: A linear model that uses the logistic function to model binary outcomes
        2. Random Forest: An ensemble of decision trees using bagging (bootstrap aggregating)
        3. Gradient Boosting: An ensemble of decision trees using sequential, additive learning
        4. SVM: Support Vector Machine that finds an optimal hyperplane to separate classes
        5. Neural Network: Multi-layer perceptron with two hidden layers
        """
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),  
            # Logistic Regression: Linear model with logistic function for binary classification
            # max_iter=1000: Maximum number of iterations for convergence (default is 100)
            
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            # Random Forest: Ensemble of decision trees using bagging (bootstrap aggregating)
            # n_estimators=100: Number of trees in the forest
            # Each tree is built on a random subset of samples and features
            
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            # Gradient Boosting: Sequential ensemble method that builds trees to correct errors
            # n_estimators=100: Number of boosting stages (trees)
            # Each tree focuses on correcting the errors made by previous trees
            
            'SVM': SVC(probability=True, random_state=42),
            # Support Vector Machine: Finds optimal hyperplane to separate classes
            # probability=True: Enable probability estimates (needed for ROC-AUC)
            # Uses kernel trick to handle non-linear decision boundaries
            
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            # Multi-layer Perceptron: Neural network with multiple layers
            # hidden_layer_sizes=(100, 50): Two hidden layers with 100 and 50 neurons respectively
            # Uses backpropagation to learn the weights between neurons
        }
    
    def train_single_model(self, model_name, model, use_grid_search=False):
        """Train a single model with optional hyperparameter tuning using grid search
        
        Args:
            model_name: Name of the model (string)
            model: The model instance to train
            use_grid_search: Whether to perform grid search for hyperparameter tuning (default False)
            
        Returns:
            dict: Dictionary containing all evaluation metrics for the trained model
        """
        print(f"\nTraining {model_name}...")
        
        # Apply grid search for hyperparameter tuning if requested (currently only for Random Forest)
        if use_grid_search and model_name == 'Random Forest':
            # Define parameter grid to search
            param_grid = {
                'n_estimators': [50, 100, 200],  # Number of trees to try
                'max_depth': [10, 20, None],  # Maximum depth of trees (None = unlimited)
                'min_samples_split': [2, 5, 10]  # Minimum samples required to split a node
            }
            # Initialize grid search with 5-fold cross-validation and ROC-AUC scoring
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            # Fit grid search to find best parameters
            grid_search.fit(self.X_train, self.y_train)
            # Get the model with best parameters
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # If not using grid search, use the model as is
            best_model = model
            # Fit the model on the training data
            best_model.fit(self.X_train, self.y_train)
        
        # Make predictions on test set
        y_pred = best_model.predict(self.X_test)  # Class predictions (0 or 1)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]  # Probability of class 1 (Depression)
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),  # Proportion of correct predictions
            'precision': precision_score(self.y_test, y_pred),  # True positives / (True positives + False positives)
            'recall': recall_score(self.y_test, y_pred),  # True positives / (True positives + False negatives)
            'f1_score': f1_score(self.y_test, y_pred),  # Harmonic mean of precision and recall
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),  # Area under ROC curve
            'y_pred': y_pred,  # Store predictions for later analysis
            'y_pred_proba': y_pred_proba,  # Store probability predictions for ROC curve
            'model': best_model  # Store the trained model
        }
        
        # Perform 5-fold cross-validation to get more robust performance estimate
        cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()  # Mean of cross-validation scores
        metrics['cv_std'] = cv_scores.std()  # Standard deviation of cross-validation scores
        
        return metrics
    
    def train_all_models(self, use_grid_search=False):
        """Train all models and store results
        
        Args:
            use_grid_search: Whether to perform grid search for hyperparameter tuning (default False)
            
        Returns:
            dict: Dictionary containing results for all models
        """
        # Initialize all models with default parameters
        self.initialize_models()
        
        # Train each model and store its results
        for model_name, model in self.models.items():
            self.results[model_name] = self.train_single_model(model_name, model, use_grid_search)
        
        return self.results
    
    def plot_results(self):
        """Create comprehensive visualization of model performance results
        
        Returns:
            matplotlib.figure.Figure: Figure containing all plots
        """
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison Bar Chart - compares all evaluation metrics across models
        ax1 = axes[0, 0]
        # Create a DataFrame with all metrics for each model
        metrics_df = pd.DataFrame({
            model: [results['accuracy'], results['precision'], results['recall'], 
                   results['f1_score'], results['roc_auc']]
            for model, results in self.results.items()
        })
        metrics_df.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        # Plot metrics as grouped bar chart
        metrics_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 1)  # Set y-axis from 0 to 1 for consistent scale
        
        # 2. ROC Curves - shows trade-off between true positive rate and false positive rate
        ax2 = axes[0, 1]
        for model_name, results in self.results.items():
            # Calculate ROC curve points
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            # Plot ROC curve with AUC in the legend
            ax2.plot(fpr, tpr, label=f"{model_name} (AUC={results['roc_auc']:.3f})")
        # Add diagonal reference line (random classifier)
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves Comparison')
        ax2.legend()
        
        # 3. Best Model Confusion Matrix - shows true/false positives/negatives
        ax3 = axes[1, 0]
        # Find the model with highest ROC-AUC score
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        # Generate confusion matrix for best model
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['y_pred'])
        # Plot confusion matrix as heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model_name}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Feature Importance (for Random Forest model)
        ax4 = axes[1, 1]
        if 'Random Forest' in self.results:
            # Get the trained Random Forest model
            rf_model = self.results['Random Forest']['model']
            # Create generic feature names (actual names not preserved in scaled data)
            feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
            # Get feature importance scores
            importances = rf_model.feature_importances_
            # Sort features by importance (descending) and take top 10
            indices = np.argsort(importances)[::-1][:10]
            
            # Plot feature importance as bar chart
            ax4.bar(range(10), importances[indices])
            ax4.set_xticks(range(10))
            ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45)
            ax4.set_title('Top 10 Feature Importances (Random Forest)')
            ax4.set_ylabel('Importance')
        
        plt.tight_layout()  # Adjust subplot layout
        return fig
    
    def generate_report(self):
        """Generate comprehensive report on model performance
        
        Returns:
            str: Multi-line report text
        """
        # Initialize report as list of lines
        report = []
        report.append("=" * 80)
        report.append("STUDENT DEPRESSION PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 80)
        
        # Dataset overview section
        report.append("\n1. DATASET OVERVIEW")
        report.append(f"   - Total samples: {len(self.df)}")
        report.append(f"   - Features: {self.X_train.shape[1]}")
        report.append(f"   - Training samples: {len(self.X_train)}")
        report.append(f"   - Testing samples: {len(self.X_test)}")
        report.append(f"   - Class distribution in test set:")
        report.append(f"     No Depression: {(self.y_test == 0).sum()}")
        report.append(f"     Depression: {(self.y_test == 1).sum()}")
        
        # Model performance summary section - tabular format
        report.append("\n2. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        report.append("-" * 80)
        
        # Add a row for each model's performance metrics
        for model_name, results in self.results.items():
            report.append(f"{model_name:<20} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
                         f"{results['recall']:<10.3f} {results['f1_score']:<10.3f} {results['roc_auc']:<10.3f}")
        
        # Best model section
        best_model = max(self.results, key=lambda x: self.results[x]['roc_auc'])  # Model with highest ROC-AUC
        report.append("\n3. BEST PERFORMING MODEL")
        report.append(f"   {best_model} with ROC-AUC score of {self.results[best_model]['roc_auc']:.3f}")
        
        # Detailed classification report for best model
        report.append("\n4. DETAILED CLASSIFICATION REPORT - " + best_model)
        report.append(classification_report(self.y_test, self.results[best_model]['y_pred'],
                                          target_names=['No Depression', 'Depression']))
        
        return '\n'.join(report)  # Join all lines into a single string


class DepressionMLGUI:
    """GUI class for the depression prediction application"""
    def __init__(self, root):
        """Initialize the GUI with the given root window
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title("Student Depression ML Classifier")  # Set window title
        self.root.geometry("1200x800")  # Set window size
        
        # Initialize the ML project backend
        self.ml_project = DepressionMLProject()
        # Setup the GUI components
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI components and tab structure"""
        # Create notebook (tabbed interface) for organizing GUI sections
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True)
        
        # Tab 1: Data Loading and Preprocessing
        self.data_tab = ttk.Frame(notebook)
        notebook.add(self.data_tab, text='Data Loading')
        self.setup_data_tab()
        
        # Tab 2: Model Training
        self.training_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text='Model Training')
        self.setup_training_tab()
        
        # Tab 3: Results Visualization
        self.results_tab = ttk.Frame(notebook)
        notebook.add(self.results_tab, text='Results')
        self.setup_results_tab()
        
        # Tab 4: Prediction
        self.prediction_tab = ttk.Frame(notebook)
        notebook.add(self.prediction_tab, text='Make Prediction')
        self.setup_prediction_tab()
        
    def setup_data_tab(self):
        """Setup the data loading and preprocessing tab interface"""
        # File selection frame - for loading the dataset
        file_frame = ttk.LabelFrame(self.data_tab, text="Load Dataset", padding=10)
        file_frame.pack(fill='x', padx=10, pady=10)
        
        # File path entry and browse button
        self.file_path_var = tk.StringVar()  # Variable to hold file path
        ttk.Label(file_frame, text="Dataset Path:").grid(row=0, column=0, sticky='w')
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        ttk.Button(file_frame, text="Load Data", command=self.load_data).grid(row=0, column=3, padx=5)
        
        # Data info frame - displays dataset information
        info_frame = ttk.LabelFrame(self.data_tab, text="Dataset Information", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Text widget to display dataset information
        self.data_info_text = tk.Text(info_frame, height=20, width=80)
        self.data_info_text.pack(fill='both', expand=True)
        
        # Preprocessing options frame - for setting preprocessing parameters
        prep_frame = ttk.LabelFrame(self.data_tab, text="Preprocessing Options", padding=10)
        prep_frame.pack(fill='x', padx=10, pady=10)
        
        # SMOTE option for class balancing
        self.use_smote_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep_frame, text="Use SMOTE for class balancing", 
                       variable=self.use_smote_var).pack(anchor='w')
        
        # Test size slider (determines train/test split ratio)
        self.test_size_var = tk.DoubleVar(value=0.2)  # Default to 20% test size
        ttk.Label(prep_frame, text="Test Size:").pack(side='left', padx=5)
        ttk.Scale(prep_frame, from_=0.1, to=0.4, variable=self.test_size_var, 
                 orient='horizontal', length=200).pack(side='left', padx=5)
        ttk.Label(prep_frame, textvariable=self.test_size_var).pack(side='left')
        
        # Preprocess button
        ttk.Button(prep_frame, text="Preprocess Data", 
                  command=self.preprocess_data).pack(side='right', padx=5)
        
    def setup_training_tab(self):
        """Setup the model training tab interface"""
        # Model selection frame - checkboxes for selecting models to train
        model_frame = ttk.LabelFrame(self.training_tab, text="Model Selection", padding=10)
        model_frame.pack(fill='x', padx=10, pady=10)
        
        # Create checkboxes for each available model
        self.selected_models = {}  # Dictionary to hold checkbox variables
        models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network']
        
        for i, model in enumerate(models):
            var = tk.BooleanVar(value=True)  # Default to selected
            self.selected_models[model] = var
            ttk.Checkbutton(model_frame, text=model, variable=var).grid(row=i//3, column=i%3, sticky='w', padx=10, pady=5)
        
        # Training options frame
        options_frame = ttk.LabelFrame(self.training_tab, text="Training Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=10)
        
        # Grid search option for hyperparameter tuning
        self.use_grid_search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use Grid Search for hyperparameter tuning", 
                       variable=self.use_grid_search_var).pack(anchor='w')
        
        # Start training button
        ttk.Button(options_frame, text="Start Training", 
                  command=self.start_training).pack(pady=10)
        
        # Progress frame - shows training progress
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Text widget for displaying training progress messages
        self.progress_text = tk.Text(progress_frame, height=15, width=80)
        self.progress_text.pack(fill='both', expand=True)
        
        # Progress bar for visual indication of training progress
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
    def setup_results_tab(self):
        """Setup the results visualization tab interface"""
        # Results summary frame - displays text summary of results
        summary_frame = ttk.LabelFrame(self.results_tab, text="Results Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=10)
        
        # Text widget for displaying results summary
        self.results_text = tk.Text(summary_frame, height=10, width=80)
        self.results_text.pack(fill='x')
        
        # Visualization frame - contains plots and charts
        viz_frame = ttk.LabelFrame(self.results_tab, text="Visualizations", padding=10)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Buttons for different visualization actions
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill='x')
        
        # Show plots button
        ttk.Button(button_frame, text="Show All Plots", 
                  command=self.show_plots).pack(side='left', padx=5)
        # Generate report button
        ttk.Button(button_frame, text="Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        # Export results button
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        
        # Frame for displaying matplotlib plots
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill='both', expand=True)
        
    def setup_prediction_tab(self):
        """Setup the prediction tab interface for making predictions on new data"""
        # Instruction label
        instruction_label = ttk.Label(self.prediction_tab, 
                                    text="Enter values for all features to predict depression likelihood:",
                                    font=('Arial', 12, 'bold'))
        instruction_label.pack(pady=10)
        
        # Create scrollable frame for input fields (many features require scrolling)
        canvas = tk.Canvas(self.prediction_tab)
        scrollbar = ttk.Scrollbar(self.prediction_tab, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)  # Store reference
        
        # Configure scrolling behavior
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Dictionaries to store input widgets and their variables
        self.prediction_inputs = {}  # Store input variables
        self.prediction_widgets = {}  # Store widget references
        
        # Define feature types and options for each input field
        feature_configs = {
            'Gender': {'type': 'dropdown', 'options': ['Male', 'Female']},
            'Age': {'type': 'spinbox', 'from': 16, 'to': 35, 'default': 20},
            'City': {'type': 'dropdown', 'options': []},  # Will be populated from data
            'Profession': {'type': 'dropdown', 'options': []},  # Will be populated from data
            'Academic Pressure': {'type': 'scale', 'from': 1, 'to': 5, 'default': 3},
            'Work Pressure': {'type': 'scale', 'from': 1, 'to': 5, 'default': 3},
            'CGPA': {'type': 'spinbox', 'from': 0.0, 'to': 10.0, 'increment': 0.1, 'default': 7.5},
            'Study Satisfaction': {'type': 'scale', 'from': 1, 'to': 5, 'default': 3},
            'Job Satisfaction': {'type': 'scale', 'from': 1, 'to': 5, 'default': 3},
            'Sleep Duration': {'type': 'dropdown', 'options': ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours']},
            'Dietary Habits': {'type': 'dropdown', 'options': ['Healthy', 'Moderate', 'Unhealthy']},
            'Degree': {'type': 'dropdown', 'options': []},  # Will be populated from data
            'Have you ever had suicidal thoughts ?': {'type': 'radio', 'options': ['Yes', 'No']},
            'Work/Study Hours': {'type': 'spinbox', 'from': 0, 'to': 24, 'default': 8},
            'Financial Stress': {'type': 'radio', 'options': ['Yes', 'No']},
            'Family History of Mental Illness': {'type': 'radio', 'options': ['Yes', 'No']}
        }
        
        # Create appropriate widgets for each feature based on its type
        row = 0
        for feature, config in feature_configs.items():
            # Label for the feature
            label = ttk.Label(self.scrollable_frame, text=f"{feature}:")
            label.grid(row=row, column=0, sticky='w', padx=10, pady=5)
            
            # Create different widget types based on feature configuration
            if config['type'] == 'dropdown':
                # Dropdown menu (combobox) for categorical features with fixed options
                var = tk.StringVar()
                if config['options']:  # If options are provided
                    var.set(config['options'][0])  # Set default to first option
                widget = ttk.Combobox(self.scrollable_frame, textvariable=var, 
                                    values=config['options'], state='readonly', width=27)
                widget.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                self.prediction_inputs[feature] = var
                self.prediction_widgets[feature] = widget  # Store widget reference
                
            elif config['type'] == 'spinbox':
                # Spinbox for numerical inputs with specific ranges
                if feature == 'CGPA':  # Special case for CGPA which is a float
                    var = tk.DoubleVar(value=config.get('default', 0))
                    widget = ttk.Spinbox(self.scrollable_frame, from_=config['from'], to=config['to'],
                                       increment=config.get('increment', 1), textvariable=var,
                                       width=28, format="%.1f")
                else:  # Integer spinbox for other numerical features
                    var = tk.IntVar(value=config.get('default', 0))
                    widget = ttk.Spinbox(self.scrollable_frame, from_=config['from'], to=config['to'],
                                       textvariable=var, width=28)
                widget.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                self.prediction_inputs[feature] = var
                
            elif config['type'] == 'scale':
                # Slider (scale) for rating inputs (1-5)
                var = tk.IntVar(value=config.get('default', 3))
                frame = ttk.Frame(self.scrollable_frame)
                frame.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                
                # Create scale widget with label showing current value
                scale = ttk.Scale(frame, from_=config['from'], to=config['to'],
                                variable=var, orient='horizontal', length=200)
                scale.pack(side='left')
                
                value_label = ttk.Label(frame, textvariable=var, width=3)
                value_label.pack(side='left', padx=5)
                
                self.prediction_inputs[feature] = var
                
            elif config['type'] == 'radio':
                # Radio buttons for binary choices (Yes/No)
                var = tk.StringVar(value=config['options'][1])  # Default to 'No'
                frame = ttk.Frame(self.scrollable_frame)
                frame.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                
                # Create a radio button for each option
                for option in config['options']:
                    rb = ttk.Radiobutton(frame, text=option, variable=var, value=option)
                    rb.pack(side='left', padx=10)
                
                self.prediction_inputs[feature] = var
            
            row += 1
        
        # Add help text for understanding feature inputs
        help_frame = ttk.LabelFrame(self.scrollable_frame, text="Feature Guidelines", padding=5)
        help_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        help_text = """• Academic/Work Pressure: 1 (Low) to 5 (High)
• Study/Job Satisfaction: 1 (Very Unsatisfied) to 5 (Very Satisfied)
• CGPA: 0.0 to 10.0
• Work/Study Hours: Daily hours spent working/studying"""
        
        ttk.Label(help_frame, text=help_text, justify='left').pack(anchor='w')
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Prediction button and result frame
        pred_frame = ttk.Frame(self.prediction_tab)
        pred_frame.pack(fill='x', padx=10, pady=10)
        
        # Model info frame - shows which model is being used for prediction
        model_info_frame = ttk.LabelFrame(pred_frame, text="Model Information", padding=10)
        model_info_frame.pack(fill='x', padx=20, pady=5)
        
        self.model_info_var = tk.StringVar(value="No models trained yet")
        model_info_label = ttk.Label(model_info_frame, textvariable=self.model_info_var,
                                   font=('Arial', 11))
        model_info_label.pack()
        
        # Update model info to show current best model
        self.update_model_info()
        
        # Button frame for prediction actions
        button_frame = ttk.Frame(pred_frame)
        button_frame.pack(pady=10)
        
        # Predict button
        ttk.Button(button_frame, text="Predict Depression Risk", 
                  command=self.make_prediction, style='Accent.TButton').pack(side='left', padx=5)
        
        # Reset form button
        ttk.Button(button_frame, text="Reset Form", 
                  command=self.reset_prediction_form).pack(side='left', padx=5)
        
        # Result frame - displays prediction result
        result_frame = ttk.LabelFrame(pred_frame, text="Prediction Result", padding=10)
        result_frame.pack(fill='x', padx=20, pady=10)
        
        # Variable and label for displaying prediction result
        self.prediction_result = tk.StringVar()
        result_label = ttk.Label(result_frame, textvariable=self.prediction_result, 
                               font=('Arial', 14, 'bold'))
        result_label.pack(pady=10)
        
    def update_model_info(self):
        """Update the model info display in prediction tab to show which model is being used"""
        if hasattr(self, 'model_info_var') and self.ml_project.results:
            # Find the best model based on ROC-AUC score
            best_model_name = max(self.ml_project.results, 
                                key=lambda x: self.ml_project.results[x]['roc_auc'])
            best_score = self.ml_project.results[best_model_name]['roc_auc']
            # Update display text
            info_text = f"Using: {best_model_name} (ROC-AUC: {best_score:.3f})"
            self.model_info_var.set(info_text)
    
    def reset_prediction_form(self):
        """Reset all prediction form fields to default values"""
        # Default values for various features
        defaults = {
            'Gender': 'Male',
            'Age': 20,
            'Academic Pressure': 3,
            'Work Pressure': 3,
            'CGPA': 7.5,
            'Study Satisfaction': 3,
            'Job Satisfaction': 3,
            'Work/Study Hours': 8,
            'Have you ever had suicidal thoughts ?': 'No',
            'Financial Stress': 'No',
            'Family History of Mental Illness': 'No'
        }
        
        # Reset each input to its default value
        for feature, var in self.prediction_inputs.items():
            if feature in defaults:
                var.set(defaults[feature])
            elif hasattr(var, 'get') and isinstance(var.get(), str):
                # For dropdown boxes, set to first option if available
                if hasattr(var, '_values') and var._values:
                    var.set(var._values[0])
        
        # Clear any previous prediction result
        self.prediction_result.set("")
        
    def browse_file(self):
        """Open file dialog to browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)  # Update file path entry
    
    def load_data(self):
        """Load dataset and display information about it"""
        filepath = self.file_path_var.get()
        if not filepath:
            messagebox.showerror("Error", "Please select a dataset file")
            return
        
        try:
            # Load the dataset using the ML project backend
            df = self.ml_project.load_data(filepath)
            
            # Display dataset information in the text widget
            info_text = f"Dataset loaded successfully!\n\n"
            info_text += f"Shape: {df.shape}\n\n"  # Number of rows and columns
            info_text += f"Columns:\n{', '.join(df.columns)}\n\n"  # Column names
            info_text += f"Data types:\n{df.dtypes}\n\n"  # Data types of each column
            info_text += f"Missing values:\n{df.isnull().sum()}\n\n"  # Count of missing values per column
            info_text += f"Target distribution:\n{df['Depression'].value_counts()}"  # Class distribution
            
            # Update text widget with dataset info
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info_text)
            
            # Update prediction tab dropdowns with actual data values
            self.update_prediction_dropdowns()
            
            messagebox.showinfo("Success", "Dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def update_prediction_dropdowns(self):
        """Update dropdown values in prediction tab with actual data from dataset"""
        if self.ml_project.df is None or not hasattr(self, 'prediction_widgets'):
            return
        
        df = self.ml_project.df
        
        # Update City dropdown with unique values from dataset
        if 'City' in df.columns and 'City' in self.prediction_widgets:
            cities = sorted(df['City'].dropna().unique().tolist())
            self.prediction_widgets['City']['values'] = cities
            if cities:
                self.prediction_inputs['City'].set(cities[0])
        
        # Update Profession dropdown with unique values from dataset
        if 'Profession' in df.columns and 'Profession' in self.prediction_widgets:
            professions = sorted(df['Profession'].dropna().unique().tolist())
            self.prediction_widgets['Profession']['values'] = professions
            if professions:
                self.prediction_inputs['Profession'].set(professions[0])
        
        # Update Degree dropdown with unique values from dataset
        if 'Degree' in df.columns and 'Degree' in self.prediction_widgets:
            degrees = sorted(df['Degree'].dropna().unique().tolist())
            self.prediction_widgets['Degree']['values'] = degrees
            if degrees:
                self.prediction_inputs['Degree'].set(degrees[0])
        
        # Log the number of unique values found for each dropdown
        print(f"Updated dropdowns - Cities: {len(cities) if 'cities' in locals() else 0}, "
              f"Professions: {len(professions) if 'professions' in locals() else 0}, "
              f"Degrees: {len(degrees) if 'degrees' in locals() else 0}")
    
    def preprocess_data(self):
        """Preprocess the loaded data with selected options"""
        if self.ml_project.df is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
        
        try:
            # Run preprocessing steps from ML project backend
            X, y = self.ml_project.preprocess_data()  # Preprocess the data
            test_size = self.test_size_var.get()  # Get selected test size
            use_smote = self.use_smote_var.get()  # Get SMOTE option
            
            # Split and scale the data
            self.ml_project.split_and_scale_data(X, y, test_size=test_size, use_smote=use_smote)
            
            messagebox.showinfo("Success", "Data preprocessing completed!")
            
            # Update info text with preprocessing results
            info_text = self.data_info_text.get(1.0, tk.END)
            info_text += f"\n\nPreprocessing completed!"
            info_text += f"\nTraining samples: {len(self.ml_project.X_train)}"
            info_text += f"\nTesting samples: {len(self.ml_project.X_test)}"
            info_text += f"\nFeatures after preprocessing: {self.ml_project.X_train.shape[1]}"
            
            # Update text widget
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess data: {str(e)}")
    
    def start_training(self):
        """Start model training in a separate thread to keep GUI responsive"""
        if self.ml_project.X_train is None:
            messagebox.showerror("Error", "Please preprocess the data first")
            return
        
        # Clear previous progress
        self.progress_text.delete(1.0, tk.END)
        self.progress_bar.start()  # Start progress bar animation
        
        # Run training in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.train_models)
        thread.start()
    
    def train_models(self):
        """Train selected models - runs in a separate thread"""
        try:
            # Get list of selected models to train
            selected = [name for name, var in self.selected_models.items() if var.get()]
            
            if not selected:
                # Handle case where no models are selected
                self.root.after(0, lambda: messagebox.showerror("Error", "Please select at least one model"))
                return
            
            # Initialize only the selected models to save time
            self.ml_project.models = {}
            
            # Define all available models with their parameters
            all_models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                # Logistic Regression: Uses logistic function to model probability of binary outcome
                # max_iter=1000: Maximum number of iterations for solver convergence
                
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                # Random Forest: Ensemble method that builds multiple decision trees
                # n_estimators=100: Number of trees in the forest
                # Uses bootstrap sampling and feature randomness for diversity
                
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                # Gradient Boosting: Sequential ensemble that builds trees to correct errors
                # n_estimators=100: Number of boosting stages
                # Minimizes loss function by adding trees that correct previous errors
                
                'SVM': SVC(probability=True, random_state=42),
                # Support Vector Machine: Finds optimal hyperplane to separate classes
                # probability=True: Enable probability estimates for ROC curve
                # Uses kernel trick to handle non-linear classification problems
                
                'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                # Multi-layer Perceptron: Neural network with multiple layers
                # hidden_layer_sizes=(100, 50): Two hidden layers with 100 and 50 neurons
                # Uses backpropagation algorithm to update weights during training
            }
            
            # Add only selected models to the training list
            for model_name in selected:
                self.ml_project.models[model_name] = all_models[model_name]
            
            # Get grid search option
            use_grid = self.use_grid_search_var.get()
            
            # Train each selected model
            for model_name, model in self.ml_project.models.items():
                # Update progress text (using after to safely update from thread)
                self.root.after(0, lambda mn=model_name: self.progress_text.insert(tk.END, f"\nTraining {mn}...\n"))
                
                # Train the model and get results
                results = self.ml_project.train_single_model(model_name, model, use_grid)
                self.ml_project.results[model_name] = results
                
                # Update progress with model results
                progress_msg = f"Completed {model_name}:\n"
                progress_msg += f"  Accuracy: {results['accuracy']:.3f}\n"
                progress_msg += f"  ROC-AUC: {results['roc_auc']:.3f}\n"
                
                self.root.after(0, lambda msg=progress_msg: self.progress_text.insert(tk.END, msg))
            
            # Call training_completed on the main thread when done
            self.root.after(0, self.training_completed)
            
        except Exception as e:
            # Handle exceptions that occur during training
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
        finally:
            # Stop progress bar animation in main thread
            self.root.after(0, self.progress_bar.stop)
    
    def training_completed(self):
        """Called when training is completed to update UI"""
        self.progress_bar.stop()  # Stop progress bar animation
        messagebox.showinfo("Success", "Model training completed!")
        
        # Display results summary report
        report = self.ml_project.generate_report()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, report)
        
        # Update model info in prediction tab to show best model
        self.update_model_info()
    
    def show_plots(self):
        """Display all visualization plots"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Clear any previous plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Create new plots
            fig = self.ml_project.plot_results()
            
            # Embed matplotlib figure in tkinter using FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plots: {str(e)}")
    
    def generate_report(self):
        """Generate and save detailed report to file"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Generate the report text
            report = self.ml_project.generate_report()
            
            # Ask user where to save the report
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                # Write report to the selected file
                with open(filename, 'w') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Report saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def export_results(self):
        """Export model comparison results to CSV file"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Create a list of dictionaries with model results
            results_data = []
            for model_name, results in self.ml_project.results.items():
                results_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'ROC-AUC': results['roc_auc'],
                    'CV Mean': results.get('cv_mean', 'N/A'),  # Cross-validation mean score
                    'CV Std': results.get('cv_std', 'N/A')  # Cross-validation standard deviation
                })
            
            # Convert to DataFrame for easy CSV export
            results_df = pd.DataFrame(results_data)
            
            # Ask user where to save the CSV file
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Export DataFrame to CSV
                results_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def make_prediction(self):
        """Make depression prediction using the best trained model"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Get the best model based on ROC-AUC score
            best_model_name = max(self.ml_project.results, 
                                key=lambda x: self.ml_project.results[x]['roc_auc'])
            best_model = self.ml_project.results[best_model_name]['model']
            
            # Prepare input data from form fields
            input_data = {}
            for feature, var in self.prediction_inputs.items():
                value = var.get()
                
                # Convert to appropriate type based on feature
                if feature in ['Age', 'Work/Study Hours']:
                    input_data[feature] = float(value)  # Convert to float
                elif feature in ['Academic Pressure', 'Work Pressure', 
                               'Study Satisfaction', 'Job Satisfaction']:
                    input_data[feature] = float(value)  # Convert to float
                elif feature == 'CGPA':
                    input_data[feature] = float(value)  # Convert to float
                else:
                    input_data[feature] = value  # Keep as string for categorical
            
            # Create a DataFrame from the input data (one row)
            input_df = pd.DataFrame([input_data])
            
            # Apply same preprocessing as training data
            # Encode categorical variables using stored label encoders
            for col, le in self.ml_project.label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col])  # Transform using stored encoder
                    except ValueError:
                        # Handle case when value wasn't seen during training
                        input_df[col] = 0  # Default to 0 for unseen categories
            
            # Add engineered features (same as in training)
            input_df['Pressure_Score'] = input_df['Academic Pressure'] + input_df['Work Pressure']
            input_df['Satisfaction_Score'] = input_df['Study Satisfaction'] + input_df['Job Satisfaction']
            input_df['Stress_Index'] = (input_df['Academic Pressure'] + 
                                       input_df['Work Pressure'] + 
                                       input_df['Financial Stress']) / 3
            
            # Remove 'id' column if present
            if 'id' in input_df.columns:
                input_df = input_df.drop('id', axis=1)
            
            # Scale features using the same scaler used for training
            input_scaled = self.ml_project.scaler.transform(input_df)
            
            # Make prediction with the best model
            prediction = best_model.predict(input_scaled)[0]  # Get binary prediction (0 or 1)
            probability = best_model.predict_proba(input_scaled)[0]  # Get class probabilities
            
            # Display result with appropriate formatting based on prediction
            if prediction == 1:  # Depression predicted
                result_text = f"⚠️ Depression Risk: HIGH\n"
                result_text += f"Probability: {probability[1]:.1%}\n"  # Show probability of depression
                result_text += f"Model: {best_model_name}"
                self.prediction_result.set(result_text)
                # Note: You could add color here with Label configure if needed
            else:  # No depression predicted
                result_text = f"✓ Depression Risk: LOW\n"
                result_text += f"Probability of depression: {probability[1]:.1%}\n"
                result_text += f"Model: {best_model_name}"
                self.prediction_result.set(result_text)
            
            # Log prediction details to console for debugging
            print(f"\nPrediction made using {best_model_name}:")
            print(f"Input features: {input_data}")
            print(f"Prediction: {'Depression' if prediction == 1 else 'No Depression'}")
            print(f"Confidence: {max(probability):.1%}")
            
        except Exception as e:
            # Handle prediction errors
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging


# Standalone script for command-line usage
def main():
    """Main function for command-line execution - runs the ML pipeline without GUI"""
    print("Student Depression ML Classification Project")
    print("=" * 50)
    
    # Initialize project
    project = DepressionMLProject()
    
    # Load data
    filepath = 'student_depression_dataset.csv'  # Path to dataset file
    project.load_data(filepath)
    
    # Preprocess data
    X, y = project.preprocess_data()  # Preprocess raw data
    project.split_and_scale_data(X, y, test_size=0.2, use_smote=True)  # Split and scale with SMOTE
    
    # Train all models
    print("\nTraining models...")
    results = project.train_all_models(use_grid_search=False)  # Train without grid search for speed
    
    # Generate and print report
    print("\n" + project.generate_report())
    
    # Create and save visualizations
    fig = project.plot_results()
    plt.savefig('depression_ml_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'depression_ml_results.png'")
    
    # Save detailed results to text file
    with open('depression_ml_report.txt', 'w') as f:
        f.write(project.generate_report())
    print("Detailed report saved as 'depression_ml_report.txt'")
    
    # Export results to CSV for further analysis
    results_data = []
    for model_name, model_results in results.items():
        results_data.append({
            'Model': model_name,
            'Accuracy': model_results['accuracy'],
            'Precision': model_results['precision'],
            'Recall': model_results['recall'],
            'F1-Score': model_results['f1_score'],
            'ROC-AUC': model_results['roc_auc']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("Results exported to 'model_comparison_results.csv'")


# GUI launcher
def launch_gui():
    """Launch the GUI application"""
    root = tk.Tk()  # Create root window
    app = DepressionMLGUI(root)  # Initialize application with root window
    root.mainloop()  # Start the GUI event loop


if __name__ == "__main__":
    # Ask user for mode (GUI or CLI)
    mode = input("Enter mode (gui/cli): ").lower().strip()
    
    if mode == 'gui':
        launch_gui()  # Launch GUI mode
    else:
        main()  # Run command-line mode