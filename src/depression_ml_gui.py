import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# For GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DepressionMLProject:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load and perform initial data exploration"""
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print("\nDataset Info:")
        print(self.df.info())
        print("\nTarget Distribution:")
        print(self.df['Depression'].value_counts())
        return self.df
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        df_processed = self.df.copy()
        
        # Handle missing values
        numeric_columns = ['Age', 'CGPA', 'Academic Pressure', 'Work Pressure', 
                          'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours']
        
        # Impute numeric columns
        for col in numeric_columns:
            if col in df_processed.columns:
                imputer = SimpleImputer(strategy='median')
                df_processed[col] = imputer.fit_transform(df_processed[[col]])
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'City', 'Profession', 'Sleep Duration', 
                              'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                              'Financial Stress', 'Family History of Mental Illness']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Feature engineering
        # Create interaction features
        df_processed['Pressure_Score'] = df_processed['Academic Pressure'] + df_processed['Work Pressure']
        df_processed['Satisfaction_Score'] = df_processed['Study Satisfaction'] + df_processed['Job Satisfaction']
        df_processed['Stress_Index'] = (df_processed['Academic Pressure'] + 
                                       df_processed['Work Pressure'] + 
                                       df_processed['Financial Stress']) / 3
        
        # Prepare features and target
        X = df_processed.drop(['Depression', 'id'], axis=1)
        y = df_processed['Depression']
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, use_smote=True):
        """Split data and apply scaling"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply SMOTE if requested
        if use_smote and y_train.value_counts().min() / y_train.value_counts().max() < 0.8:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("SMOTE applied to balance classes")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize all ML models with optimal parameters"""
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
    
    def train_single_model(self, model_name, model, use_grid_search=False):
        """Train a single model with optional grid search"""
        print(f"\nTraining {model_name}...")
        
        if use_grid_search and model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': best_model
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def train_all_models(self, use_grid_search=False):
        """Train all models and store results"""
        self.initialize_models()
        
        for model_name, model in self.models.items():
            self.results[model_name] = self.train_single_model(model_name, model, use_grid_search)
        
        return self.results
    
    def plot_results(self):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison Bar Chart
        ax1 = axes[0, 0]
        metrics_df = pd.DataFrame({
            model: [results['accuracy'], results['precision'], results['recall'], 
                   results['f1_score'], results['roc_auc']]
            for model, results in self.results.items()
        })
        metrics_df.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 1)
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            ax2.plot(fpr, tpr, label=f"{model_name} (AUC={results['roc_auc']:.3f})")
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves Comparison')
        ax2.legend()
        
        # 3. Best Model Confusion Matrix
        ax3 = axes[1, 0]
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model_name}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Feature Importance (if Random Forest is available)
        ax4 = axes[1, 1]
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            ax4.bar(range(10), importances[indices])
            ax4.set_xticks(range(10))
            ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45)
            ax4.set_title('Top 10 Feature Importances (Random Forest)')
            ax4.set_ylabel('Importance')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = []
        report.append("=" * 80)
        report.append("STUDENT DEPRESSION PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("\n1. DATASET OVERVIEW")
        report.append(f"   - Total samples: {len(self.df)}")
        report.append(f"   - Features: {self.X_train.shape[1]}")
        report.append(f"   - Training samples: {len(self.X_train)}")
        report.append(f"   - Testing samples: {len(self.X_test)}")
        report.append(f"   - Class distribution in test set:")
        report.append(f"     No Depression: {(self.y_test == 0).sum()}")
        report.append(f"     Depression: {(self.y_test == 1).sum()}")
        
        report.append("\n2. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        report.append("-" * 80)
        
        for model_name, results in self.results.items():
            report.append(f"{model_name:<20} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
                         f"{results['recall']:<10.3f} {results['f1_score']:<10.3f} {results['roc_auc']:<10.3f}")
        
        # Best model
        best_model = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        report.append("\n3. BEST PERFORMING MODEL")
        report.append(f"   {best_model} with ROC-AUC score of {self.results[best_model]['roc_auc']:.3f}")
        
        report.append("\n4. DETAILED CLASSIFICATION REPORT - " + best_model)
        report.append(classification_report(self.y_test, self.results[best_model]['y_pred'],
                                          target_names=['No Depression', 'Depression']))
        
        return '\n'.join(report)


class DepressionMLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Depression ML Classifier")
        self.root.geometry("1200x800")
        
        self.ml_project = DepressionMLProject()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Create notebook for tabs
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
        """Setup data loading tab"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.data_tab, text="Load Dataset", padding=10)
        file_frame.pack(fill='x', padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, text="Dataset Path:").grid(row=0, column=0, sticky='w')
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        ttk.Button(file_frame, text="Load Data", command=self.load_data).grid(row=0, column=3, padx=5)
        
        # Data info frame
        info_frame = ttk.LabelFrame(self.data_tab, text="Dataset Information", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.data_info_text = tk.Text(info_frame, height=20, width=80)
        self.data_info_text.pack(fill='both', expand=True)
        
        # Preprocessing options
        prep_frame = ttk.LabelFrame(self.data_tab, text="Preprocessing Options", padding=10)
        prep_frame.pack(fill='x', padx=10, pady=10)
        
        self.use_smote_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep_frame, text="Use SMOTE for class balancing", 
                       variable=self.use_smote_var).pack(anchor='w')
        
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Label(prep_frame, text="Test Size:").pack(side='left', padx=5)
        ttk.Scale(prep_frame, from_=0.1, to=0.4, variable=self.test_size_var, 
                 orient='horizontal', length=200).pack(side='left', padx=5)
        ttk.Label(prep_frame, textvariable=self.test_size_var).pack(side='left')
        
        ttk.Button(prep_frame, text="Preprocess Data", 
                  command=self.preprocess_data).pack(side='right', padx=5)
        
    def setup_training_tab(self):
        """Setup model training tab"""
        # Model selection frame
        model_frame = ttk.LabelFrame(self.training_tab, text="Model Selection", padding=10)
        model_frame.pack(fill='x', padx=10, pady=10)
        
        self.selected_models = {}
        models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network']
        
        for i, model in enumerate(models):
            var = tk.BooleanVar(value=True)
            self.selected_models[model] = var
            ttk.Checkbutton(model_frame, text=model, variable=var).grid(row=i//3, column=i%3, sticky='w', padx=10, pady=5)
        
        # Training options
        options_frame = ttk.LabelFrame(self.training_tab, text="Training Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=10)
        
        self.use_grid_search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use Grid Search for hyperparameter tuning", 
                       variable=self.use_grid_search_var).pack(anchor='w')
        
        ttk.Button(options_frame, text="Start Training", 
                  command=self.start_training).pack(pady=10)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.progress_text = tk.Text(progress_frame, height=15, width=80)
        self.progress_text.pack(fill='both', expand=True)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
    def setup_results_tab(self):
        """Setup results visualization tab"""
        # Results summary frame
        summary_frame = ttk.LabelFrame(self.results_tab, text="Results Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=10)
        
        self.results_text = tk.Text(summary_frame, height=10, width=80)
        self.results_text.pack(fill='x')
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self.results_tab, text="Visualizations", padding=10)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Buttons for different visualizations
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="Show All Plots", 
                  command=self.show_plots).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        
        # Canvas for plots
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill='both', expand=True)
        
    def setup_prediction_tab(self):
        """Setup prediction tab for new data"""
        instruction_label = ttk.Label(self.prediction_tab, 
                                    text="Enter values for all features to predict depression likelihood:",
                                    font=('Arial', 12, 'bold'))
        instruction_label.pack(pady=10)
        
        # Create scrollable frame for input fields
        canvas = tk.Canvas(self.prediction_tab)
        scrollbar = ttk.Scrollbar(self.prediction_tab, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)  # Store reference
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input fields with appropriate widgets
        self.prediction_inputs = {}
        self.prediction_widgets = {}  # Store widget references
        
        # Define feature types and options
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
        
        # Create appropriate widgets for each feature
        row = 0
        for feature, config in feature_configs.items():
            # Label
            label = ttk.Label(self.scrollable_frame, text=f"{feature}:")
            label.grid(row=row, column=0, sticky='w', padx=10, pady=5)
            
            if config['type'] == 'dropdown':
                var = tk.StringVar()
                if config['options']:
                    var.set(config['options'][0])
                widget = ttk.Combobox(self.scrollable_frame, textvariable=var, 
                                    values=config['options'], state='readonly', width=27)
                widget.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                self.prediction_inputs[feature] = var
                self.prediction_widgets[feature] = widget  # Store widget reference
                
            elif config['type'] == 'spinbox':
                if feature == 'CGPA':
                    var = tk.DoubleVar(value=config.get('default', 0))
                    widget = ttk.Spinbox(self.scrollable_frame, from_=config['from'], to=config['to'],
                                       increment=config.get('increment', 1), textvariable=var,
                                       width=28, format="%.1f")
                else:
                    var = tk.IntVar(value=config.get('default', 0))
                    widget = ttk.Spinbox(self.scrollable_frame, from_=config['from'], to=config['to'],
                                       textvariable=var, width=28)
                widget.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                self.prediction_inputs[feature] = var
                
            elif config['type'] == 'scale':
                var = tk.IntVar(value=config.get('default', 3))
                frame = ttk.Frame(self.scrollable_frame)
                frame.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                
                scale = ttk.Scale(frame, from_=config['from'], to=config['to'],
                                variable=var, orient='horizontal', length=200)
                scale.pack(side='left')
                
                value_label = ttk.Label(frame, textvariable=var, width=3)
                value_label.pack(side='left', padx=5)
                
                self.prediction_inputs[feature] = var
                
            elif config['type'] == 'radio':
                var = tk.StringVar(value=config['options'][1])  # Default to 'No'
                frame = ttk.Frame(self.scrollable_frame)
                frame.grid(row=row, column=1, padx=10, pady=5, sticky='w')
                
                for option in config['options']:
                    rb = ttk.Radiobutton(frame, text=option, variable=var, value=option)
                    rb.pack(side='left', padx=10)
                
                self.prediction_inputs[feature] = var
            
            row += 1
        
        # Add some helpful text
        help_frame = ttk.LabelFrame(self.scrollable_frame, text="Feature Guidelines", padding=5)
        help_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        help_text = """• Academic/Work Pressure: 1 (Low) to 5 (High)
• Study/Job Satisfaction: 1 (Very Unsatisfied) to 5 (Very Satisfied)
• CGPA: 0.0 to 10.0
• Work/Study Hours: Daily hours spent working/studying"""
        
        ttk.Label(help_frame, text=help_text, justify='left').pack(anchor='w')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Prediction button and result
        pred_frame = ttk.Frame(self.prediction_tab)
        pred_frame.pack(fill='x', padx=10, pady=10)
        
        # Model info frame
        model_info_frame = ttk.LabelFrame(pred_frame, text="Model Information", padding=10)
        model_info_frame.pack(fill='x', padx=20, pady=5)
        
        self.model_info_var = tk.StringVar(value="No models trained yet")
        model_info_label = ttk.Label(model_info_frame, textvariable=self.model_info_var,
                                   font=('Arial', 11))
        model_info_label.pack()
        
        # Update model info when models are trained
        self.update_model_info()
        
        button_frame = ttk.Frame(pred_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Predict Depression Risk", 
                  command=self.make_prediction, style='Accent.TButton').pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Reset Form", 
                  command=self.reset_prediction_form).pack(side='left', padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(pred_frame, text="Prediction Result", padding=10)
        result_frame.pack(fill='x', padx=20, pady=10)
        
        self.prediction_result = tk.StringVar()
        result_label = ttk.Label(result_frame, textvariable=self.prediction_result, 
                               font=('Arial', 14, 'bold'))
        result_label.pack(pady=10)
        
    def update_model_info(self):
        """Update the model info display in prediction tab"""
        if hasattr(self, 'model_info_var') and self.ml_project.results:
            best_model_name = max(self.ml_project.results, 
                                key=lambda x: self.ml_project.results[x]['roc_auc'])
            best_score = self.ml_project.results[best_model_name]['roc_auc']
            info_text = f"Using: {best_model_name} (ROC-AUC: {best_score:.3f})"
            self.model_info_var.set(info_text)
    
    def reset_prediction_form(self):
        """Reset all prediction form fields to defaults"""
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
        
        for feature, var in self.prediction_inputs.items():
            if feature in defaults:
                var.set(defaults[feature])
            elif hasattr(var, 'get') and isinstance(var.get(), str):
                # For dropdown boxes, set to first option if available
                if hasattr(var, '_values') and var._values:
                    var.set(var._values[0])
        
        self.prediction_result.set("")
        
    def browse_file(self):
        """Browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def load_data(self):
        """Load dataset and display info"""
        filepath = self.file_path_var.get()
        if not filepath:
            messagebox.showerror("Error", "Please select a dataset file")
            return
        
        try:
            df = self.ml_project.load_data(filepath)
            
            # Display data info
            info_text = f"Dataset loaded successfully!\n\n"
            info_text += f"Shape: {df.shape}\n\n"
            info_text += f"Columns:\n{', '.join(df.columns)}\n\n"
            info_text += f"Data types:\n{df.dtypes}\n\n"
            info_text += f"Missing values:\n{df.isnull().sum()}\n\n"
            info_text += f"Target distribution:\n{df['Depression'].value_counts()}"
            
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
        
        # Update City dropdown
        if 'City' in df.columns and 'City' in self.prediction_widgets:
            cities = sorted(df['City'].dropna().unique().tolist())
            self.prediction_widgets['City']['values'] = cities
            if cities:
                self.prediction_inputs['City'].set(cities[0])
        
        # Update Profession dropdown
        if 'Profession' in df.columns and 'Profession' in self.prediction_widgets:
            professions = sorted(df['Profession'].dropna().unique().tolist())
            self.prediction_widgets['Profession']['values'] = professions
            if professions:
                self.prediction_inputs['Profession'].set(professions[0])
        
        # Update Degree dropdown
        if 'Degree' in df.columns and 'Degree' in self.prediction_widgets:
            degrees = sorted(df['Degree'].dropna().unique().tolist())
            self.prediction_widgets['Degree']['values'] = degrees
            if degrees:
                self.prediction_inputs['Degree'].set(degrees[0])
        
        print(f"Updated dropdowns - Cities: {len(cities) if 'cities' in locals() else 0}, "
              f"Professions: {len(professions) if 'professions' in locals() else 0}, "
              f"Degrees: {len(degrees) if 'degrees' in locals() else 0}")
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        if self.ml_project.df is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
        
        try:
            X, y = self.ml_project.preprocess_data()
            test_size = self.test_size_var.get()
            use_smote = self.use_smote_var.get()
            
            self.ml_project.split_and_scale_data(X, y, test_size=test_size, use_smote=use_smote)
            
            messagebox.showinfo("Success", "Data preprocessing completed!")
            
            # Update info
            info_text = self.data_info_text.get(1.0, tk.END)
            info_text += f"\n\nPreprocessing completed!"
            info_text += f"\nTraining samples: {len(self.ml_project.X_train)}"
            info_text += f"\nTesting samples: {len(self.ml_project.X_test)}"
            info_text += f"\nFeatures after preprocessing: {self.ml_project.X_train.shape[1]}"
            
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess data: {str(e)}")
    
    def start_training(self):
        """Start model training in a separate thread"""
        if self.ml_project.X_train is None:
            messagebox.showerror("Error", "Please preprocess the data first")
            return
        
        # Clear previous progress
        self.progress_text.delete(1.0, tk.END)
        self.progress_bar.start()
        
        # Run training in separate thread
        thread = threading.Thread(target=self.train_models)
        thread.start()
    
    def train_models(self):
        """Train selected models"""
        try:
            # Filter selected models
            selected = [name for name, var in self.selected_models.items() if var.get()]
            
            if not selected:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please select at least one model"))
                return
            
            # Initialize only selected models
            self.ml_project.models = {}
            all_models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
            
            for model_name in selected:
                self.ml_project.models[model_name] = all_models[model_name]
            
            # Train models
            use_grid = self.use_grid_search_var.get()
            
            for model_name, model in self.ml_project.models.items():
                self.root.after(0, lambda mn=model_name: self.progress_text.insert(tk.END, f"\nTraining {mn}...\n"))
                
                results = self.ml_project.train_single_model(model_name, model, use_grid)
                self.ml_project.results[model_name] = results
                
                # Update progress
                progress_msg = f"Completed {model_name}:\n"
                progress_msg += f"  Accuracy: {results['accuracy']:.3f}\n"
                progress_msg += f"  ROC-AUC: {results['roc_auc']:.3f}\n"
                
                self.root.after(0, lambda msg=progress_msg: self.progress_text.insert(tk.END, msg))
            
            self.root.after(0, self.training_completed)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
        finally:
            self.root.after(0, self.progress_bar.stop)
    
    def training_completed(self):
        """Called when training is completed"""
        self.progress_bar.stop()
        messagebox.showinfo("Success", "Model training completed!")
        
        # Display results summary
        report = self.ml_project.generate_report()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, report)
        
        # Update model info in prediction tab
        self.update_model_info()
    
    def show_plots(self):
        """Display all visualization plots"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Clear previous plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Create plots
            fig = self.ml_project.plot_results()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plots: {str(e)}")
    
    def generate_report(self):
        """Generate and save detailed report"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            report = self.ml_project.generate_report()
            
            # Ask user where to save
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Report saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def export_results(self):
        """Export results to CSV"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Create results dataframe
            results_data = []
            for model_name, results in self.ml_project.results.items():
                results_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'ROC-AUC': results['roc_auc'],
                    'CV Mean': results.get('cv_mean', 'N/A'),
                    'CV Std': results.get('cv_std', 'N/A')
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Ask user where to save
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                results_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def make_prediction(self):
        """Make prediction using the best model"""
        if not self.ml_project.results:
            messagebox.showerror("Error", "Please train models first")
            return
        
        try:
            # Get best model
            best_model_name = max(self.ml_project.results, 
                                key=lambda x: self.ml_project.results[x]['roc_auc'])
            best_model = self.ml_project.results[best_model_name]['model']
            
            # Prepare input data
            input_data = {}
            for feature, var in self.prediction_inputs.items():
                value = var.get()
                
                # Convert to appropriate type based on feature
                if feature in ['Age', 'Work/Study Hours']:
                    input_data[feature] = float(value)
                elif feature in ['Academic Pressure', 'Work Pressure', 
                               'Study Satisfaction', 'Job Satisfaction']:
                    input_data[feature] = float(value)
                elif feature == 'CGPA':
                    input_data[feature] = float(value)
                else:
                    input_data[feature] = value
            
            # Create dataframe and preprocess
            input_df = pd.DataFrame([input_data])
            
            # Apply same preprocessing as training data
            for col, le in self.ml_project.label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col])
                    except ValueError:
                        # Handle unseen categories
                        input_df[col] = 0
            
            # Add engineered features
            input_df['Pressure_Score'] = input_df['Academic Pressure'] + input_df['Work Pressure']
            input_df['Satisfaction_Score'] = input_df['Study Satisfaction'] + input_df['Job Satisfaction']
            input_df['Stress_Index'] = (input_df['Academic Pressure'] + 
                                       input_df['Work Pressure'] + 
                                       input_df['Financial Stress']) / 3
            
            # Ensure all columns match training data
            # Remove 'id' column if present
            if 'id' in input_df.columns:
                input_df = input_df.drop('id', axis=1)
            
            # Scale features
            input_scaled = self.ml_project.scaler.transform(input_df)
            
            # Make prediction
            prediction = best_model.predict(input_scaled)[0]
            probability = best_model.predict_proba(input_scaled)[0]
            
            # Display result with color coding
            if prediction == 1:
                result_text = f"⚠️ Depression Risk: HIGH\n"
                result_text += f"Probability: {probability[1]:.1%}\n"
                result_text += f"Model: {best_model_name}"
                self.prediction_result.set(result_text)
                # You could add color here if using a Label widget with configure
            else:
                result_text = f"✓ Depression Risk: LOW\n"
                result_text += f"Probability of depression: {probability[1]:.1%}\n"
                result_text += f"Model: {best_model_name}"
                self.prediction_result.set(result_text)
            
            # Log prediction details
            print(f"\nPrediction made using {best_model_name}:")
            print(f"Input features: {input_data}")
            print(f"Prediction: {'Depression' if prediction == 1 else 'No Depression'}")
            print(f"Confidence: {max(probability):.1%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()


# Standalone script for command-line usage
def main():
    """Main function for command-line execution"""
    print("Student Depression ML Classification Project")
    print("=" * 50)
    
    # Initialize project
    project = DepressionMLProject()
    
    # Load data
    filepath = 'student_depression_dataset.csv'  # Update this path
    project.load_data(filepath)
    
    # Preprocess data
    X, y = project.preprocess_data()
    project.split_and_scale_data(X, y, test_size=0.2, use_smote=True)
    
    # Train all models
    print("\nTraining models...")
    results = project.train_all_models(use_grid_search=False)
    
    # Generate report
    print("\n" + project.generate_report())
    
    # Create visualizations
    fig = project.plot_results()
    plt.savefig('depression_ml_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'depression_ml_results.png'")
    
    # Save detailed results
    with open('depression_ml_report.txt', 'w') as f:
        f.write(project.generate_report())
    print("Detailed report saved as 'depression_ml_report.txt'")
    
    # Export results to CSV
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
    root = tk.Tk()
    app = DepressionMLGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # Ask user for mode
    mode = input("Enter mode (gui/cli): ").lower().strip()
    
    if mode == 'gui':
        launch_gui()
    else:
        main()