import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning App")

        self.train_file = ""
        self.test_file = ""

        self.create_widgets()

    def create_widgets(self):
        # Train file selection
        self.train_button = tk.Button(self.root, text="Select Train Data", command=self.load_train_data)
        self.train_button.pack(pady=10)

        # Test file selection
        self.test_button = tk.Button(self.root, text="Select Test Data", command=self.load_test_data)
        self.test_button.pack(pady=10)

        # Run analysis button
        self.run_button = tk.Button(self.root, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(pady=10)

        # Prediction output
        self.output_label = tk.Label(self.root, text="Prediction Results will appear here.")
        self.output_label.pack(pady=10)

    def load_train_data(self):
        self.train_file = filedialog.askopenfilename(title="Select Train CSV File", filetypes=[("CSV Files", "*.csv")])
        if self.train_file:
            messagebox.showinfo("File Selected", f"Train Data loaded from {self.train_file}")

    def load_test_data(self):
        self.test_file = filedialog.askopenfilename(title="Select Test CSV File", filetypes=[("CSV Files", "*.csv")])
        if self.test_file:
            messagebox.showinfo("File Selected", f"Test Data loaded from {self.test_file}")

    def run_analysis(self):
        if not self.train_file or not self.test_file:
            messagebox.showerror("Error", "Please load both train and test files.")
            return

        # Load the data
        train = pd.read_csv(self.train_file)
        test = pd.read_csv(self.test_file)

        # Data Preprocessing
        self.preprocess_data(train, test)

        # Train the model
        self.train_model(train)

    def preprocess_data(self, train, test):
        # Handle missing values and preprocess data here
        # Drop unwanted columns
        train.drop(['num_outbound_cmds'], axis=1, inplace=True)
        test.drop(['num_outbound_cmds'], axis=1, inplace=True)

        # Standardize numeric columns
        scaler = StandardScaler()
        cols = train.select_dtypes(include=['float64','int64']).columns
        sc_train = scaler.fit_transform(train[cols])
        sc_test = scaler.transform(test[cols])

        # Encode categorical columns
        encoder = LabelEncoder()
        cattrain = train.select_dtypes(include=['object']).copy()
        cattest = test.select_dtypes(include=['object']).copy()
        traincat = cattrain.apply(encoder.fit_transform)
        testcat = cattest.apply(encoder.fit_transform)

        # Merge processed data
        self.train_x = pd.concat([pd.DataFrame(sc_train, columns=cols), traincat.drop(['class'], axis=1)], axis=1)
        self.train_y = train['class']
        self.test_df = pd.concat([pd.DataFrame(sc_test, columns=cols), testcat], axis=1)

    def train_model(self, train):
        # Example Model: KNN Classifier
        X_train, X_test, Y_train, Y_test = train_test_split(self.train_x, self.train_y, train_size=0.60, random_state=2)
        
        model = KNeighborsClassifier(n_jobs=-1)
        model.fit(X_train, Y_train)

        # Evaluate the model
        accuracy = metrics.accuracy_score(Y_test, model.predict(X_test))
        classification_report = metrics.classification_report(Y_test, model.predict(X_test))

        # Display results
        self.output_label.config(text=f"Model Accuracy: {accuracy:.4f}\n{classification_report}")

        # Predict on the test set and display prediction
        prediction = model.predict(self.test_df)
        test['prediction'] = prediction
        test.to_csv("predictions.csv", index=False)

        messagebox.showinfo("Analysis Complete", "Model trained and predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
