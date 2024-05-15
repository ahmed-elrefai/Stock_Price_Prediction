from tkinter import messagebox
from customtkinter import *
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from tkinter import Menu, filedialog, ttk
import csv

class GUI:
    def __init__(self, master: CTk) -> None:
        self.master = master
        self.bgcolor = "#f0f0f0"  # Light gray background
        self.fgcolor = "#333333"  # Dark gray foreground
        self.accent_color = "#1e90ff"  # Dodger blue accent color
        self.master.geometry("1200x500")
        self.values = ["Decision Tree", "Random Forest", "Logistic Regression", "Support Vector Machine"]
        self.input_fields = []
        self.fit_button = None
        self.algo = ""
        self.model = None
        self.accuracy = None
        self.data = []
        self.prediction_data = pd.DataFrame(columns=["AMZN", "DPZ", "BTC", "NFLX", "month", "year", "Prediction"])

        self.init_widgets()

    def init_widgets(self):
        main_frame = CTkFrame(self.master, fg_color=self.bgcolor)
        main_frame.pack(fill="both", expand=True)

        menubar = Menu(self.master)
        self.master.configure(menu=menubar)

        file_menu = Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Save Model", command=self.save_model)

        graph_menu = Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Graphs", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self.load_model)


        control_frame = CTkFrame(main_frame, fg_color=self.accent_color)  # Use accent color for control frame
        control_frame.pack(fill="y", side=LEFT)

        self.algo_choice = CTkOptionMenu(control_frame, values=self.values, fg_color=self.fgcolor)  # Use dark gray foreground
        self.algo_choice.pack(side="top", padx=7, pady=15)
        self.accuracy_label = CTkLabel(control_frame,text="",text_color="black")
        self.accuracy_label.pack()

        input_frame = CTkFrame(control_frame, fg_color=self.fgcolor)  # Use dark gray foreground
        input_frame.pack(fill="both", side="bottom", expand=True)

        for field_name in ["AMZN", "DPZ", "BTC", "NFLX", "month", "year"]:
            label = CTkLabel(input_frame, text=field_name, fg_color=self.fgcolor)
            label.pack(side="top", padx=3, pady=3)
            entry = CTkEntry(input_frame, fg_color=self.accent_color)  # Use accent color for input fields
            entry.pack(side="top", padx=3, pady=3)
            self.input_fields.append(entry)

        X,Y = self.data_processing()
        self.fit_button = CTkButton(input_frame, text="Fit", command=lambda:self.fit_model(X,Y), fg_color=self.accent_color)  # Use accent color for button
        self.fit_button.pack(side="top", padx=7, pady=15)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, side="right", padx=10, pady=10)

        self.tree_frame = CTkFrame(notebook, fg_color=self.bgcolor)
        self.tree_frame.pack(fill="both", expand=True)
        notebook.add(self.tree_frame, text="Data")

        self.prediction_history_frame = CTkFrame(notebook, fg_color=self.bgcolor)
        self.prediction_history_frame.pack(fill="both", expand=True)
        notebook.add(self.prediction_history_frame, text="Prediction History")

        self.retrain_button = CTkButton(input_frame, text="Retrain", command=lambda:self.fit_model(X,Y), fg_color=self.accent_color)  # Use accent color for button
        self.load_csv()

        self.prediction_tree = ttk.Treeview(self.prediction_history_frame)

        columns = list(self.prediction_data.columns)
        self.prediction_tree["columns"] = tuple("#" + str(i) for i in range(1, len(columns) + 1))
        for i, column in enumerate(columns, start=1):
            self.prediction_tree.heading(f"#{i}", text=column)
            self.prediction_tree.column(f"#{i}", width=100)

        self.prediction_tree.pack(fill="both", expand=True)

    def load_csv(self, csv_path="stock.csv"):
        try:
            with open(csv_path, "r") as file:
                csv_reader = csv.reader(file)
                X, Y = self.data_processing()
                self.data = (X, Y)  
                self.display_csv(X, Y)
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find 'stock.csv' file.")

    def display_predictions(self):
        # Clear the existing Treeview
        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)

        # Configure columns including the prediction column
        columns = list(self.prediction_data.columns)
        self.prediction_tree["columns"] = tuple("#" + str(i) for i in range(1, len(columns) + 1))
        for i, column in enumerate(columns, start=1):
            self.prediction_tree.heading(f"#{i}", text=column)
            self.prediction_tree.column(f"#{i}", width=100)  # Adjust the width as needed

        # Insert data into the treeview
        for index, row in self.prediction_data.iterrows():
            # Convert numerical values to strings before insertion
            values = tuple(str(value) for value in row)
            self.prediction_tree.insert("", "end", values=values)

        self.prediction_tree.pack(fill="both", expand=True)

    def display_csv(self, X, Y):
        self.tree = ttk.Treeview(self.tree_frame)
        
        # Configure columns including Y column
        self.tree["columns"] = tuple("#" + str(i) for i in range(1, len(X.columns) + 2))  # Include an extra column for Y
        for i, column in enumerate(X.columns, start=1):
            self.tree.heading(f"#{i}", text=column)
            self.tree.column(f"#{i}", width=100)
        self.tree.heading(f"#{len(X.columns) + 1}", text="Y")
        self.tree.column(f"#{len(X.columns) + 1}", width=100)

        # Insert data into the treeview with Y column
        for index, (row, y_value) in enumerate(zip(X.values, Y), start=1):  # Starting from 1 for row numbers
            self.tree.insert("", "end", values=list(row) + [y_value])

        self.tree.pack(fill="both", expand=True)

    def load_model(self):
        model_path = filedialog.askopenfilename()
        self.model = joblib.load(model_path)
        if self.model != None:
            self.fit_button.configure(text="Predict", command=self.predict)

    def data_processing(self):
        data = pd.read_csv('stock.csv')

        data.drop(data.columns[data.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
        
        # Fill NaN values with column means
        data['AMZN'] = data['AMZN'].fillna(data['AMZN'].median())
        data['DPZ'] = data['DPZ'].fillna(data['DPZ'].median())
        data['BTC'] = data['BTC'].fillna(data['BTC'].median())
        data['NFLX'] = data['NFLX'].fillna(data['NFLX'].median())

        data['Date'] = pd.to_datetime(data['Date']).ffill()

        data['month'] = data['Date'].dt.month
        data['year'] = data['Date'].dt.year
        data.drop(['Date'], axis=1, inplace=True)

        # change date attribs to int64
        data['month'] = data['month'].astype(int)
        data['year'] = data['year'].astype(int)

        X = data.drop(["Price Movement "], axis=1)
        Y = data["Price Movement "]
        return X, Y


    def fit_model(self, X, Y):

        self.algo = self.algo_choice.get()

        if self.algo == "Decision Tree":
            self.model, self.accuracy = self.train_dt(X, Y)

        elif self.algo == "Random Forest":
            self.model, self.accuracy = self.train_rf(X, Y)
        elif self.algo == "Logistic Regression":
            self.model, self.accuracy = self.train_lr(X, Y)
        elif self.algo == "Support Vector Machine":
            self.model, self.accuracy = self.train_svm(X, Y)
        
        self.accuracy_label.configure(text="accuracy: {:.2f}%".format(self.accuracy))
        accuracy = "{} Accuracy is: {:.2f}%".format(self.algo, self.accuracy)
        messagebox.showinfo("Accuracy", accuracy)

        if self.model != None:
            self.fit_button.configure(text="Predict", command=self.predict)
            self.retrain_button.pack(side="top", padx=7, pady=15)

    def save_model(self):
        if self.model != None:
            joblib.dump(self.model, self.algo + ".joblib")
            messagebox.showinfo("success!", "model saved as: " + self.algo + ".joblib")
        else:
            messagebox.showerror("error", "there is no model to save, press the fit button to train one")

    def retrain_model(self):
        input_values = [float(entry.get()) for entry in self.input_fields]

        # Append the new input data to the existing prediction data
        new_input_data = pd.DataFrame([input_values + [""]], columns=["AMZN", "DPZ", "BTC", "NFLX", "Prediction"])
        self.prediction_data = pd.concat([self.prediction_data, new_input_data], ignore_index=True)

        # Perform any necessary preprocessing steps
        X_updated = self.prediction_data.drop(["Prediction"], axis=1)
        Y_updated = self.prediction_data["Prediction"]

        self.algo = self.algo_choice.get()
        # Retrain the model with the updated data
        if self.algo == "Decision Tree":
            self.model, self.accuracy = self.train_dt(X_updated, Y_updated)
        elif self.algo == "Random Forest":
            self.model, self.accuracy = self.train_rf(X_updated, Y_updated)
        elif self.algo == "Logistic Regression":
            self.model, self.accuracy = self.train_lr(X_updated, Y_updated)
        elif self.algo == "Support Vector Machine":
            self.model, self.accuracy = self.train_svm(X_updated, Y_updated)

        # Display the accuracy after retraining
        accuracy = "{} Accuracy is: {:.2f}%".format(self.algo, accuracy)
        messagebox.showinfo("Accuracy", accuracy)

        if self.model is not None:
            self.fit_button.configure(text="Predict", command=self.predict)


    def predict(self):
        input_values = [float(entry.get()) for entry in self.input_fields]
        
        feature_names = ["AMZN", "DPZ", "BTC", "NFLX","month","year"]

        prediction = self.model.predict([input_values])
        print("Prediction", f"Predicted Outcome: {prediction}")

        # Store input values and prediction in the prediction dataframe
        input_data = pd.DataFrame([input_values + [prediction[0]]], columns=feature_names + ["Prediction"])

        # Exclude empty or all-NA columns before concatenation
        input_data = input_data.dropna(axis=1, how='all')
        self.prediction_data = pd.concat([self.prediction_data, input_data], ignore_index=True)
        self.display_predictions()

    def train_dt(self, X, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)

        decision_tree_model = DecisionTreeClassifier(min_samples_leaf=3)
        cv_scores_dt = cross_val_score(decision_tree_model, X, Y, cv=5)

        decision_tree_model.fit(x_train, y_train)
        predictions_dt_train = decision_tree_model.predict(x_train)
        accuracy_train_dt = accuracy_score(y_train, predictions_dt_train)

        print("Cross-validation scores (Decision Tree):", cv_scores_dt)
        print("Average Cross-validation score (Decision Tree): {:.2f}%".format(cv_scores_dt.mean() * 100))
        print("\nDecision Tree model Training Accuracy: {:.2f}%".format(accuracy_train_dt * 100))
        predictions_dt_test = decision_tree_model.predict(x_test)
        accuracy_test_dt = accuracy_score(y_test, predictions_dt_test)
        print("Decision Tree model Testing Accuracy: {:.2f}%".format(accuracy_test_dt * 100))
        return (decision_tree_model ,accuracy_test_dt * 100)

    def train_svm(self, X, Y):

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)  

        # Model Accuracy (SVM)
        svm_model = SVC(kernel='linear',C=20, random_state=0)
        svm_model.fit(x_train_scaled, y_train)
        cv_scores_svm = cross_val_score(svm_model, X, Y, cv=5)

        predictions_svm_train = svm_model.predict(x_train_scaled)
        accuracy_train_svm = accuracy_score(y_train, predictions_svm_train)
        predictions_svm_test = svm_model.predict(x_test_scaled)
        accuracy_test_svm = accuracy_score(y_test, predictions_svm_test)

        print("\nCross-validation scores (SVM):", cv_scores_svm)
        print("Average Cross-validation score (SVM): {:.2f}%".format(cv_scores_svm.mean() * 100))
        print("\nSVM model Training Accuracy: {:.2f}%".format(accuracy_train_svm * 100))
        print("SVM model Testing Accuracy: {:.2f}%".format(accuracy_test_svm * 100))
        return (svm_model, accuracy_test_svm * 100)
    
    def train_lr(self, X, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)

        log_model = LogisticRegression().fit(x_train_scaled, y_train)
        prediction_log = log_model.predict(x_train_scaled)
        accuracy_train_log = accuracy_score(y_train, prediction_log)
        print("\nLogistic Regression model Training Accuracy: {:.2f}%".format(accuracy_train_log * 100))

        # Evaluate Logistic Regression model
        predictions_log_test = log_model.predict(x_test_scaled)
        accuracy_test_log = accuracy_score(y_test, predictions_log_test)
        print("Logistic Regression model Test Accuracy: {:.2f}%".format(accuracy_test_log * 100))
        return (log_model, accuracy_test_log * 100)

    def train_rf(self, X,Y):

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
        random_forest_model = RandomForestClassifier(n_estimators=10,random_state=0,ccp_alpha=0.2)  # Reduce number of estimators and depth

        cv_scores_rf = cross_val_score(random_forest_model, X, Y, cv=5)
        random_forest_model.fit(x_train, y_train)


        # Model Prediction & Accuracy (Random Forest)
        predictions_rf_train = random_forest_model.predict(x_train)
        accuracy_train_rf = accuracy_score(y_train, predictions_rf_train)
        predictions_rf_test = random_forest_model.predict(x_test)
        accuracy_test_rf = accuracy_score(y_test, predictions_rf_test)


        print("\nCross-validation scores (Random Forest):", cv_scores_rf)
        print("Average Cross-validation score (Random Forest): {:.2f}%".format(cv_scores_rf.mean() * 100))
        print("\nRandom Forest model Training Accuracy: {:.2f}%".format(accuracy_train_rf * 100))
        predictions_rf_test = random_forest_model.predict(x_test)
        accuracy_test_rf = accuracy_score(y_test, predictions_rf_test)
        print("Random Forest model Testing Accuracy: {:.2f}%".format(accuracy_test_rf * 100))
        return (random_forest_model, accuracy_test_rf * 100)

if __name__ == "__main__":
    root = CTk()
    app = GUI(root)
    root.mainloop()
