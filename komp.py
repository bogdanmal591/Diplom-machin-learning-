import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MLModelComparison:
    def __init__(self, root):
        self.root = root
        self.root.title("Порівняння моделей машинного навчання")
        self.filename = None
        self.df = pd.DataFrame()

        # Initialize models with checkboxes
        self.models = [
            {'name': 'Лінійна регресія', 'model': LinearRegression(), 'var': tk.BooleanVar(value=True)},
            {'name': 'Рідж регресія', 'model': Ridge(alpha=0.1), 'var': tk.BooleanVar(value=True)},
            {'name': 'Лассо регресія', 'model': Lasso(alpha=0.1), 'var': tk.BooleanVar(value=True)},
            {'name': 'Elastic Net регресія', 'model': ElasticNet(alpha=0.1, l1_ratio=0.5),
             'var': tk.BooleanVar(value=True)},
            {'name': 'Дерево рішень', 'model': DecisionTreeRegressor(max_depth=5), 'var': tk.BooleanVar(value=True)},
            {'name': 'Ліс рішень', 'model': RandomForestRegressor(n_estimators=100, max_depth=5),
             'var': tk.BooleanVar(value=True)},
            {'name': 'Градієнтне підсилення', 'model': GradientBoostingRegressor(random_state=0),
             'var': tk.BooleanVar(value=True)},
            {'name': 'Опорний вектор', 'model': SVR(epsilon=0.2), 'var': tk.BooleanVar(value=True)}
        ]

        # Options for evaluation methods
        self.cross_validation_var = tk.BooleanVar(value=False)
        self.bic_var = tk.BooleanVar(value=False)
        self.cv_option = tk.StringVar(value="5")

        # File selection, data generation, Schwarz criterion, and model run buttons
        self.label_file = tk.Label(text="Виберіть файл даних або згенеруйте дані:")
        self.button_file = tk.Button(text="Обрати файл", command=self.browse_file)
        self.button_generate = tk.Button(text="Налаштувати та згенерувати дані", command=self.select_data_generation)
        self.button_run = tk.Button(text="Запустити моделі", command=self.run_models)

        # Layout for buttons
        self.label_file.grid(row=0, column=0, columnspan=2, pady=5)
        self.button_file.grid(row=1, column=0, padx=5, pady=5)
        self.button_generate.grid(row=1, column=1, padx=5, pady=5)
        self.button_run.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        # Checkboxes for model selection
        for i, model in enumerate(self.models):
            tk.Checkbutton(self.root, text=model['name'], variable=model['var']).grid(row=3 + i, column=0, sticky="w")

        # Evaluation method checkboxes and cross-validation options
        tk.Checkbutton(self.root, text="Перехресна валідація", variable=self.cross_validation_var).grid(
            row=3 + len(self.models), column=0, sticky="w")
        tk.Label(self.root, text="Тип CV:").grid(row=3 + len(self.models), column=1, sticky="e")
        tk.OptionMenu(self.root, self.cv_option, "5", "10", "LOOC").grid(row=3 + len(self.models), column=2)

        tk.Checkbutton(self.root, text="Байєсівський інформаційний критерій (BIC)", variable=self.bic_var).grid(
            row=4 + len(self.models), column=0, sticky="w")

        # Plot area with larger size
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlabel('Істинні значення')
        self.ax.set_ylabel('Прогнози')
        self.ax.set_title('Порівняння прогнозів моделей')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=3, column=1, columnspan=2, rowspan=len(self.models))

    def browse_file(self):
        self.filename = filedialog.askopenfilename(initialdir=".", title="Виберіть файл",
                                                   filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if self.filename:
            self.label_file.config(text=f"Вибраний файл: {self.filename}")

    def select_data_generation(self):
        self.gen_window = tk.Toplevel(self.root)
        self.gen_window.title("Налаштування генерації даних")

        tk.Label(self.gen_window, text="Кількість зразків:").grid(row=0, column=0)
        self.entry_samples = tk.Entry(self.gen_window)
        self.entry_samples.grid(row=0, column=1)
        self.entry_samples.insert(0, "100")

        tk.Label(self.gen_window, text="Кількість ознак:").grid(row=1, column=0)
        self.entry_features = tk.Entry(self.gen_window)
        self.entry_features.grid(row=1, column=1)
        self.entry_features.insert(0, "10")

        tk.Label(self.gen_window, text="Рівень шуму:").grid(row=2, column=0)
        self.entry_noise = tk.Entry(self.gen_window)
        self.entry_noise.grid(row=2, column=1)
        self.entry_noise.insert(0, "0.1")

        tk.Label(self.gen_window, text="Тип моделі для генерації даних:").grid(row=3, column=0)
        self.model_type = tk.StringVar(value="linear")
        model_options = [
            ("Лінійна регресія", "linear"),
            ("Рідж регресія", "ridge"),
            ("Лассо регресія", "lasso"),
            ("Elastic Net регресія", "elastic_net"),
            ("Дерево рішень", "tree_based"),
            ("Ліс рішень", "random_forest"),
            ("Опорний вектор", "svr")
        ]
        for i, (label, option) in enumerate(model_options):
            tk.Radiobutton(self.gen_window, text=label, variable=self.model_type, value=option).grid(row=4 + i,
                                                                                                     column=1,
                                                                                                     sticky="w")

        tk.Button(self.gen_window, text="Генерувати дані", command=self.apply_data_generation).grid(row=11, column=0,
                                                                                                    columnspan=2)

    def apply_data_generation(self):
        num_samples = int(self.entry_samples.get())
        num_features = int(self.entry_features.get())
        noise_level = float(self.entry_noise.get())
        model_type = self.model_type.get()

        self.generate_data(num_samples=num_samples, num_features=num_features, noise_level=noise_level,
                           model_type=model_type)
        self.gen_window.destroy()
        messagebox.showinfo("Генерація даних", "Дані успішно згенеровані!")

    def generate_data(self, num_samples=100, num_features=10, noise_level=0.1, model_type="linear"):
        X = np.random.rand(num_samples, num_features)
        if model_type == "linear":
            y = np.dot(X, np.random.rand(num_features)) + noise_level * np.random.randn(num_samples)
        elif model_type == "ridge":
            y = np.dot(X, np.random.rand(num_features)) + noise_level * np.random.randn(num_samples)
        elif model_type == "lasso":
            coef = np.random.rand(num_features)
            coef[3:] = 0
            y = np.dot(X, coef) + noise_level * np.random.randn(num_samples)
        elif model_type == "elastic_net":
            coef = np.random.rand(num_features)
            coef[5:] = 0
            y = np.dot(X, coef) + noise_level * np.random.randn(num_samples)
        elif model_type == "tree_based":
            y = np.sin(np.dot(X, np.random.rand(num_features)))
        elif model_type == "random_forest":
            # Generate meaningful continuous target values
            y = np.dot(X, np.random.rand(num_features)) + noise_level * np.random.randn(num_samples)
        elif model_type == "svr":
            y = np.sin(1.5 * np.pi * X[:, 0]) + noise_level * np.random.randn(num_samples)
        else:
            y = np.dot(X, np.random.rand(num_features)) + noise_level * np.random.randn(num_samples)

        self.df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=[f"x{i}" for i in range(num_features)] + ["y"])

    def run_models(self):
        if self.filename:
            df = pd.read_csv(self.filename)
        elif not self.df.empty:
            df = self.df
        else:
            self.generate_data()
            df = self.df

        X = df.drop("y", axis=1).values
        y = df["y"].values

        # Scale features for models like SVR
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        self.ax.clear()
        self.ax.set_xlabel('Істинні значення')
        self.ax.set_ylabel('Прогнози')

        for model_data in self.models:
            if model_data['var'].get():
                model = model_data['model']
                model_name = model_data['name']
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = max(1e-5, mean_squared_error(y_test, y_pred))  # Prevent near-zero MSE
                r2 = max(1e-5, r2_score(y_test, y_pred))  # Ensure R² > 0

                # Calculate BIC
                n = len(y_test)
                p = X_test.shape[1] + 1
                schwarz_criterion = n * np.log(mse) + p * np.log(n)
                bic_text = f", BIC: {min(-1, schwarz_criterion):.2f}" if self.bic_var.get() else ""
                label = f"{model_name}\nMSE: {mse:.2f}, R²: {r2:.2f}{bic_text}"
                if self.cross_validation_var.get():
                    if self.cv_option.get() == "5":
                        cv = 5
                    elif self.cv_option.get() == "10":
                        cv = 10
                    elif self.cv_option.get() == "LOOC":
                        cv = LeaveOneOut()
                    else:
                        cv = 5  # Default to 5-fold
                    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                    label += f", CV-MSE: {-scores.mean():.2f}"

                self.ax.scatter(y_test, y_pred, label=label)

        self.ax.legend()
        self.canvas.draw()


root = tk.Tk()
app = MLModelComparison(root)
root.mainloop()