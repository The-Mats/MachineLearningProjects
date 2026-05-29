import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
)

# --- Imports needed for the function ---
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Import for the progress bar
from tqdm import tqdm


def build_nn_model(
    input_dim, layers=[64, 32], activation="relu", dropout_rate=0.2, optimizer="adam"
):
    """Builds the Keras Sequential model."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in layers:
        model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))  # Binary classification
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def run_models(
    x_train,
    y_train,
    x_test,
    y_test,
    seed,
    n_jobs,
    class_weight=None,
    preprocessor_name="",
):
    """
    Trains multiple models, collects their performance, and plots a comparison.
    """

    # --- 1. Prepare class_weight argument ---
    cw_arg = {}
    if class_weight is not None:
        cw_arg["class_weight"] = class_weight

    # --- 2. Define Models ---
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=seed, max_iter=1000, n_jobs=n_jobs, **cw_arg
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, **cw_arg),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_jobs=n_jobs
        ),  # KNN does not support class_weight
        "Random Forest": RandomForestClassifier(
            random_state=seed, n_jobs=n_jobs, **cw_arg
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=seed
        ),  # GB needs weights in .fit(), not __init__
        "MLP Classifier": MLPClassifier(
            random_state=seed, max_iter=1000
        ),  # MLP does not support class_weight
    }

    results = {}

    # --- 3. Training loop ---
    print(f"Training models for {preprocessor_name}...")

    for model_name, model in tqdm(models.items(), desc="Training Models"):

        # Special Handling: Gradient Boosting with Balanced Weights
        if model_name == "Gradient Boosting" and class_weight == "balanced":
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(x_train, y_train, sample_weight=weights)
        else:
            # Standard fit for LogReg, RF, KNN, DT, MLP
            model.fit(x_train, y_train)

        # --- 4. Predictions ---
        y_pred = model.predict(x_test)

        # Get Probabilities for AUROC/AP
        if hasattr(model, "predict_proba"):
            # Logistic, RF, MLP, KNN, GB all support this
            y_probs = model.predict_proba(x_test)[:, 1]
        else:
            # Fallback (SVC without probability=True, etc)
            y_probs = (
                model.decision_function(x_test)
                if hasattr(model, "decision_function")
                else y_pred
            )

        # --- 5. Collect Metrics ---
        results[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUROC": roc_auc_score(y_test, y_probs),
            "AP_Score": average_precision_score(y_test, y_probs),
            "F1_Binary": f1_score(y_test, y_pred, pos_label=1),
        }

    # --- Plotting Section ---

    # 1. Convert results dictionary to a DataFrame
    results_df = pd.DataFrame(results).T.reset_index()
    results_df = results_df.rename(columns={"index": "Model"})

    # Add Preprocessor name for the global dataframe later
    results_df["Preprocessor"] = preprocessor_name

    # 2. "Melt" for Seaborn
    results_long_df = pd.melt(
        results_df,
        id_vars=["Model", "Preprocessor"],
        var_name="Metric",
        value_name="Score",
    )

    # 3. Create the bar plot
    print("Generating model comparison plot...")
    plt.figure(figsize=(12, 8))
    ax = sb.barplot(
        data=results_long_df,
        x="Score",
        y="Model",
        hue="Metric",
        orient="h",
        palette="viridis",
    )

    # --- 4. Add text labels ---
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(
                width + 0.01,
                p.get_y() + p.get_height() / 2.0,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                color="black",
            )

    # 5. Customize
    plt.title(f"Model Performance Comparison ({preprocessor_name})")
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.xlim(0, 1.15)  # Extra room for labels

    plt.legend(title="Metric", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save with dynamic name so they don't overwrite each other in the loop
    filename = (
        f"compare_{preprocessor_name.replace(' ', '_').replace('+', '').strip()}.png"
    )
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()

    return results_df
