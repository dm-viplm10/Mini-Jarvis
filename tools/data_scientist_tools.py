from .data_team_tools import DataTools
import pandas as pd  # type: ignore
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor  # type: ignore
from sklearn.linear_model import LogisticRegression, LinearRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from datetime import datetime
import joblib  # type: ignore
from scipy import stats  # type: ignore


class DataScientistTools(DataTools):
    async def analyze_data_distribution(
        self, dataset_path: str, columns: List[str], name: str
    ) -> str:
        """
        Analyze the distribution of specified columns with statistical tests.

        Args:
            dataset_path: Path to the dataset
            columns: List of columns to analyze
            name: Name prefix for saved files

        Returns:
            str: Path to the saved analysis results
        """
        try:
            df = (
                pd.read_csv(dataset_path)
                if dataset_path.endswith(".csv")
                else pd.read_excel(dataset_path)
            )

            analysis_results = {}
            for col in columns:
                if col not in df.columns:
                    continue
                # Basic statistics
                try:
                    stats_dict = {
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "std": float(df[col].std()),
                        "skewness": float(stats.skew(df[col].dropna())),
                        "kurtosis": float(stats.kurtosis(df[col].dropna())),
                        "normality_test": stats.normaltest(df[col].dropna())[1],
                    }

                    # Create distribution plot
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x=col, kde=True)
                    plt.title(f"Distribution of {col}")

                    # Save plot
                    plot_path = await self.save_output(
                        data=plt.gcf(),
                        name=f"{name}_{col}_distribution",
                        output_type="figure",
                        subdirectory="distributions",
                    )
                    plt.close()

                    analysis_results[col] = {
                        "statistics": stats_dict,
                        "plot_path": plot_path,
                    }
                except Exception as e:
                    print(f"Error: {e}")
                    pass

            # Save analysis results
            return await self.save_output(
                data=analysis_results,
                name=f"{name}_distribution_analysis",
                output_type="text",
                subdirectory="analysis",
            )
        except Exception as e:
            return "Analysed with some errors. Please check the logs for more details."

    async def perform_feature_engineering(
        self,
        dataset_path: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
        name: str,
        scaling_method: str = "standard",
        create_interactions: bool = False,
    ) -> str:
        """
        Perform feature engineering on a dataset by encoding categorical variables, scaling numerical variables,
        and splitting into train/test sets.

        Args:
            dataset_path: Path to dataset
            numerical_columns: List of numerical columns
            categorical_columns: List of categorical columns
            name: Name prefix for saved files
            scaling_method: Method for scaling ('standard', 'minmax', 'robust')
            create_interactions: Whether to create interaction features
        """
        try:
            df = (
                pd.read_csv(dataset_path)
                if dataset_path.endswith(".csv")
                else pd.read_excel(dataset_path)
            )

            # Store original columns
            original_columns = df.columns.tolist()

            # Handle numerical features
            if numerical_columns:
                if scaling_method == "standard":
                    scaler = StandardScaler()
                elif scaling_method == "minmax":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()

                numerical_columns = [
                    col for col in numerical_columns if col in df.columns
                ]
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

            # Handle categorical features
            encoders = {}
            if categorical_columns:
                categorical_columns = [
                    col for col in categorical_columns if col in df.columns
                ]
                for col in categorical_columns:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    encoders[col] = le

            # Create interaction features if requested
            if create_interactions and len(numerical_columns) > 1:
                for i in range(len(numerical_columns)):
                    for j in range(i + 1, len(numerical_columns)):
                        col1, col2 = numerical_columns[i], numerical_columns[j]
                        df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]

            # Save preprocessed data
            output_path = await self.save_output(
                data=df,
                name=f"{name}_engineered",
                output_type="tabular",
                subdirectory="engineered_data",
                metadata={
                    "original_columns": original_columns,
                    "scaling_method": scaling_method,
                    "interaction_features": create_interactions,
                },
            )

            return output_path
        except Exception as e:
            print(f"Error: {e}")
            return "Feature engineering with some errors. Please check the logs for more details."

    async def train_and_evaluate_model(
        self,
        dataset_path: str,
        target_column: str,
        features: List[str],
        model_type: str,
        name: str,
        perform_cv: bool = True,
        hyperparameter_tuning: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train and evaluate a machine learning model with mixed data types.
        Automatically detects if the problem is classification or regression.
        """
        try:
            print(f"Loading dataset from {dataset_path}")
            df = (
                pd.read_csv(dataset_path)
                if dataset_path.endswith(".csv")
                else pd.read_excel(dataset_path)
            )

            if target_column not in df.columns:
                return "Target column not found in the dataset.", {}

            # Verify and filter features
            features = [col for col in features if col in df.columns]
            print(f"Features to be used: {features}")

            # Create feature matrix X and target y
            X = df[features].copy()
            y = df[target_column].copy()

            # Print data information
            print("Data types of columns:")
            print(X.dtypes)
            print(f"Target dtype: {y.dtype}")
            print(f"Number of unique target values: {len(y.unique())}")

            # Determine if this is a classification or regression problem
            is_classification = False
            if y.dtype == "object" or y.dtype.name == "category":
                is_classification = True
            else:
                # If numeric, check if it's discrete or continuous
                unique_count = len(y.unique())
                if unique_count < 10:  # Arbitrary threshold for classification
                    is_classification = True

            print(
                f"Problem type: {'Classification' if is_classification else 'Regression'}"
            )

            # Handle categorical features
            categorical_columns = X.select_dtypes(
                include=["object", "category"]
            ).columns
            print(f"Categorical columns: {categorical_columns}")

            # Create dummy variables for categorical features
            if not categorical_columns.empty:
                X = pd.get_dummies(X, columns=categorical_columns)

            # Convert features to float
            X = X.astype(float)

            # Handle target variable
            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y)
                print(f"Encoded target classes: {le.classes_}")
            else:
                y = y.astype(float)

            print(f"Final X shape: {X.shape}, y shape: {y.shape}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Initialize model based on problem type
            if is_classification:
                if model_type == "random_forest":
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=None, random_state=42
                    )
                else:  # gradient_boosting
                    model = GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42,
                    )
            else:
                if model_type == "random_forest":
                    model = RandomForestRegressor(
                        n_estimators=100, max_depth=None, random_state=42
                    )
                else:  # gradient_boosting
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42,
                    )

            print(f"Training model: {model.__class__.__name__}")

            # Fit the model
            model.fit(X_train, y_train)

            # Generate predictions
            y_pred = model.predict(X_test)

            # Calculate metrics based on problem type
            metrics = {}
            if is_classification:
                metrics["classification_report"] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

                # Add probability predictions if available
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)

                    # Generate ROC curve for binary classification
                    if len(le.classes_) == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        metrics["roc_auc"] = auc(fpr, tpr)
            else:

                metrics["mean_squared_error"] = mean_squared_error(y_test, y_pred)
                metrics["root_mean_squared_error"] = np.sqrt(
                    mean_squared_error(y_test, y_pred)
                )
                metrics["mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
                metrics["r2_score"] = r2_score(y_test, y_pred)

            # Save feature importance plot
            if hasattr(model, "feature_importances_"):
                importance = pd.DataFrame(
                    {"feature": X.columns, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)

                plt.figure(figsize=(12, 6))
                sns.barplot(data=importance.head(20), x="importance", y="feature")
                plt.title("Top 20 Feature Importance")
                plt.xticks(rotation=45)
                plt.tight_layout()

                importance_path = await self.save_output(
                    data=plt.gcf(),
                    name=f"{name}_feature_importance",
                    output_type="figure",
                    subdirectory="model_evaluation",
                )
                plt.close()

            # Save model and metrics
            model_artifacts = {
                "model": model,
                "feature_names": X.columns.tolist(),
                "metrics": metrics,
                "is_classification": is_classification,
                "target_encoder": (
                    le if is_classification and "le" in locals() else None
                ),
            }

            model_path = await self.save_output(
                data=model_artifacts,
                name=f"{name}_model",
                output_type="model",
                subdirectory="models",
                metadata={
                    "features": features,
                    "encoded_features": X.columns.tolist(),
                    "metrics": metrics,
                    "is_classification": is_classification,
                },
            )

            print(f"Model saved successfully at {model_path}")
            return model_path, metrics

        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return f"Model training failed: {str(e)}", {}

    async def create_feature_importance_analysis(
        self, model_path: str, features: List[str], name: str
    ) -> str:
        """
        Analyze and visualize feature importance from a trained model.

        Args:
            model_path: Path to saved model
            features: List of feature names
            name: Name prefix for saved files
        """
        # Load model
        model = joblib.load(model_path)

        # Get feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_[0])

        # Create importance DataFrame
        importance_df = pd.DataFrame(
            {"feature": features, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Create importance plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title("Feature Importance")

        # Save plot
        plot_path = await self.save_output(
            data=plt.gcf(),
            name=f"{name}_feature_importance",
            output_type="figure",
            subdirectory="feature_importance",
        )
        plt.close()

        # Save detailed results
        results = {
            "importance_values": importance_df.to_dict("records"),
            "plot_path": plot_path,
        }

        return await self.save_output(
            data=results,
            name=f"{name}_importance_analysis",
            output_type="text",
            subdirectory="analysis",
        )

    async def generate_prediction_insights(
        self, model_path: str, test_data_path: str, features: List[str], name: str
    ) -> str:
        """
        Generate insights from model predictions on test data.

        Args:
            model_path: Path to saved model
            test_data_path: Path to test dataset
            features: List of feature columns
            name: Name prefix for saved files
        """
        # Load model and data
        model = joblib.load(model_path)
        df = (
            pd.read_csv(test_data_path)
            if test_data_path.endswith(".csv")
            else pd.read_excel(test_data_path)
        )

        # Generate predictions
        X_test = df[features]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Add predictions to dataframe
        df["predicted_class"] = predictions
        df["prediction_probability"] = probabilities[:, 1]

        # Generate insights
        insights = {
            "prediction_distribution": df["predicted_class"].value_counts().to_dict(),
            "high_confidence_predictions": len(df[df["prediction_probability"] > 0.9]),
            "low_confidence_predictions": len(df[df["prediction_probability"] < 0.6]),
        }

        # Create prediction distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="prediction_probability", bins=30)
        plt.title("Distribution of Prediction Probabilities")

        # Save plot
        plot_path = await self.save_output(
            data=plt.gcf(),
            name=f"{name}_prediction_distribution",
            output_type="figure",
            subdirectory="predictions",
        )
        plt.close()

        # Save results
        results = {
            "insights": insights,
            "plot_path": plot_path,
            "predictions_sample": df[["predicted_class", "prediction_probability"]]
            .head(10)
            .to_dict("records"),
        }

        return await self.save_output(
            data=results,
            name=f"{name}_prediction_insights",
            output_type="text",
            subdirectory="analysis",
        )
