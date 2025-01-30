from typing import Any, Dict, List, Optional, Tuple
from pydantic_ai import Agent, RunContext
from models.agent_dependencies import DataScientistDeps
from pydantic_ai.models.openai import OpenAIModel
from config import openrouter_api_key, openrouter_base_url
from tools.data_scientist_tools import DataScientistTools
from tools.data_team_tools import DataTools


data_scientist_agent = Agent(
    model=OpenAIModel(
        "openai/gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Data Scientist responsible for analysis, modeling, and insights generation. You work with processed data and create various outputs in a structured directory system.

Core Capabilities:
1. Data Analysis
2. Visualization:
   - Create plots and charts
   - Generate statistical visualizations
   - Save visualizations as PNG files
3. Machine Learning:
   - Train and evaluate models
   - Save models with metadata
   - Generate performance reports
4. Output Management:
   - Save in organized directories:
     └── project_directory/
         ├── analysis/
         ├── models/
         ├── visualizations/
         └── reports/

When handling tasks:
1. First, analyze which datasets are relevant to the task and tool
2. Pass the file paths of the relevant datasets to the appropriate tool based on the task
3. Begin by analyzing data quality and distribution
4. Create appropriate visualizations
5. Apply suitable modeling techniques
6. Save all outputs with detailed metadata
7. Precisely identify the parameters and the features for the model.
8. Use logistic regression if all features are numerical. Use random forest if there are categorical features. Use gradient boosting if there are both numerical and categorical features.
9. Highlight the best model and its metrics in the final report.

In the end you must decide what tools to use based on the query and the data available. Keep the tools use to minimal and only use the tools that are necessary to complete the task.
""",
    deps_type=DataScientistDeps,
)


@data_scientist_agent.tool
def scan_directory_for_latest_data(
    ctx: RunContext[DataScientistDeps], query: str
) -> dict:
    """
    Scan the directory for the latest engineered and analyzed data ONLY before training and evaluating a model. You must strictly follow this instruction.
    """
    data_tools = DataTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    )
    return data_tools.scan_directory(data_tools.directories["processed_data"])


@data_scientist_agent.tool
async def analyze_data_distribution(
    ctx: RunContext[DataScientistDeps],
    query: str,
    dataset_path: str,
    columns: List[str],
    name_prefix: str,
) -> str:
    """
    Analyze the distribution of specified columns in a dataset with statistical tests and visualizations.

    Args:
        ctx (RunContext[DataScientistDeps]): The run context containing dependencies
        query (str): The user query/request being processed
        dataset_path (str): Path to the dataset file to analyze
        columns (List[str]): List of column names to analyze
        name_prefix (str): Prefix to use for output file names

    Returns:
        str: Path to the saved analysis results file containing:
            - Basic statistics (mean, median, std, skewness, kurtosis)
            - Normality test results
            - Distribution plots for each column
    """
    return await DataScientistTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    ).analyze_data_distribution(dataset_path, columns, name_prefix)


@data_scientist_agent.tool
async def perform_feature_engineering(
    ctx: RunContext[DataScientistDeps],
    query: str,
    dataset_path: str,
    categorical_columns: List[str],
    numerical_columns: List[str],
    name_prefix: str,
) -> str:
    """
    Perform feature engineering on a dataset by encoding categorical variables, scaling numerical variables,
    and splitting into train/test sets.

    Args:
        dataset_path: Path to dataset to perform feature engineering on
        numerical_columns: List of numerical columns
        categorical_columns: List of categorical columns
        name: Name prefix for saved files
        scaling_method: Method for scaling ('standard', 'minmax', 'robust')
        create_interactions: Whether to create interaction features
    Returns:
        str: Path to the saved feature engineering results file containing:
            - Encoded categorical variables
            - Scaled numerical variables
            - Train/test splits
    """
    return await DataScientistTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    ).perform_feature_engineering(
        dataset_path,
        numerical_columns,
        categorical_columns,
        name_prefix,
    )


@data_scientist_agent.tool
async def train_and_evaluate_model(
    ctx: RunContext[DataScientistDeps],
    query: str,
    dataset_path: str,
    target_column: str,
    features: List[str],
    model_type: str,
    name: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Train and evaluate a machine learning model.

    Args:
        dataset_path: Path to engineered dataset to train and evaluate model on
        target_column: Name of the target/dependent variable column
        features: List of feature columns
        model_type: Type of model ('random_forest', 'logistic_regression', 'gradient_boosting'). Use logistic regression if all features are numerical. Use random forest or naive-bayesif there are categorical features. Use gradient boosting if there are both numerical and categorical features.
        name: Name prefix for saved files
        perform_cv: Whether to perform cross-validation
        hyperparameter_tuning: Whether to perform hyperparameter tuning based on the dataset and model type

    Returns:
        Tuple[str, Dict[str, Any]]: Path to the saved model file and the model metrics
    """
    return await DataScientistTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    ).train_and_evaluate_model(
        dataset_path,
        target_column,
        features,
        model_type,
        name,
    )
