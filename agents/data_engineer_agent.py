from typing import Annotated, Any, Dict, List, Optional, Tuple
import pandas as pd  # type: ignore
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from models.agent_dependencies import DataEngineeringDeps
from pydantic_ai.models.openai import OpenAIModel

from config import openrouter_api_key, openrouter_base_url
from tools.data_engineer_tools import DataEngineerTools


class DataFrameModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


DataFrame = Annotated[pd.DataFrame, DataFrameModel]

data_engineering_agent = Agent(
    model=OpenAIModel(
        "openai/gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Data Engineer responsible for data processing and preparation. You work with local files and organize outputs in a structured directory system.

Core Capabilities:
1. Read and process input files
2. Data Processing
3. Access to read/write to local storage
4. Output Management:
   - Save processed data as CSV/Excel
   - Generate quality reports as JSON/text
   - Maintain organized directory structure:
     └── project_directory/
         ├── raw_data/
         ├── processed_data/
         ├── reports/
         └── temp/

When handling tasks:
1. First validate input file formats and contents
2. Apply appropriate data cleaning and transformation
3. Save outputs with clear naming and metadata
4. Generate comprehensive quality reports
5. Maintain data lineage documentation""",
    deps_type=DataEngineeringDeps,
)


@data_engineering_agent.tool
async def process_datasets(
    ctx: RunContext[DataEngineeringDeps],
    operations: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Read multiple input files of different types and process datasets with specified operations and return the processed dataset and the name of the processed dataset.
    Run this process only once for all files in starting the project.

    Args:
        ctx: Run context containing dependencies
        operations: List of operations to apply (e.g. ["clean", "normalize"])
        metadata: Optional metadata to save with the processed dataset

    Returns:
        path where processed datasets were saved
    """
    return await DataEngineerTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    ).process_datasets(ctx, operations=operations, metadata=metadata)


@data_engineering_agent.tool
async def generate_data_quality_report(
    ctx: RunContext[DataEngineeringDeps], name: str
) -> str:
    """
    Generate a data quality report for a dataset after processing and return the path to the report
    Run this process only once for all files after the data processing is done.
    Args:
        ctx: Run context containing dependencies
        name: Name for the processed dataset

    Returns:
        Path where data quality report was saved
    """
    return await DataEngineerTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    ).generate_data_quality_report(ctx, name=name)
