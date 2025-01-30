import json
from pydantic_ai.usage import UsageLimits
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from models.agent_dependencies import (
    ProductManagerDeps,
)
from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import RunContext
from typing import List, Dict, Any

from agents.data_engineer_agent import data_engineering_agent
from agents.data_scientist_agent import data_scientist_agent
from tools.data_team_tools import DataTools


product_manager_agent = Agent(
    model=OpenAIModel(
        "openai/gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Product Manager leading a data team consisting of a Data Engineer and Data Scientist. You coordinate their efforts to solve complex data problems.

Core Responsibilities:
1. Problem Analysis:
   - Break down complex data requests
   - Identify required data sources
   - Plan analysis approach
   - Set quality standards

2. Team Coordination:
   - Assign tasks to appropriate team members
   - Manage dependencies between tasks
   - Ensure output quality and consistency
   - Synthesize results into business insights

3. File Handling:
   - Identify the correct inputs for each tool and pass them to the tool along with the input query

4. Project Structure:
   - Maintain organized project structure:
     └── project_directory/
         ├── raw_data/
         ├── processed_data/
         ├── analysis/
         ├── models/
         ├── visualizations/
         ├── reports/
         └── temp/  

Be very precise with your analysis and decide what tools to use based on the query and the data available. Keep the tools use to minimal and only use the tools that are necessary to complete the task.
""",
    deps_type=ProductManagerDeps,
)


@product_manager_agent.tool
async def create_folder_structure(
    ctx: RunContext[ProductManagerDeps], query: str
) -> str:
    """
    create a folder structure for the project
    """
    try:
        data_tools = DataTools(
            base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
        )
        return "Folder structure created successfully"
    except Exception as e:
        return f"Error creating folder structure: {e}"


@product_manager_agent.tool
async def run_data_engineering_agent(
    ctx: RunContext[ProductManagerDeps], query: str, file_names: List[str]
) -> str:
    """
    You are a Data Engineer whose expertise is in data engineering includes
    - Extracting data from the files provided as context
    - Data Quality Validation
    - Data Cleaning
    - Dataset Merging
    - Data Summarization
    - Access to read/write to local storage
    """
    response = await data_engineering_agent.run(
        user_prompt=f"Perform the required actions based on the inputs provided. Identify the correct inputs for each tool and pass them to the tool along with the input query: {query} and the file names: {file_names}",
        deps=ctx.deps.data_engineer_deps,
    )
    return response.data


@product_manager_agent.tool
async def run_data_scientist_agent(
    ctx: RunContext[ProductManagerDeps], query: str
) -> str:
    """
    You are a Data Scientist whose expertise is in data science includes:
    - Machine learning algorithms
    - Statistical modeling
    - Data visualization
    - Deep learning
    - Model deployment
    - Research and experimentation

    You have been passed along the cleaned and processed data. Use it to perform the required actions.
    """
    # Get available datasets from each directory with timestamps
    data_tools = DataTools(
        base_output_dir=ctx.deps.base_output_dir, session_id=ctx.deps.session_id
    )
    datasets = {
        "raw_data": data_tools.scan_directory(data_tools.directories["raw_data"]),
        "processed_data": data_tools.scan_directory(
            data_tools.directories["processed_data"]
        ),
        "models": data_tools.scan_directory(data_tools.directories["models"]),
        "reports": data_tools.scan_directory(data_tools.directories["reports"]),
        "visualizations": data_tools.scan_directory(
            data_tools.directories["visualizations"]
        ),
    }

    # Log available datasets
    print(f"Available datasets:\n{json.dumps(datasets, indent=2)}")
    response = await data_scientist_agent.run(
        user_prompt=f"""You have been shared the existing datasets available for analysis and the required actions. The datasets contain atleast 50 random rows of data. You can infer the information about the data from the sample data as the complete dataset can be huge. Only perform the required actions based on the inputs provided and the datasets available for reference. Identify the correct inputs for each tool and pass them to the tool along with the input query: {query}.
        Even if there are some errors in the process, you can still proceed with the analysis and report both the errors and the analysis in the final report.""",
        deps=ctx.deps.data_scientist_deps,
    )
    return response.data
