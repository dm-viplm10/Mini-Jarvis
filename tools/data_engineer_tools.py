import glob
import os
from pydantic_ai import RunContext
from models.agent_dependencies import DataEngineeringDeps
from .data_team_tools import DataTools
import pandas as pd  # type: ignore
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json


class DataEngineerTools(DataTools):
    async def process_datasets(
        self,
        ctx: RunContext[DataEngineeringDeps],
        operations: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process dataset with specified operations and save results."""
        try:
            data_dict = self.read_input_files(ctx)
            for file_name, file_data in data_dict.items():
                if file_data["type"] == "tabular":
                    processed_data = file_data["data"].copy()

                    for operation in operations:
                        if operation == "clean":
                            processed_data = processed_data.dropna()
                            processed_data = processed_data.drop_duplicates()
                        elif operation == "normalize":
                            numeric_cols = processed_data.select_dtypes(
                                include=[np.number]
                            ).columns
                            processed_data[numeric_cols] = (
                                processed_data[numeric_cols]
                                - processed_data[numeric_cols].mean()
                            ) / processed_data[numeric_cols].std()

                # Save processed dataset
                output_path = await self.save_output(
                    data=processed_data,
                    name=f"{file_name.split('.')[0]}_processed",
                    output_type="tabular",
                    subdirectory="processed",
                    metadata={
                        "operations": operations,
                        "original_shape": file_data["data"].shape,
                        "processed_shape": len(processed_data),
                        **(metadata or {}),
                    },
                )

            return f"Processed datasets saved to {output_path}"
        except Exception as e:
            return f"Error processing datasets: {str(e)}. Move to the next step."

    async def generate_data_quality_report(
        self, ctx: RunContext[DataEngineeringDeps], name: str
    ) -> str:
        """Generate and save data quality report."""
        try:
            reports = {}
            path = os.path.join(
                ctx.deps.base_output_dir,
                f"{ctx.deps.session_id}/processed_data/processed",
            )
            file_paths = glob.glob(os.path.join(path, "*.csv")) + glob.glob(
                os.path.join(path, "*.xlsx")
            )

            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                if file_path.endswith(".csv"):
                    data = pd.read_csv(file_path)  # type: ignore
                else:
                    data = pd.read_excel(file_path)  # type: ignore

                reports[file_name] = {
                    "basic_stats": {
                        "rows": int(len(data)),
                        "columns": int(len(data.columns)),
                        "missing_values": {
                            k: int(v) for k, v in data.isnull().sum().to_dict().items()
                        },
                        "duplicates": int(data.duplicated().sum()),
                    },
                    "column_stats": {
                        col: {
                            "dtype": str(data[col].dtype),
                            "unique_values": int(data[col].nunique()),
                            "missing_percentage": float(
                                (data[col].isnull().sum() / len(data)) * 100
                            ),
                        }
                        for col in data.columns
                    },
                }

            # Save consolidated report
            output_path = await self.save_output(
                data=reports,
                name=f"{name}_quality_report",
                output_type="text",
                subdirectory="quality_reports",
            )

            return f"Quality report saved to {output_path}"
        except Exception as e:
            return f"Error generating quality report: {str(e)}. Move to the next step."
