import base64
import io
import re
from typing import Dict, Any

from pydantic_ai import RunContext
from db.supabase_client import supabase

import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json
import shutil
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from models.agent_dependencies import DataEngineeringDeps  # type: ignore


class DataTools:
    def __init__(self, base_output_dir: str, session_id: str):
        """
        Initialize with base output directory for all results.

        Args:
            base_output_dir: Base directory path where all outputs will be stored
        """
        self.base_dir = Path(base_output_dir)
        self.session_id = session_id
        self._initialize_directory_structure(session_id)

    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        try:
            results = {}
            for item in directory.glob("**/*"):
                # Skip raw_data directory
                if "raw_data" in str(item):
                    continue

                if item.is_file():
                    # Skip metadata files
                    if item.name.endswith(".metadata.json"):
                        continue

                    # Get timestamp from filename
                    timestamp_match = re.search(r"_(\d{8}_\d{6})\.", item.name)
                    if timestamp_match:
                        timestamp = datetime.strptime(
                            timestamp_match.group(1), "%Y%m%d_%H%M%S"
                        )

                        # Get relative path from base directory
                        rel_path = str(item.relative_to(directory))

                        # Check for associated metadata
                        metadata_file = item.with_suffix(".metadata.json")
                        metadata = None
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)

                        file_info = {
                            "timestamp": timestamp.isoformat(),
                            "size": item.stat().st_size,
                            "metadata": metadata,
                            "filename": item.name,
                        }

                        # Read sample data for CSV/Excel files
                        if item.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                            try:
                                if item.suffix.lower() == ".csv":
                                    df = pd.read_csv(item)
                                else:
                                    df = pd.read_excel(item)

                                # Get all columns
                                file_info["columns"] = df.columns.tolist()

                                # Get random 50 rows if file has more than 50 rows
                                if len(df) > 50:
                                    sample_df = df.sample(n=50, random_state=42)
                                else:
                                    sample_df = df

                                file_info["sample_data"] = sample_df.to_dict("records")
                            except Exception as e:
                                file_info["error"] = f"Failed to read file: {str(e)}"

                        results[
                            f"{self.base_dir.name}/{self.session_id}/{directory.name}/{rel_path}"
                        ] = file_info
            return results
        except Exception as e:
            return {}

    def _initialize_directory_structure(self, session_id: str):
        """Create the initial directory structure for outputs."""
        try:
            # Create main directories
            directories = {
                "raw_data": self.base_dir / session_id / "raw_data",
                "processed_data": self.base_dir / session_id / "processed_data",
                "analysis": self.base_dir / session_id / "analysis",
                "models": self.base_dir / session_id / "models",
                "visualizations": self.base_dir / session_id / "visualizations",
                "reports": self.base_dir / session_id / "reports",
                "temp": self.base_dir / session_id / "temp",
            }

            for dir_path in directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)

            self.directories = directories
        except Exception as e:
            raise Exception(f"Error initializing directory structure: {str(e)}")

    def convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        try:
            if isinstance(obj, np.integer):
                return str(obj)
            elif isinstance(obj, np.floating):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: self.convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self.convert_to_serializable(item) for item in obj]
            return obj
        except Exception as e:
            return str(obj)

    def read_input_files(self, ctx: RunContext[DataEngineeringDeps]) -> Dict[str, Any]:
        """
        Read multiple input files of different types.

        Returns:
            Dictionary containing loaded data with file names as keys
        """
        try:
            files = (
                supabase.table("messages")
                .select("message")
                .eq("session_id", ctx.deps.session_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
                .data[0]["message"]["data"]["files"]
            )

            if len(files) > 5:
                raise ValueError("Maximum 5 files can be processed at once")

            path = self.directories["raw_data"]

            data_dict = {}
            for i, file in enumerate(files):
                file_type = file["type"]

                try:
                    # Read file based on type
                    if "csv" in file_type or "excel" in file_type:
                        df = self.process_file_to_dataframe(file)
                        data_dict[file["name"]] = {
                            "type": "tabular",
                            "data": df,
                            "base64": file["base64"],
                        }

                        df.to_csv(path / file["name"], index=False)

                    elif file_type in ["text/plain"]:
                        # Decode base64 text content
                        content = base64.b64decode(file["base64"]).decode("utf-8")
                        data_dict[file["name"]] = {
                            "type": "text",
                            "data": content,
                            "base64": file["base64"],
                        }

                        with open(path / file["name"], "w") as f:
                            f.write(content)

                    elif file_type in ["image/jpeg", "image/png"]:
                        # Decode base64 image data
                        img_data = file["img_data"]
                        data_dict[file["name"]] = {
                            "type": "image",
                            "data": img_data,
                            "base64": file["base64"],
                        }

                        img_data = base64.b64decode(file["base64"])
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        cv2.imwrite(str(path / file["name"]), img)

                    else:
                        raise ValueError(f"Unsupported file type: {file_type}")

                except Exception as e:
                    data_dict[file["name"]] = {
                        "type": "error",
                        "error": f"Error processing file: {str(e)}",
                    }

            return data_dict
        except Exception as e:
            return {"error": f"Error reading input files: {str(e)}"}

    async def save_output(
        self,
        data: Any,
        name: str,
        output_type: str,
        subdirectory: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save output data to appropriate directory with metadata.

        Args:
            data: Data to save
            name: Name for the output file
            output_type: Type of output (tabular, figure, text, model)
            subdirectory: Optional subdirectory within the output type directory
            metadata: Optional metadata to save with the output

        Returns:
            Path to saved output
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine base directory based on output type
            if output_type == "tabular":
                base_dir = self.directories["processed_data"]
                extension = ".csv"
            elif output_type == "figure":
                base_dir = self.directories["visualizations"]
                extension = ".png"
            elif output_type == "text":
                base_dir = self.directories["reports"]
                extension = ".txt"
            elif output_type == "model":
                base_dir = self.directories["models"]
                extension = ".joblib"
            else:
                raise ValueError(f"Unsupported output type: {output_type}")

            # Create subdirectory if specified
            if subdirectory:
                base_dir = base_dir / subdirectory
                base_dir.mkdir(exist_ok=True)

            # Create output filename
            filename = f"{name.replace('.','_')}_{timestamp}{extension}"
            output_path = base_dir / filename

            # Save data based on type
            if output_type == "tabular":
                if isinstance(data, pd.DataFrame):
                    data.to_csv(output_path, index=False)
                else:
                    pd.DataFrame(data).to_csv(output_path, index=False)

            elif output_type == "figure":
                if isinstance(data, (plt.Figure, sns.FacetGrid)):
                    data.savefig(output_path)
                else:
                    plt.figure()
                    plt.imshow(data)
                    plt.savefig(output_path)
                plt.close()

            elif output_type == "text":
                if isinstance(data, (dict, list)):
                    with open(output_path, "w") as f:
                        json.dump(data, f, indent=2)
                else:
                    with open(output_path, "w") as f:
                        f.write(str(data))

            elif output_type == "model":
                joblib.dump(data, output_path)

            # Save metadata if provided
            if metadata:
                metadata_path = output_path.with_suffix(".metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(
                        {
                            "filename": filename,
                            "timestamp": timestamp,
                            "type": output_type,
                            **self.convert_to_serializable(metadata),
                        },
                        f,
                        indent=2,
                    )

            return str(output_path)
        except Exception as e:
            return f"Error saving output: {str(e)}"

    def process_file_to_dataframe(self, file: Dict[str, Any]) -> pd.DataFrame:
        """Convert a base64 encoded CSV/Excel file into a pandas DataFrame."""
        try:
            decoded_content = base64.b64decode(file["base64"])
            buffer = io.BytesIO(decoded_content)
            file_name = file["name"].lower()

            if file_name.endswith(".csv"):
                return pd.read_csv(buffer)
            elif file_name.endswith((".xls", ".xlsx")):
                return pd.read_excel(buffer)
            else:
                raise ValueError(
                    f"Unsupported file format for {file_name}. Please use CSV or Excel files."
                )
        except Exception as e:
            raise Exception(f"Error processing file to dataframe: {str(e)}")
