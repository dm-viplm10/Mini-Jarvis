import os
from llama_index.embeddings.nomic import NomicEmbedding

base_output_dir = "data"

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"

nomic_api_key = os.getenv("NOMIC_API_KEY")

github_token = os.getenv("GITHUB_TOKEN")
brave_api_key = os.getenv("BRAVE_API_KEY")

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=768,
    model_name="nomic-embed-text-v1.5",
)
