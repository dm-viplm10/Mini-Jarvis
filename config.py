import os
import nest_asyncio  # type: ignore
from llama_index.embeddings.nomic import NomicEmbedding

nest_asyncio.apply()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"

nomic_api_key = os.getenv("NOMIC_API_KEY")

github_token = os.getenv("GITHUB_TOKEN")

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=768,
    model_name="nomic-embed-text-v1.5",
)
