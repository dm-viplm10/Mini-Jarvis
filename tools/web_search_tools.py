import base64
import pickle
import logfire
import tiktoken
from bs4 import BeautifulSoup
from pydantic_ai import RunContext
from models.agent_dependencies import WebResearcherDeps
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
)
from config import embed_model
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.supabase import SupabaseVectorStore
from db.supabase_client import connection_string
import vecs  # type: ignore


vx = vecs.create_client(connection_string)

tokenizer = tiktoken.get_encoding("gpt2")
text_splitter_html = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=50
)
text_splitter_md = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024, chunk_overlap=100, separators=["|\n|"]
)

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "header2")])


# Split on H2, but merge small h2 chunks together to avoid too small.
async def split_html_on_h2(html, min_chunk_size=50, max_chunk_size=1024):
    if not html:
        return []
    h2_chunks = html_splitter.split_text(html)
    chunks = []
    previous_chunk = ""
    # Merge chunks together to add text before h2 and avoid too small docs.
    for c in h2_chunks:
        # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
        content = c.metadata.get("header2", "") + "\n" + c.page_content
        if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size / 2:
            previous_chunk += content + "\n"
        else:
            chunks.extend(text_splitter_html.split_text(previous_chunk.strip()))
            previous_chunk = content + "\n"
    if previous_chunk:
        chunks.extend(text_splitter_html.split_text(previous_chunk.strip()))
    # Discard too small chunks
    return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]


# Split on H1, but merge small h1 chunks together to avoid too small.
async def split_md_on_header(md, min_chunk_size=50, max_chunk_size=1024):
    if not md:
        return []
    h1_chunks = md_splitter.split_text(md)
    chunks = []
    previous_chunk = ""
    # Merge chunks together to add text before h1 and avoid too small docs.
    for c in h1_chunks:
        # Concat the h1 (note: we could remove the previous chunk to avoid duplicate h1)
        content = c.metadata.get("header1", "") + "\n" + c.page_content
        if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size / 2:
            previous_chunk += content + "\n"
        else:
            chunks.extend(text_splitter_md.split_text(previous_chunk.strip()))
            previous_chunk = content + "\n"
    if previous_chunk:
        chunks.extend(text_splitter_md.split_text(previous_chunk.strip()))
    # Discard too small chunks
    return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]


async def fetch_and_split_url_content(
    ctx: RunContext[WebResearcherDeps], web_results: list[dict]
) -> list:
    """Fetch the content of a URL and extract the main text."""
    docs: list = []
    for item in web_results[:3]:
        title = item.get("title", "")
        url = item.get("url", "")
        if title and url:
            with logfire.span("fetching URL content", url=url):
                try:
                    r = await ctx.deps.client.get(url)
                    soup = BeautifulSoup(r.text, "html.parser")
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    split_chunks = await split_html_on_h2(str(soup))
                    # Get text and clean it up
                    for i, text in enumerate(split_chunks):
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (
                            phrase.strip()
                            for line in lines
                            for phrase in line.split("  ")
                        )
                        text = " ".join(chunk for chunk in chunks if chunk)
                        docs.append(
                            (
                                f"{url}_{i}",
                                embed_model.get_text_embedding(text),
                                {"title": title, "url": url, "text": text},
                            )
                        )
                except Exception as e:
                    logfire.warning(f"Error fetching URL content: {str(e)}... Skipping")
    return docs


async def get_search_index(ctx: RunContext[WebResearcherDeps]):
    """Retrieve the query engine from Supabase"""
    try:
        return vx.get_or_create_collection(name=ctx.deps.session_id, dimension=768)
    except Exception as e:
        return None


async def get_index_ready(docs: list, session_id: str):
    vector_store = vx.get_or_create_collection(name=session_id, dimension=768)
    vector_store.upsert(docs)
    vector_store.create_index(
        method=vecs.IndexMethod.hnsw,
        measure=vecs.IndexMeasure.cosine_distance,
        index_arguments=vecs.IndexArgsHNSW(m=8),
    )
    return vector_store


async def retrieve_similar_search_results(index, query: str) -> str:
    results = index.query(
        data=embed_model.get_text_embedding(query), limit=10, include_metadata=True
    )
    combined_text = ""
    total_tokens = 0

    for result in results:
        text = result[1]["text"]
        tokens = len(tokenizer.encode(text))

        if total_tokens + tokens <= 4000:
            # If entire result fits, add it completely
            combined_text += text + "\n\n"
            total_tokens += tokens
        else:
            # For the last result that would exceed the limit,
            # add as much as possible while staying under 4000 tokens
            remaining_tokens = 4000 - total_tokens
            if remaining_tokens > 0:
                # Encode the full text and take only the tokens we need
                all_tokens = tokenizer.encode(text)
                partial_tokens = all_tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                combined_text += partial_text + "..."
            break

    return combined_text.strip()


async def fetch_web_rag_search_result(
    ctx: RunContext[WebResearcherDeps], web_results: list[dict], query: str
) -> str:
    """Fetch and process web search results using RAG.

    Args:
        ctx: The context.
        web_results: List of web search results containing titles and URLs.
        query: The search query.

    Returns:
        str: The rag based search result content.
    """
    try:
        docs = await fetch_and_split_url_content(ctx, web_results)
        index = await get_index_ready(docs, ctx.deps.session_id)
        content = await retrieve_similar_search_results(index, query)
        vx.disconnect()

        return content
    except Exception as e:
        return f"Error fetching content: {str(e)}"
