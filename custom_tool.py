import json
import os
import hashlib
import uuid
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict, field_validator
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient

class DocumentSearchResult(BaseModel):
    """Structured result for document search operations."""
    ok: bool
    text: str = ""
    error: str | None = None

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

    @field_validator("query", mode="before")
    @classmethod
    def coerce_query(cls, value):
        def unwrap(val):
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                if "query" in val:
                    return unwrap(val.get("query"))
                if isinstance(val.get("description"), str):
                    return val["description"]
                if isinstance(val.get("text"), str):
                    return val["text"]
            return val

        coerced = unwrap(value)
        if isinstance(coerced, str):
            return coerced
        return str(coerced)


class FireCrawlWebSearchToolInput(BaseModel):
    """Input schema for FireCrawlWebSearchTool."""
    query: str = Field(..., description="Query to search the web.")
    limit: int = Field(5, description="Number of results to return.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")  # For small experiments
        self._has_chunks = False
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        if not raw_text.strip():
            self._has_chunks = False
            return
        chunks = self._create_chunks(raw_text)
        if not chunks:
            self._has_chunks = False
            return
        self._has_chunks = True
        
        docs = [chunk.text for chunk in chunks]
        # Stable, namespaced document id based on path + content fingerprint
        base_name = os.path.basename(self.file_path)
        doc_fingerprint = hashlib.sha1(
            f"{os.path.abspath(self.file_path)}|{raw_text[:1024]}".encode("utf-8")
        ).hexdigest()
        doc_id = f"{base_name}-{doc_fingerprint[:8]}"

        # Include richer metadata and unique UUID ids per chunk
        metadata = [
            {"source": base_name, "doc_id": doc_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
        ids = [
            uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{i}")
            for i in range(len(chunks))
        ]

        self.client.add(
            collection_name="demo_collection",
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: str) -> DocumentSearchResult:
        """Search the document and return a structured result."""
        if not self._has_chunks:
            return DocumentSearchResult(
                ok=False,
                text="",
                error=(
                    "No searchable text was extracted from the PDF. "
                    "Ensure the PDF contains selectable text or run OCR."
                ),
            )
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        docs = [chunk.document for chunk in relevant_chunks]
        separator = "\n___\n"
        joined = separator.join(docs).strip()
        if joined:
            return DocumentSearchResult(ok=True, text=joined)
        return DocumentSearchResult(ok=False, text="", error="No matching chunks found.")


class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "Search the web for the given query using FireCrawl."
    args_schema: Type[BaseModel] = FireCrawlWebSearchToolInput

    @staticmethod
    def is_enabled() -> bool:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return False
        lowered = api_key.strip().lower()
        return lowered not in {"your_firecrawl_api_key", "your_firecrawl_api_key_here"}

    def _run(self, query: str, limit: int = 5) -> str:
        if not self.is_enabled():
            return "FireCrawl web search is disabled because the API key is missing or uses the default placeholder."

        api_key = os.getenv("FIRECRAWL_API_KEY")

        payload = {
            "query": query,
            "limit": limit
        }

        request = Request(
            url="https://api.firecrawl.dev/v1/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST"
        )

        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return f"FireCrawl request failed with status {exc.code}."
        except URLError:
            return "FireCrawl request failed due to a network error."
        except json.JSONDecodeError:
            return "FireCrawl response could not be parsed."

        if not data.get("success"):
            return "FireCrawl search did not return results."

        results = data.get("data", [])
        if not results:
            return "FireCrawl search returned no results."

        formatted = []
        for item in results:
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            snippet = item.get("description") or item.get("snippet") or ""
            formatted.append(f"{title}\n{url}\n{snippet}".strip())

        return "\n___\n".join(formatted)

# Test the implementation
def test_document_searcher():
    # Test file path
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "static",
        "history_of_machine_learning.pdf"
    )

    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)

    # Test search
    result = searcher._run("Who was a prominent pioneer in machine learning?")
    if result.ok:
        print("Search Results:", result.text)
    else:
        print("PDF Search Error:", result.error)

if __name__ == "__main__":
    test_document_searcher()
