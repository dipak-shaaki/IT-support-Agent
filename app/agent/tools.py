import json

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.core.config import get_settings
from app.rag.vectorstore import get_retriever

settings = get_settings()
_web_search = None


def web_search_provider():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


@tool
def search_private_kb(query: str) -> str:
    """Search the company's private IT knowledge base for policies and support guidance."""
    documents = get_retriever().invoke(query)
    return json.dumps(
        {
            "documents": [
                {
                    "source": document.metadata.get("source", "Private KB"),
                    "content": document.page_content,
                }
                for document in documents
            ]
        },
        ensure_ascii=False,
    )


@tool
def search_web(query: str) -> str:
    """Search public web information after checking the private KB."""
    result = web_search_provider().invoke({"query": query})
    if isinstance(result, dict):
        answer = result.get("answer", "")
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in result.get("results", [])
        ]
    else:
        answer = ""
        results = [{"title": "", "url": "", "content": str(result)}]
    return json.dumps({"answer": answer, "results": results}, ensure_ascii=False)


TOOLS = [search_private_kb, search_web]
