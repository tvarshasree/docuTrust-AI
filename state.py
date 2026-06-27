from typing import List, Any, TypedDict

class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Any]
    relevance_scores: List[float]
    attempts: int
    graded_valid: bool
    answer: str
    logs: List[str]
