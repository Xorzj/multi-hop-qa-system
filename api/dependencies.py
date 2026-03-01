from __future__ import annotations

from fastapi import Request

from src.knowledge_graph.graph_retriever import GraphRetriever
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.llm.base_client import BaseLLMClient
from src.qa_engine.answer_generator import AnswerGenerator
from src.qa_engine.context_assembler import ContextAssembler
from src.qa_engine.question_parser import QuestionParser
from src.reasoning.reasoning_orchestrator import ReasoningOrchestrator


def get_neo4j_client(request: Request) -> Neo4jClient | None:
    return getattr(request.app.state, "neo4j_client", None)


def get_llm_client(request: Request) -> BaseLLMClient:
    return request.app.state.llm_client


def get_graph_retriever(request: Request) -> GraphRetriever | None:
    neo4j_client = get_neo4j_client(request)
    if neo4j_client is None:
        return None
    return GraphRetriever(neo4j_client)


def get_question_parser(request: Request) -> QuestionParser:
    return QuestionParser(get_llm_client(request))


def get_reasoning_orchestrator(request: Request) -> ReasoningOrchestrator | None:
    graph_retriever = get_graph_retriever(request)
    if graph_retriever is None:
        return None
    return ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=get_llm_client(request),
    )


def get_context_assembler() -> ContextAssembler:
    return ContextAssembler()


def get_answer_generator(request: Request) -> AnswerGenerator:
    return AnswerGenerator(get_llm_client(request))
