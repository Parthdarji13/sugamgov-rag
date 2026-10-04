"""
api/chat/routes.py
==================
FastAPI router for conversational chat endpoints:
  POST   /api/chat              - Multi-turn grounded conversational chat
  DELETE /api/chat/{session_id} - Terminate and remove conversation session
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path
from fastapi.responses import StreamingResponse

from api.chat.models import ChatRequest, ChatResponse, DeleteSessionResponse
from api.chat.session_store import InMemorySessionStore, get_session_store
from api.chat.service import ChatService
from api.dependencies import (
    get_retrieval_service,
    get_context_builder,
    get_rag_generator,
)
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.context_builder import EvidenceContextBuilder
from src.generation.rag_generator import RAGGenerator

router = APIRouter(prefix="/api/chat", tags=["Conversational Chat"])


def get_chat_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    context_builder: EvidenceContextBuilder = Depends(get_context_builder),
    rag_generator: RAGGenerator = Depends(get_rag_generator),
    session_store: InMemorySessionStore = Depends(get_session_store),
) -> ChatService:
    """Dependency provider for ChatService."""
    return ChatService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        rag_generator=rag_generator,
        session_store=session_store,
    )


@router.post(
    "",
    response_model=ChatResponse,
    summary="Multi-Turn Conversational Chat",
    description="""
Processes a conversational chat turn with bounded history and deterministic query rewriting.
- If `session_id` is omitted, creates a new session.
- If `session_id` is supplied, continues the existing conversation.
- If `session_id` is unknown, returns HTTP 404.
- Previous assistant replies are used for intent understanding only, never as factual evidence.
    """,
)
async def chat_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Handles a conversational query within a session."""
    return await chat_service.process_chat(request)


@router.post(
    "/stream",
    summary="Progressive Streaming Conversational Chat",
    description="""
Progressively streams grounded answer tokens using Server-Sent Events (SSE).
Retrieval and evidence context bounding are completed before streaming begins.
Returns `text/event-stream` with deterministic `metadata`, `token`, and `done` events.
On successful stream completion, the full assistant response is stored in the session history.
    """,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Progressive Server-Sent Events stream.",
        }
    },
)
async def chat_stream_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Streams conversational answer tokens via Server-Sent Events."""
    event_generator = chat_service.process_chat_stream(request)
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Delete Chat Session",
    description="""
Deletes an active conversation session and its in-memory history.
Returns HTTP 404 if the session ID does not exist.
    """,
)
async def delete_session_endpoint(
    session_id: str = Path(..., description="The ID of the session to delete"),
    session_store: InMemorySessionStore = Depends(get_session_store),
) -> DeleteSessionResponse:
    """Deletes an in-memory session."""
    clean_sid = session_id.strip()
    deleted = session_store.delete_session(clean_sid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{clean_sid}' not found.",
        )
    return DeleteSessionResponse(status="deleted", session_id=clean_sid)
