"""
api/chat/service.py
===================
Conversational Orchestration Service for SugamGov AI RAG.

Key Capabilities:
  1. Deterministic Query Context Builder:
     - Formulates search queries for follow-up turns by merging recent topic keywords
       without calling Gemini (protecting quota and ensuring testability).
  2. Strict Grounding Integrity:
     - Assistant messages are NEVER treated as factual evidence. Only retrieved
       database records are authoritative.
  3. Session Lifecycle Management:
     - Bounded in-memory conversation history.
     - Clean session isolation.
"""

import re
import json
import logging
from typing import List, Optional, Dict, Any, Set, AsyncGenerator
from fastapi import HTTPException, status

from api.chat.models import ChatRequest, ChatResponse, ChatMessage, SessionData
from api.chat.session_store import InMemorySessionStore
from api.schemas import EvidenceCitationSchema
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.context_builder import EvidenceContextBuilder
from src.generation.rag_generator import (
    RAGGenerator,
    NO_EVIDENCE_MESSAGES,
    _determine_confidence,
)
from src.generation.prompt import detect_language

logger = logging.getLogger("sugamgov_chat")


def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """Formats payload as standard Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# Common conversational stopwords across English, Hindi, and Gujarati
CONVERSATIONAL_STOPWORDS: Set[str] = {
    # English
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "tell", "me",
    "what", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "doing", "would", "could", "i", "you", "he", "she", "it",
    "we", "they", "them", "their", "this", "that", "these", "those", "am", "give",
    "information", "details", "please", "sir", "madam",
    # Hindi (romanized / standard terms)
    "kya", "kaise", "hai", "hain", "ke", "ki", "ko", "ka", "me", "mein", "se", "aur",
    # Gujarati (romanized / standard terms)
    "su", "chhe", "ma", "mate", "ne", "thi", "ane",
}

# Follow-up trigger patterns indicating context-dependent queries
FOLLOWUP_PATTERNS = [
    re.compile(r"^(what|which|how|who|can|is|are|tell me|give me|list)\b", re.IGNORECASE),
    re.compile(r"\b(documents|eligibility|criteria|apply|process|benefits|amount|fees|deadline|procedure)\b", re.IGNORECASE),
    re.compile(r"\b(दस्तावेज|पात्रता|आवेदन|लाभ|दस्तावेज़|कागजात)\b"),
    re.compile(r"\b(દસ્તાવેજ|પાત્રતા|અરજી|લાભ)\b"),
]


# Aspect keywords that describe scheme sections rather than domain topics
ASPECT_KEYWORDS: Set[str] = {
    "document", "documents", "required", "need", "needed", "eligibility",
    "eligible", "criteria", "apply", "applying", "application", "process",
    "procedure", "steps", "benefits", "benefit", "deadline", "date", "dates",
    "fee", "fees", "cost", "amount", "money", "portal", "office", "where",
    "who", "which", "how", "what", "दस्तावेज", "पात्रता", "आवेदन", "लाभ",
    "દસ્તાવેજ", "પાત્રતા", "અરજી", "લાભ",
}


def extract_topic_keywords(text: str) -> List[str]:
    """
    Extracts informative keywords from a user message, filtering out conversational noise.
    Supports English, Hindi, and Gujarati scripts by using unicode-aware token splitting.
    """
    tokens = [w.strip(".,?!;:\"'()[]{}«»—–-") for w in text.lower().split()]
    keywords = []
    for w in tokens:
        if len(w) >= 2 and w not in CONVERSATIONAL_STOPWORDS:
            keywords.append(w)
    return keywords


def get_conversation_topic_keywords(prev_user_msgs: List[str]) -> List[str]:
    """
    Scans previous user messages to extract core domain topic keywords,
    prioritizing non-aspect domain nouns (e.g. 'student', 'scholarship', 'farmer').
    """
    topic_kws: List[str] = []
    for msg in prev_user_msgs:
        kws = extract_topic_keywords(msg)
        non_aspect = [k for k in kws if k not in ASPECT_KEYWORDS]
        for k in non_aspect:
            if k not in topic_kws:
                topic_kws.append(k)

    # Fallback to all keywords if only aspect terms were found
    if not topic_kws:
        for msg in reversed(prev_user_msgs):
            for k in extract_topic_keywords(msg):
                if k not in topic_kws:
                    topic_kws.append(k)
            if topic_kws:
                break

    return topic_kws


def build_conversational_query(
    current_message: str,
    previous_messages: List[ChatMessage],
) -> str:
    """
    Deterministically constructs an effective retrieval query for follow-up turns.

    Rules:
      1. If no previous user messages exist, return current_message directly.
      2. If previous user messages exist, identify if the current message is a follow-up.
      3. If follow-up, extract topic keywords from previous user turns and combine
         them with the current query, avoiding duplication.
      4. Never use assistant-generated text as query expansion source.
    """
    clean_current = current_message.strip()
    if not previous_messages:
        return clean_current

    # Find previous user messages
    prev_user_msgs = [m.content for m in previous_messages if m.role == "user"]
    if not prev_user_msgs:
        return clean_current

    # Check if current message looks like a follow-up or lacks topic anchor
    is_followup = any(pattern.search(clean_current) for pattern in FOLLOWUP_PATTERNS)
    current_keywords = extract_topic_keywords(clean_current)

    # If current message is brief (e.g. "what documents?") or matched follow-up patterns
    if is_followup or len(current_keywords) <= 3:
        prev_keywords = get_conversation_topic_keywords(prev_user_msgs)
        # Combine unique keywords preserving order
        combined_keywords: List[str] = []
        for kw in prev_keywords:
            if kw not in combined_keywords:
                combined_keywords.append(kw)
        for kw in current_keywords:
            if kw not in combined_keywords:
                combined_keywords.append(kw)

        if combined_keywords:
            reformulated = " ".join(combined_keywords)
            logger.info("Conversational query reformulated: '%s' -> '%s'", clean_current, reformulated)
            return reformulated

    return clean_current


class ChatService:
    """
    Orchestration service for multi-turn conversational RAG interactions.
    """
    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: EvidenceContextBuilder,
        rag_generator: RAGGenerator,
        session_store: InMemorySessionStore,
    ):
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.rag_generator = rag_generator
        self.session_store = session_store

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Processes a conversational chat turn:
          1. Validates or creates the session.
          2. Retrieves recent conversation history.
          3. Reformulates retrieval query deterministically for follow-ups.
          4. Executes hybrid retrieval via RetrievalService.
          5. Builds bounded evidence context via EvidenceContextBuilder.
          6. Generates strictly grounded answer via RAGGenerator.
          7. Records the turn in the session store.
          8. Returns structured ChatResponse.
        """
        # 1. Resolve Session
        if request.session_id:
            session = self.session_store.get_session(request.session_id)
            if not session:
                # Automatically restore/create session with requested ID instead of failing
                session = self.session_store.create_session(session_id=request.session_id)
        else:
            session = self.session_store.create_session()

        # Seed session from request.history if session messages are empty
        if not session.messages and request.history:
            import time
            for item in request.history:
                if item.content and item.content.strip():
                    session.messages.append(
                        ChatMessage(role=item.role, content=item.content.strip(), timestamp=time.time())
                    )
            if len(session.messages) > self.session_store.max_history_messages:
                session.messages = session.messages[-self.session_store.max_history_messages:]

        # 2. Formulate Retrieval Query
        retrieval_query = build_conversational_query(
            current_message=request.message,
            previous_messages=session.messages,
        )
        logger.info(
            "\n" + "="*70 +
            "\n[PIPELINE] USER QUERY   : %s" +
            "\n[PIPELINE] RETRIEVAL QUERY (sent to DB): %s" +
            "\n" + "="*70,
            request.message, retrieval_query
        )

        # 3. Retrieve Evidence from Database
        try:
            retrieval_response = self.retrieval_service.retrieve(
                query=retrieval_query,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                state=request.state,
                level=request.level,
                category=request.category,
            )
        except Exception as e:
            logger.error("Retrieval failed during chat turn: %s", type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal retrieval error occurred during chat turn.",
            )

        # 4. Build Evidence Context
        evidence_context = self.context_builder.build(retrieval_response=retrieval_response)

        # ── RAG Retrieval Summary Log ─────────────────────────────────────────
        scheme_names_found = [s.scheme_name for s in evidence_context.schemes]
        chunk_count = sum(len(s.evidence_chunks) for s in evidence_context.schemes)
        logger.info(
            "\n" + "-"*70 +
            "\n[RAG RETRIEVAL] Schemes found in DB  : %d  →  %s" +
            "\n[RAG RETRIEVAL] Evidence chunks total : %d" +
            "\n[RAG RETRIEVAL] Retrieval method      : %s" +
            "\n[RAG RETRIEVAL] Is grounded?          : %s" +
            "\n" + "-"*70,
            retrieval_response.total_results,
            scheme_names_found,
            chunk_count,
            retrieval_response.retrieval_method,
            str(evidence_context.total_schemes > 0),
        )
        if evidence_context.total_schemes == 0:
            logger.warning(
                "[RAG RETRIEVAL] ⚠️  NO evidence found in DB for query: '%s' — "
                "Gemini will NOT be called; deterministic no-evidence reply will be returned.",
                retrieval_query,
            )
        else:
            for s in evidence_context.schemes:
                for c in s.evidence_chunks:
                    logger.debug(
                        "[RAG CHUNK] scheme='%s'  field='%s'  score=%.4f  text_preview='%s'",
                        s.scheme_name, c.field_name,
                        getattr(c, 'score', 0.0),
                        (c.chunk_text[:120].replace('\n', ' ') + '...') if c.chunk_text else '',
                    )

        # 5. Generate Grounded Answer
        # Uses the user's actual question as the prompt target, with evidence retrieved via conversational query
        try:
            rag_answer = self.rag_generator.generate(
                query=request.message,
                evidence_context=evidence_context,
                language=request.language or "auto",
            )
        except Exception as e:
            err_str = str(e)
            logger.error("Generation failed during chat turn: %s", type(e).__name__)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Gemini API rate limit or quota exceeded. Please retry after a brief pause.",
                )
            if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Gemini generation service is temporarily unavailable due to high demand.",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Grounded answer generation failed to complete.",
            )

        # ── Final Answer Log ──────────────────────────────────────────────────
        logger.info(
            "\n" + "="*70 +
            "\n[GEMINI RESPONSE] grounded=%s  confidence=%s  language=%s" +
            "\n[GEMINI RESPONSE] raw answer (first 500 chars):\n%s" +
            "\n" + "="*70,
            rag_answer.grounded, rag_answer.confidence, rag_answer.language,
            rag_answer.answer[:500],
        )
        logger.info(
            "\n" + "*"*70 +
            "\n[FINAL → USER] answer length=%d chars  grounded=%s  confidence=%s" +
            "\n[FINAL → USER] schemes cited: %s" +
            "\n" + "*"*70,
            len(rag_answer.answer),
            rag_answer.grounded,
            rag_answer.confidence,
            [s.get('scheme_name', '') if isinstance(s, dict) else getattr(s, 'scheme_name', str(s)) for s in rag_answer.schemes],
        )

        # 6. Record Conversation Turn
        self.session_store.add_turn(
            session_id=session.session_id,
            user_message=request.message,
            assistant_message=rag_answer.answer,
        )

        # 7. Assemble Structured Citations
        evidence_used_schema: List[EvidenceCitationSchema] = []
        for c in rag_answer.evidence_used:
            if isinstance(c, dict):
                evidence_used_schema.append(
                    EvidenceCitationSchema(
                        scheme_id=str(c.get("scheme_id", "")),
                        scheme_name=str(c.get("scheme_name", "")),
                        chunk_id=str(c.get("chunk_id", "")),
                        field_name=str(c.get("field_name", "")),
                    )
                )
            else:
                evidence_used_schema.append(
                    EvidenceCitationSchema(
                        scheme_id=getattr(c, "scheme_id", ""),
                        scheme_name=getattr(c, "scheme_name", ""),
                        chunk_id=getattr(c, "chunk_id", ""),
                        field_name=getattr(c, "field_name", ""),
                    )
                )

        schemes_list = [
            s.get("scheme_id", str(s)) if isinstance(s, dict) else getattr(s, "scheme_id", str(s))
            for s in rag_answer.schemes
        ]

        return ChatResponse(
            session_id=session.session_id,
            message=request.message,
            answer=rag_answer.answer,
            language=rag_answer.language,
            grounded=rag_answer.grounded,
            confidence=rag_answer.confidence,
            schemes=schemes_list,
            evidence_used=evidence_used_schema,
            limitations=rag_answer.limitations,
            retrieval_method=retrieval_response.retrieval_method,
            total_results=retrieval_response.total_results,
            coverage_info=retrieval_response.coverage_info,
        )

    async def process_chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        Processes a conversational chat turn with progressive SSE streaming:
          1. Resolves or validates the session.
          2. Formulates retrieval query deterministically from history.
          3. Retrieves evidence and builds bounded context.
          4. Emits `event: metadata` containing session, coverage, schemes, and citations.
          5. Progressively yields `event: token` containing generated answer chunks.
          6. Records completed response in session store (only if successful).
          7. Emits `event: done` with total character counts.
          8. Handles errors safely by emitting `event: error` without credentials.
        """
        # 1. Resolve Session
        if request.session_id:
            session = self.session_store.get_session(request.session_id)
            if not session:
                # Automatically restore/create session with requested ID instead of failing
                session = self.session_store.create_session(session_id=request.session_id)
        else:
            session = self.session_store.create_session()

        # Seed session from request.history if session messages are empty
        if not session.messages and request.history:
            import time
            for item in request.history:
                if item.content and item.content.strip():
                    session.messages.append(
                        ChatMessage(role=item.role, content=item.content.strip(), timestamp=time.time())
                    )
            if len(session.messages) > self.session_store.max_history_messages:
                session.messages = session.messages[-self.session_store.max_history_messages:]

        # 2. Formulate Retrieval Query
        retrieval_query = build_conversational_query(
            current_message=request.message,
            previous_messages=session.messages,
        )
        logger.info(
            "\n" + "="*70 +
            "\n[PIPELINE STREAM] USER QUERY          : %s" +
            "\n[PIPELINE STREAM] RETRIEVAL QUERY (DB): %s" +
            "\n" + "="*70,
            request.message, retrieval_query
        )

        # 3. Retrieve Evidence from Database
        try:
            retrieval_response = self.retrieval_service.retrieve(
                query=retrieval_query,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                state=request.state,
                level=request.level,
                category=request.category,
            )
        except Exception as e:
            logger.error("Retrieval failed during streaming chat turn: %s", type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal retrieval error occurred during chat turn.",
            )

        # 4. Build Evidence Context
        evidence_context = self.context_builder.build(retrieval_response=retrieval_response)

        # ── RAG Retrieval Summary Log (streaming path) ────────────────────────
        stream_scheme_names = [s.scheme_name for s in evidence_context.schemes]
        stream_chunk_count = sum(len(s.evidence_chunks) for s in evidence_context.schemes)
        logger.info(
            "\n" + "-"*70 +
            "\n[RAG RETRIEVAL STREAM] Schemes found in DB  : %d  →  %s" +
            "\n[RAG RETRIEVAL STREAM] Evidence chunks total : %d" +
            "\n[RAG RETRIEVAL STREAM] Retrieval method      : %s" +
            "\n" + "-"*70,
            retrieval_response.total_results,
            stream_scheme_names,
            stream_chunk_count,
            retrieval_response.retrieval_method,
        )
        if evidence_context.total_schemes == 0:
            logger.warning(
                "[RAG RETRIEVAL STREAM] ⚠️  NO evidence found in DB for query: '%s' — "
                "Gemini will NOT be called; deterministic no-evidence reply will be returned.",
                retrieval_query,
            )
        else:
            for s in evidence_context.schemes:
                for c in s.evidence_chunks:
                    logger.debug(
                        "[RAG CHUNK STREAM] scheme='%s'  field='%s'  score=%.4f  text_preview='%s'",
                        s.scheme_name, c.field_name,
                        getattr(c, 'score', 0.0),
                        (c.chunk_text[:120].replace('\n', ' ') + '...') if c.chunk_text else '',
                    )

        # 5. Extract Metadata & Citations
        is_empty = (evidence_context.total_schemes == 0)
        confidence, limitations = _determine_confidence(evidence_context)
        clean_msg = request.message.strip()
        target_lang = detect_language(clean_msg) if (request.language == "auto" or not request.language) else request.language
        if target_lang not in ["en", "hi", "gu"]:
            target_lang = "en"

        schemes_list = [s.scheme_id for s in evidence_context.schemes]
        evidence_used_list = [
            {
                "scheme_id": s.scheme_id,
                "scheme_name": s.scheme_name,
                "chunk_id": c.chunk_id,
                "field_name": c.field_name,
            }
            for s in evidence_context.schemes
            for c in s.evidence_chunks
        ]

        metadata_payload = {
            "session_id": session.session_id,
            "language": target_lang,
            "grounded": not is_empty,
            "confidence": confidence if not is_empty else "low",
            "retrieval_method": retrieval_response.retrieval_method,
            "total_results": retrieval_response.total_results,
            "coverage_info": retrieval_response.coverage_info,
            "schemes": schemes_list,
            "evidence_used": evidence_used_list,
            "limitations": limitations,
        }

        # Emit initial metadata event
        yield format_sse("metadata", metadata_payload)

        # 6. Handle Empty-Evidence Case Deterministically (Zero Gemini Calls)
        if is_empty:
            no_ev_msg = NO_EVIDENCE_MESSAGES.get(target_lang, NO_EVIDENCE_MESSAGES["en"])
            yield format_sse("token", {"text": no_ev_msg})
            self.session_store.add_turn(
                session_id=session.session_id,
                user_message=request.message,
                assistant_message=no_ev_msg,
            )
            yield format_sse("done", {
                "session_id": session.session_id,
                "grounded": False,
                "confidence": "low",
                "total_chars": len(no_ev_msg),
            })
            return

        # 7. Progressive Streaming via RAGGenerator
        streamed_tokens: List[str] = []
        logger.info(
            "[GEMINI CALL STREAM] Sending evidence context to Gemini — "
            "%d schemes, %d chunks, language='%s'",
            evidence_context.total_schemes,
            sum(len(s.evidence_chunks) for s in evidence_context.schemes),
            target_lang,
        )
        try:
            for chunk_text in self.rag_generator.generate_stream(
                query=request.message,
                evidence_context=evidence_context,
                language=request.language or "auto",
            ):
                streamed_tokens.append(chunk_text)
                yield format_sse("token", {"text": chunk_text})

            full_answer = "".join(streamed_tokens).strip()

            # Verify grounding from complete answer text
            insufficient_markers = [
                "does not contain enough evidence",
                "insufficient information",
                "पर्याप्त जानकारी नहीं मिली",
                "પૂરતી માહિતી મળી નથી",
            ]
            is_grounded = not any(m in full_answer.lower() for m in insufficient_markers)
            final_confidence = confidence if is_grounded else "low"

            logger.info(
                "\n" + "="*70 +
                "\n[GEMINI RESPONSE STREAM] grounded=%s  confidence=%s  total_chars=%d" +
                "\n[GEMINI RESPONSE STREAM] raw answer (first 500 chars):\n%s" +
                "\n" + "="*70,
                is_grounded, final_confidence, len(full_answer),
                full_answer[:500],
            )
            logger.info(
                "\n" + "*"*70 +
                "\n[FINAL → USER STREAM] answer length=%d chars  grounded=%s  confidence=%s" +
                "\n[FINAL → USER STREAM] schemes cited: %s" +
                "\n" + "*"*70,
                len(full_answer), is_grounded, final_confidence,
                stream_scheme_names,
            )

            # Store turn only on complete success
            self.session_store.add_turn(
                session_id=session.session_id,
                user_message=request.message,
                assistant_message=full_answer,
            )

            yield format_sse("done", {
                "session_id": session.session_id,
                "grounded": is_grounded,
                "confidence": final_confidence,
                "total_chars": len(full_answer),
            })

        except Exception as e:
            err_str = str(e).lower()
            logger.error("Error during streaming generation: %s", type(e).__name__)
            if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                err_msg = "Gemini API rate limit or quota exceeded. Please retry after a brief pause."
            elif "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                err_msg = "Gemini generation service is temporarily unavailable due to high demand."
            else:
                err_msg = "Grounded answer streaming failed to complete."

            yield format_sse("error", {"detail": err_msg})
