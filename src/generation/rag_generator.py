"""
src/generation/rag_generator.py
===============================
Grounded RAG Answer Generator for SugamGov AI.

Architecture & Role in SugamGov:
--------------------------------
The RAGGenerator forms the final answer-synthesis layer:

User Query + EvidenceContext
            ↓
       RAGGenerator
(Strict System Prompt + Grounding)
            ↓
        RAGAnswer
(Factual Structured Output + Citations)

Key Capabilities:
  1. Strict Grounding:
     - Constrains generation strictly to retrieved evidence context.
     - Never hallucinates facts, amounts, or URLs.
     - Transparently reports missing information if evidence is insufficient.
  2. Multi-Lingual Support:
     - Supports English ('en'), Hindi ('hi'), and Gujarati ('gu').
     - Automatically detects query language when language='auto'.
     - Keeps official scheme names intact as provided in evidence.
  3. No-Evidence Short-Circuit:
     - Returns deterministic ungrounded responses when EvidenceContext has 0 schemes.
     - Strictly avoids making unnecessary Gemini API calls when no evidence exists.
  4. Prompt-Injection Resistance:
     - Formats retrieved evidence as untrusted passive reference data.
     - Disregards any instructions or prompt manipulations found inside scheme texts.
  5. Citations & Categorical Confidence:
     - Associates factual statements with explicit scheme and chunk identifiers.
     - Computes categorical confidence ('high', 'medium', 'low') based on evidence depth.
     - Explicitly documents source limitations (sources table currently 0).
  6. Robust Error Handling:
     - Intercepts API rate limits (429) and outages (500/503) cleanly.
     - Never leaks API keys, passwords, or connection strings in logs or exceptions.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from src.retrieval.context_builder import EvidenceContext
from src.generation.models import RAGAnswer, EvidenceCitation
from src.generation.prompt import (
    SYSTEM_INSTRUCTION,
    detect_language,
    build_grounded_prompt,
)

logger = logging.getLogger("sugamgov_generator")

# Matches patterns like:
# [S2080 / S2080_scheme_name_0]
# [S2080 / S2080_benefits_0]
# [S2080_eligibility_0]
# [S2080]
# [LMS_xyz]
CITATION_REGEX = re.compile(
    r"\[\s*(?:S\d+|LMS_)[A-Za-z0-9_\-]*(?:\s*(?:/|-)\s*[A-Za-z0-9_\-]+)*\s*\]",
    re.IGNORECASE
)

def strip_technical_citations(text: str) -> str:
    """Removes ugly internal database/chunk bracketed codes like [S2080_eligibility_0] from user text."""
    if not text:
        return ""
    # Strip citation tags
    cleaned = CITATION_REGEX.sub("", text)
    # Clean up double spaces left by removal before punctuation or words
    cleaned = re.sub(r" +([.,!?:;])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"

# Standard deterministic no-evidence responses across languages
NO_EVIDENCE_MESSAGES = {
    "en": "I couldn't find enough relevant government-scheme information in the available knowledge base to answer this question.",
    "hi": "उपलब्ध सरकारी योजना ज्ञानकोष में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।",
    "gu": "ઉપલબ્ધ સરકારી યોજના જ્ઞાનકોશમાં આ પ્રશ્નનો જવાબ આપવા માટે પૂરતી માહિતી મળી નથી.",
}


def _determine_confidence(evidence_context: Optional[EvidenceContext]) -> tuple[str, str]:
    """
    Computes categorical confidence and limitation notices based strictly on
    evidence availability and depth (not arbitrary probabilities).
    """
    if not evidence_context or evidence_context.total_schemes == 0:
        return "low", "No relevant schemes retrieved in the evidence context."

    total_chunks = evidence_context.total_evidence_chunks
    if total_chunks >= 3 and evidence_context.total_schemes >= 1:
        return (
            "high",
            "Based on retrieved government-scheme records. Official live website verification has not yet been performed (sources=0).",
        )
    elif total_chunks >= 1:
        return (
            "medium",
            "Limited evidence retrieved. Specific eligibility or documentation details may be incomplete in the knowledge base.",
        )
    else:
        return "low", "Insufficient evidence retrieved."


class RAGGenerator:
    """
    Grounded LLM Generator for SugamGov AI RAG.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """
        Initializes the RAGGenerator with model configuration and GenAI client.

        Args:
            model_name: Optional Gemini model name. If None, reads from
                        GEMINI_GENERATION_MODEL env var, defaulting to gemini-3.5-flash-lite.
            client: Optional pre-configured Google GenAI client instance.
        """
        if ENV_PATH.exists():
            load_dotenv(ENV_PATH)

        self.model_name = (
            model_name
            or os.getenv("GEMINI_GENERATION_MODEL")
            or DEFAULT_GENERATION_MODEL
        )

        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required to initialize RAGGenerator.")
            from google import genai
            self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        query: str,
        evidence_context: Optional[EvidenceContext] = None,
        language: str = "auto",
    ) -> RAGAnswer:
        """
        Generates a grounded, structured answer from retrieved evidence.

        Args:
            query: User search text or question.
            evidence_context: EvidenceContext object containing retrieved schemes & chunks.
            language: Target language ('auto', 'en', 'hi', 'gu'). Defaults to 'auto'.

        Returns:
            RAGAnswer dataclass containing answer, language, confidence, citations, and disclaimers.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        clean_query = query.strip()

        # Resolve target language
        target_lang = detect_language(clean_query) if language == "auto" else language
        if target_lang not in ["en", "hi", "gu"]:
            target_lang = "en"

        # 1. NO-EVIDENCE SHORT-CIRCUIT:
        # If no query or evidence context is empty, return deterministic message without calling Gemini
        if not clean_query or not evidence_context or evidence_context.total_schemes == 0:
            no_ev_msg = NO_EVIDENCE_MESSAGES.get(target_lang, NO_EVIDENCE_MESSAGES["en"])
            return RAGAnswer(
                query=clean_query,
                answer=no_ev_msg,
                language=target_lang,
                grounded=False,
                confidence="low",
                schemes=[],
                evidence_used=[],
                limitations="No relevant government-scheme records found in the retrieved evidence.",
            )

        # 2. Extract Scheme Summaries & Evidence Citations from Context
        schemes_summary: List[Dict[str, Any]] = []
        evidence_citations: List[Dict[str, Any]] = []

        for s in evidence_context.schemes:
            schemes_summary.append({
                "scheme_id": s.scheme_id,
                "scheme_name": s.scheme_name,
                "level": s.level,
                "state": s.state,
                "states": s.states,
                "categories": s.categories,
            })
            for c in s.evidence_chunks:
                evidence_citations.append({
                    "scheme_id": s.scheme_id,
                    "scheme_name": s.scheme_name,
                    "chunk_id": c.chunk_id,
                    "field_name": c.field_name,
                })

        confidence, limitations = _determine_confidence(evidence_context)

        # 3. Construct Grounded Prompt
        prompt_content = build_grounded_prompt(
            query=clean_query,
            evidence_context=evidence_context,
            language=target_lang,
        )

        # Log the prompt being sent to Gemini
        prompt_preview = str(prompt_content)[:800] if prompt_content else ''
        logger.info(
            "\n" + "-"*70 +
            "\n[GEMINI PROMPT] model=%s  language=%s" +
            "\n[GEMINI PROMPT] Evidence schemes count: %d" +
            "\n[GEMINI PROMPT] prompt preview (first 800 chars):\n%s" +
            "\n" + "-"*70,
            self.model_name,
            target_lang,
            evidence_context.total_schemes if evidence_context else 0,
            prompt_preview,
        )

        # 4. Invoke Gemini Generation API
        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,  # Zero temperature for strictly factual, deterministic grounding
                ),
            )

            raw_answer = response.text.strip() if response and response.text else ""
            cleaned_answer = strip_technical_citations(raw_answer)

            # Log Gemini raw response
            logger.info(
                "\n" + "-"*70 +
                "\n[GEMINI RAW RESPONSE] length=%d chars" +
                "\n[GEMINI RAW RESPONSE] text (first 800 chars):\n%s" +
                "\n" + "-"*70,
                len(raw_answer),
                raw_answer[:800],
            )

            # Check if model reported insufficient information
            insufficient_markers = [
                "does not contain enough evidence",
                "insufficient information",
                "पर्याप्त जानकारी नहीं मिली",
                "પૂરતી માહિતી મળી નથી",
            ]
            is_grounded = not any(marker in raw_answer.lower() for marker in insufficient_markers)

            return RAGAnswer(
                query=clean_query,
                answer=cleaned_answer,
                language=target_lang,
                grounded=is_grounded,
                confidence=confidence if is_grounded else "low",
                schemes=schemes_summary,
                evidence_used=evidence_citations,
                limitations=limitations,
            )

        except Exception as e:
            err_str = str(e).lower()
            # Cleanly handle quota exhaustion (429) or service outages (500/503)
            if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                raise RuntimeError("Gemini generation API quota limit reached. Please try again later.") from None
            if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                raise RuntimeError("Gemini generation model is temporarily unavailable due to high demand.") from None

            # Re-raise generic error without exposing any internal credentials or URLs
            raise RuntimeError(f"Error during grounded answer generation: {type(e).__name__}") from None

    def generate_stream(
        self,
        query: str,
        evidence_context: Optional[EvidenceContext] = None,
        language: str = "auto",
    ):
        """
        Streams grounded answer tokens from retrieved evidence.
        Yields text chunks progressively as they arrive from the Gemini generation model.

        If no evidence exists in evidence_context, yields the deterministic
        no-evidence message without making any Gemini API calls.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        clean_query = query.strip()
        target_lang = detect_language(clean_query) if language == "auto" else language
        if target_lang not in ["en", "hi", "gu"]:
            target_lang = "en"

        # 1. No-evidence short-circuit: zero API calls
        if not clean_query or not evidence_context or evidence_context.total_schemes == 0:
            no_ev_msg = NO_EVIDENCE_MESSAGES.get(target_lang, NO_EVIDENCE_MESSAGES["en"])
            yield no_ev_msg
            return

        # 2. Construct Grounded Prompt
        prompt_content = build_grounded_prompt(
            query=clean_query,
            evidence_context=evidence_context,
            language=target_lang,
        )

        # Log the prompt being sent to Gemini (streaming)
        prompt_preview = str(prompt_content)[:800] if prompt_content else ''
        logger.info(
            "\n" + "-"*70 +
            "\n[GEMINI PROMPT STREAM] model=%s  language=%s" +
            "\n[GEMINI PROMPT STREAM] Evidence schemes: %d" +
            "\n[GEMINI PROMPT STREAM] prompt preview (first 800 chars):\n%s" +
            "\n" + "-"*70,
            self.model_name,
            target_lang,
            evidence_context.total_schemes if evidence_context else 0,
            prompt_preview,
        )

        # 3. Stream from Gemini Generation API
        try:
            from google.genai import types

            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )
            
            # Buffer small amount of text across tokens to prevent leaking partial [S... tags
            stream_buf = ""
            for chunk in response_stream:
                if chunk and chunk.text:
                    stream_buf += chunk.text

                    # If there's an open '[' without a matching ']' in the buffer, hold onto it
                    open_bracket_idx = stream_buf.rfind("[")
                    close_bracket_idx = stream_buf.rfind("]")

                    if open_bracket_idx != -1 and (close_bracket_idx == -1 or open_bracket_idx > close_bracket_idx):
                        # Potential unclosed tag: yield safe text before '['
                        safe_text = stream_buf[:open_bracket_idx]
                        stream_buf = stream_buf[open_bracket_idx:]
                    else:
                        safe_text = stream_buf
                        stream_buf = ""

                    cleaned_text = strip_technical_citations(safe_text)
                    if cleaned_text:
                        yield cleaned_text

            if stream_buf:
                cleaned_final = strip_technical_citations(stream_buf)
                if cleaned_final:
                    yield cleaned_final

        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                raise RuntimeError("Gemini generation API quota limit reached. Please try again later.") from None
            if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                raise RuntimeError("Gemini generation model is temporarily unavailable due to high demand.") from None
            raise RuntimeError(f"Error during grounded answer streaming: {type(e).__name__}") from None
