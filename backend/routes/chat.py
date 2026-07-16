

import json
import re
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Path as FastPath
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Add parent directory to path so we can import from database and ai_provider
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from ..database import (
        create_conversation,
        get_conversations_by_session,
        delete_conversation,
        insert_chat_message,
        get_chat_history,
        get_db_connection,
        check_conversation_exists,
    )
    from ..ai_provider import get_ai_provider
    from ..constants import (
        MAX_MESSAGE_LENGTH,
        CHAT_HISTORY_WINDOW_HOURS,
        CHAT_RECENT_PREDICTIONS_LIMIT,
        CHAT_EXTENDED_PREDICTIONS_LIMIT,
    )
except ImportError:
    from database import (
        create_conversation,
        get_conversations_by_session,
        delete_conversation,
        insert_chat_message,
        get_chat_history,
        get_db_connection,
        check_conversation_exists,
    )
    from ai_provider import get_ai_provider
    from constants import (
        MAX_MESSAGE_LENGTH,
        CHAT_HISTORY_WINDOW_HOURS,
        CHAT_RECENT_PREDICTIONS_LIMIT,
        CHAT_EXTENDED_PREDICTIONS_LIMIT,
    )

logger = logging.getLogger("hykero.routes.chat")

# Basic prompt injection guard — patterns checked against every incoming message
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "forget previous",
    "you are now",
    "act as",
    "jailbreak",
    "system prompt",
    "override instructions",
]

router = APIRouter()


def _get_ai():
    """Lazily retrieves the AI provider."""
    return get_ai_provider()


# ── Pydantic Request Models ───────────────────────────────────────────────────
class ConversationCreate(BaseModel):
    session_id: str = Field(..., description="Unique guest session identifier")
    title: str = Field("New Conversation", description="Title of the conversation")


class MessageSend(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH, description="The user's query")
    stream: bool = Field(True, description="Enable Server-Sent Events (SSE) streaming")


# ── Database Context Enrichment ───────────────────────────────────────────────

def _parse_date_from_query(text: str):
    """
    Try to extract a date (and optional time) from a user query.
    Supports formats like:
      - 26-03-2026, 2026-03-26, 26/03/2026, March 26 2026
      - "at 11:00 pm", "at 23:00", "11 pm"
    Returns (date_str, time_str_or_None) or (None, None).
    """
    date_obj = None
    time_obj = None
    
    # Try DD-MM-YYYY or DD/MM/YYYY
    m = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            if d > 12:
                date_obj = datetime(y, mo, d)
            elif mo > 12:
                date_obj = datetime(y, d, mo)
            else:
                date_obj = datetime(y, mo, d)
        except ValueError:
            try:
                date_obj = datetime(y, d, mo)
            except ValueError:
                pass
    
    # Try YYYY-MM-DD
    if not date_obj:
        m = re.search(r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', text)
        if m:
            try:
                date_obj = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
    
    # Try textual dates (e.g. "March 26, 2026" or "March 26 2026" or "26 March 2026")
    if not date_obj:
        months = ["january", "february", "march", "april", "may", "june", "july",
                  "august", "september", "october", "november", "december",
                  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month_pattern = "|".join(months)
        
        # Format: Month DD YYYY
        m = re.search(r'(' + month_pattern + r')\s+(\d{1,2})\s*,?\s*(\d{4})', text, re.IGNORECASE)
        if m:
            m_str, d_str, y_str = m.group(1).lower(), m.group(2), m.group(3)
            # Find month index
            for idx, name in enumerate(months[:12]):
                if m_str.startswith(name):
                    try:
                        date_obj = datetime(int(y_str), idx + 1, int(d_str))
                        break
                    except ValueError:
                        pass
        
        # Format: DD Month YYYY
        if not date_obj:
            m = re.search(r'(\d{1,2})\s+(' + month_pattern + r')\s*,?\s*(\d{4})', text, re.IGNORECASE)
            if m:
                d_str, m_str, y_str = m.group(1), m.group(2).lower(), m.group(3)
                for idx, name in enumerate(months[:12]):
                    if m_str.startswith(name):
                        try:
                            date_obj = datetime(int(y_str), idx + 1, int(d_str))
                            break
                        except ValueError:
                            pass

    if not date_obj:
        return None, None

    # Parse time
    m = re.search(r'(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text, re.IGNORECASE)
    if m:
        hour_val = int(m.group(1))
        min_val = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        
        if hour_val <= 24:
            if ampm:
                ampm = ampm.lower()
                if ampm == "pm" and hour_val < 12:
                    hour_val += 12
                elif ampm == "am" and hour_val == 12:
                    hour_val = 0
            
            if hour_val < 24 and min_val < 60:
                time_obj = f"{hour_val:02d}:{min_val:02d}"
    
    date_str = date_obj.strftime("%Y-%m-%d")
    return date_str, time_obj


def _lookup_predictions(date_str: str, time_str: str = None):
    """
    Query the predictions database for records matching the given date (and optional time).
    Returns a list of matching prediction dicts.
    """
    try:
        try:
            from ..database import _format_query, _is_postgres
        except ImportError:
            from database import _format_query, _is_postgres

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            is_pg = _is_postgres()

            if time_str:
                target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                start_dt = target_dt - timedelta(hours=CHAT_HISTORY_WINDOW_HOURS)
                end_dt = target_dt + timedelta(hours=CHAT_HISTORY_WINDOW_HOURS)

                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                target_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")

                if is_pg:
                    cursor.execute(_format_query("""
                        SELECT sample_ts, shift, actual, predicted, residual,
                               confidence_lower, confidence_upper
                        FROM predictions
                        WHERE sample_ts >= ? AND sample_ts <= ?
                        ORDER BY ABS(EXTRACT(EPOCH FROM (sample_ts::timestamp - ?::timestamp)))
                        LIMIT ?
                    """), (start_str, end_str, target_str, CHAT_RECENT_PREDICTIONS_LIMIT))
                else:
                    cursor.execute("""
                        SELECT sample_ts, shift, actual, predicted, residual,
                               confidence_lower, confidence_upper
                        FROM predictions
                        WHERE datetime(sample_ts) >= datetime(?)
                          AND datetime(sample_ts) <= datetime(?)
                        ORDER BY ABS(julianday(sample_ts) - julianday(?))
                        LIMIT ?
                    """, (start_str, end_str, target_str, CHAT_RECENT_PREDICTIONS_LIMIT))
            else:
                if is_pg:
                    cursor.execute(_format_query("""
                        SELECT sample_ts, shift, actual, predicted, residual,
                               confidence_lower, confidence_upper
                        FROM predictions
                        WHERE sample_ts >= ? AND sample_ts < ?
                        ORDER BY sample_ts ASC
                        LIMIT ?
                    """), (f"{date_str} 00:00:00", f"{date_str} 23:59:59", CHAT_EXTENDED_PREDICTIONS_LIMIT))
                else:
                    cursor.execute("""
                        SELECT sample_ts, shift, actual, predicted, residual,
                               confidence_lower, confidence_upper
                        FROM predictions
                        WHERE date(sample_ts) = date(?)
                        ORDER BY datetime(sample_ts) ASC
                        LIMIT ?
                    """, (date_str, CHAT_EXTENDED_PREDICTIONS_LIMIT))

            rows = cursor.fetchall()
            
            results = []
            for r in rows:
                results.append({
                    "sample_ts": r["sample_ts"],
                    "shift": r["shift"],
                    "actual": r["actual"],
                    "predicted": r["predicted"],
                    "residual": r["residual"],
                    "confidence_lower": r["confidence_lower"],
                    "confidence_upper": r["confidence_upper"],
                })
            return results
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error looking up predictions: {e}", exc_info=True)
        return []


def _enrich_prompt_with_db_context(prompt: str) -> str:
    """
    Detect if the user is asking about flash point data for a specific date/time.
    If so, query the database and inject the results into the prompt for the AI.
    """
    flash_keywords = ["flash", "flashpoint", "prediction", "predicted", "actual", 
                      "value", "result", "reading", "record", "tell", "show", "what"]
    prompt_lower = prompt.lower()
    
    # Only enrich if there's a date reference AND some flash-point-related keyword
    has_flash_ref = any(kw in prompt_lower for kw in flash_keywords)
    if not has_flash_ref:
        return prompt

    date_str, time_str = _parse_date_from_query(prompt)
    if not date_str:
        return prompt
    
    records = _lookup_predictions(date_str, time_str)
    
    if not records:
        time_desc = f" at {time_str}" if time_str else ""
        context = (
            f"\n\n[DATABASE LOOKUP: No prediction records found for {date_str}{time_desc}. "
            f"The available data may not cover this date. Please inform the user that no "
            f"predictions were recorded for this date/time in the database.]\n"
        )
    else:
        lines = []
        for r in records:
            actual_str = f"{r['actual']:.2f}°C" if r['actual'] is not None else "N/A"
            predicted_str = f"{r['predicted']:.2f}°C" if r['predicted'] is not None else "N/A"
            residual_str = f"{r['residual']:.2f}°C" if r['residual'] is not None else "N/A"
            ci_str = f"[{r['confidence_lower']:.1f} – {r['confidence_upper']:.1f}]°C" if r['confidence_lower'] is not None else "N/A"
            lines.append(
                f"  • {r['sample_ts']} | Shift: {r['shift']} | "
                f"Actual: {actual_str} | Predicted: {predicted_str} | "
                f"Residual: {residual_str} | 95% CI: {ci_str}"
            )
        
        time_desc = f" around {time_str}" if time_str else ""
        context = (
            f"\n\n[DATABASE LOOKUP RESULTS for {date_str}{time_desc} — "
            f"{len(records)} record(s) found:\n"
            + "\n".join(lines) +
            f"\n\nUse these actual database records to answer the user's question. "
            f"Present the data clearly with actual and predicted values.]\n"
        )
    
    return prompt + context


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/chat/conversations")
def list_conversations(session_id: str = Query(..., description="Unique guest session identifier")):
    """List all active chat conversations associated with a session ID."""
    if not session_id.strip():
        raise HTTPException(400, "session_id cannot be empty")
    try:
        conversations = get_conversations_by_session(session_id)
        return {"status": "success", "conversations": conversations}
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(500, "Failed to retrieve conversations.")


@router.post("/chat/conversations")
def create_new_conversation(body: ConversationCreate):
    """Create a new conversation tab for a session ID."""
    if not body.session_id.strip():
        raise HTTPException(400, "session_id cannot be empty")
    try:
        conv_id = create_conversation(body.session_id, body.title)
        return {"status": "success", "conversation_id": conv_id, "title": body.title}
    except Exception as e:
        logger.error(f"Error creating conversation: {e}", exc_info=True)
        raise HTTPException(500, "Failed to create conversation.")


@router.delete("/chat/conversations/{conversation_id}")
def delete_conversation_route(conversation_id: int = FastPath(..., description="ID of the conversation to delete")):
    """Delete a conversation and all its associated messages."""
    try:
        # Check if conversation exists
        if not check_conversation_exists(conversation_id):
            raise HTTPException(404, "Conversation not found.")
        delete_conversation(conversation_id)
        return {"status": "success", "message": "Conversation deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        raise HTTPException(500, "Failed to delete conversation.")


@router.get("/chat/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int = FastPath(..., description="ID of the conversation")):
    """Retrieve the full sequence of messages in a conversation."""
    try:
        messages = get_chat_history(conversation_id)
        if not messages:
            # Check if conversation exists in DB by trying listing it
            # (or we can just return empty since get_chat_history returns empty list for new conversations)
            pass
        return {"status": "success", "messages": messages}
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}", exc_info=True)
        raise HTTPException(500, "Failed to fetch chat history.")


@router.post("/chat/conversations/{conversation_id}/messages")
def post_message(
    body: MessageSend,
    conversation_id: int = FastPath(..., description="ID of the conversation")
):
    """
    Post a user message. Returns a Server-Sent Events (SSE) stream yielding response chunks,
    or a simple JSON object if stream=False. Sanitizes inputs to prevent prompt injection.
    Enriches prompt with database context when date-specific flash point queries are detected.
    """
    sanitized_prompt = body.message.strip()
    if not sanitized_prompt:
        raise HTTPException(400, "Message cannot be empty.")

    # Basic prompt injection guard
    if any(pat in sanitized_prompt.lower() for pat in _INJECTION_PATTERNS):
        raise HTTPException(400, "Message contains disallowed patterns.")

    try:
        history = get_chat_history(conversation_id)
    except Exception as e:
        logger.error(f"Error validating conversation {conversation_id}: {e}")
        raise HTTPException(404, "Conversation not found.")

    try:
        insert_chat_message(conversation_id, "user", sanitized_prompt)
        history.append({"role": "user", "content": sanitized_prompt})
    except Exception as e:
        logger.error(f"Error saving user message to database: {e}", exc_info=True)
        raise HTTPException(500, "Failed to save message.")

    enriched_prompt = _enrich_prompt_with_db_context(sanitized_prompt)
    ai_instance = _get_ai()

    if body.stream:
        async def event_generator():
            full_response = ""
            try:
                async for chunk in ai_instance.generate_response_stream(enriched_prompt, history):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                insert_chat_message(conversation_id, "assistant", full_response.strip())
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as ex:
                logger.error(f"Streaming generator error: {ex}", exc_info=True)
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            ai_response = ai_instance.generate_response(enriched_prompt, history).strip()
            insert_chat_message(conversation_id, "assistant", ai_response)
            return {
                "status": "success",
                "role": "assistant",
                "content": ai_response
            }
        except Exception as e:
            logger.error(f"Non-streaming generator error: {e}", exc_info=True)
            raise HTTPException(500, "An error occurred while generating the AI response. Please try again.")
