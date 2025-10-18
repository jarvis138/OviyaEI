from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, conlist
from typing import Dict


router = APIRouter(prefix="/v1", tags=["conditioning"])

_session_vectors: Dict[str, list] = {}


class VectorPatch(BaseModel):
    vector: conlist(float, min_items=5, max_items=5)


@router.get("/personality/{conversation_id}")
def get_vector(conversation_id: str) -> Dict[str, object]:
    """
    Retrieve the stored normalized vector for a conversation.
    
    Parameters:
    	conversation_id (str): Conversation identifier whose vector is requested.
    
    Returns:
    	result (Dict[str, object]): A dictionary with keys:
    		- "conversation_id": the provided conversation_id
    		- "vector": the stored normalized vector (list of 5 floats) or None if not found
    """
    v = _session_vectors.get(conversation_id)
    return {"conversation_id": conversation_id, "vector": v}


@router.patch("/personality/{conversation_id}")
def set_vector(conversation_id: str, body: VectorPatch) -> Dict[str, object]:
    """
    Store a normalized 5-element personality vector for a conversation.
    
    Validates that the provided vector's sum is greater than zero, clamps each element to at least 1e-6, normalizes the vector so its elements sum to 1, stores it in the module-level in-memory store keyed by `conversation_id`, and returns the stored vector.
    
    Parameters:
        conversation_id (str): Identifier for the conversation to associate with the vector.
        body (VectorPatch): Payload containing `vector`, a list of exactly five floats.
    
    Returns:
        dict: A mapping with keys `"conversation_id"` (the provided id) and `"vector"` (the normalized stored vector).
    
    Raises:
        HTTPException: If the sum of the input vector is less than or equal to zero (status code 400, detail "sum must be > 0").
    """
    v = body.vector
    s = sum(v)
    if s <= 0:
        raise HTTPException(status_code=400, detail="sum must be > 0")
    v = [max(1e-6, x) for x in v]
    s = sum(v)
    v = [x / s for x in v]
    _session_vectors[conversation_id] = v
    return {"conversation_id": conversation_id, "vector": v}

