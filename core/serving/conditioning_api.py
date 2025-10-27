from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List


router = APIRouter(prefix="/v1", tags=["conditioning"])

_session_vectors: Dict[str, list] = {}


class VectorPatch(BaseModel):
    vector: List[float] = Field(min_items=5, max_items=5)


@router.get("/personality/{conversation_id}")
def get_vector(conversation_id: str) -> Dict[str, object]:
    v = _session_vectors.get(conversation_id)
    return {"conversation_id": conversation_id, "vector": v}


@router.patch("/personality/{conversation_id}")
def set_vector(conversation_id: str, body: VectorPatch) -> Dict[str, object]:
    v = body.vector
    s = sum(v)
    if s <= 0:
        raise HTTPException(status_code=400, detail="sum must be > 0")
    v = [max(1e-6, x) for x in v]
    s = sum(v)
    v = [x / s for x in v]
    _session_vectors[conversation_id] = v
    return {"conversation_id": conversation_id, "vector": v}




