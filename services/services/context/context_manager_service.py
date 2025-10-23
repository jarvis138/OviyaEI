#!/usr/bin/env python3
"""
Context Manager Service
Manages conversation context using Redis and Pinecone
"""
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
import pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import hashlib
import uuid
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    turn_id: str
    session_id: str
    user_text: str
    ai_text: str
    emotion: str
    timestamp: float
    audio_context: Optional[Dict] = None
    metadata: Optional[Dict] = None

@dataclass
class SessionContext:
    """Represents a conversation session context"""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    conversation_history: List[ConversationTurn]
    current_emotion: str
    user_profile: Optional[Dict] = None
    context_summary: Optional[str] = None

class ContextManager:
    """Manages conversation context using Redis and Pinecone"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 pinecone_api_key: Optional[str] = None,
                 pinecone_environment: Optional[str] = None,
                 pinecone_index_name: str = "oviya-conversations"):
        
        self.redis_url = redis_url
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.pinecone_index_name = pinecone_index_name
        
        # Initialize Redis
        self.redis_client = None
        self._init_redis()
        
        # Initialize Pinecone
        self.pinecone_index = None
        if pinecone_api_key:
            self._init_pinecone()
        
        # In-memory cache for active sessions
        self.active_sessions: Dict[str, SessionContext] = {}
        self.session_timeout = 3600  # 1 hour
        
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            # Fallback to in-memory storage
            self.redis_client = None
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            pinecone.init(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment
            )
            
            # Get or create index
            if self.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.pinecone_index_name,
                    dimension=384,  # Sentence transformer dimension
                    metric="cosine"
                )
            
            self.pinecone_index = pinecone.Index(self.pinecone_index_name)
            logger.info("✅ Pinecone connection established")
            
        except Exception as e:
            logger.error(f"❌ Pinecone connection failed: {e}")
            self.pinecone_index = None
    
    def create_session(self, user_id: str) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            conversation_history=[],
            current_emotion="empathetic"
        )
        
        # Store in Redis
        if self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(asdict(session_context), default=str)
            )
        
        # Store in memory
        self.active_sessions[session_id] = session_context
        
        logger.info(f"✅ Created session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session context"""
        # Check memory first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Check Redis
        if self.redis_client:
            session_data = self.redis_client.get(f"session:{session_id}")
            if session_data:
                session_dict = json.loads(session_data)
                session_context = SessionContext(**session_dict)
                self.active_sessions[session_id] = session_context
                return session_context
        
        return None
    
    def add_conversation_turn(self, 
                            session_id: str,
                            user_text: str,
                            ai_text: str,
                            emotion: str,
                            audio_context: Optional[Dict] = None) -> str:
        """Add a conversation turn to the session"""
        
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Create conversation turn
        turn_id = str(uuid.uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=session_id,
            user_text=user_text,
            ai_text=ai_text,
            emotion=emotion,
            timestamp=time.time(),
            audio_context=audio_context
        )
        
        # Add to session
        session.conversation_history.append(turn)
        session.last_activity = time.time()
        session.current_emotion = emotion
        
        # Update Redis
        if self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(asdict(session), default=str)
            )
        
        # Update Pinecone with conversation embedding
        if self.pinecone_index:
            self._update_pinecone_context(session, turn)
        
        logger.info(f"✅ Added turn {turn_id} to session {session_id}")
        return turn_id
    
    def _update_pinecone_context(self, session: SessionContext, turn: ConversationTurn):
        """Update Pinecone with conversation context"""
        try:
            # Create context text
            context_text = f"User: {turn.user_text}\nAI: {turn.ai_text}\nEmotion: {turn.emotion}"
            
            # Generate embedding (simplified - in production, use sentence transformer)
            embedding = self._generate_simple_embedding(context_text)
            
            # Upsert to Pinecone
            self.pinecone_index.upsert([
                {
                    "id": turn.turn_id,
                    "values": embedding,
                    "metadata": {
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "timestamp": turn.timestamp,
                        "emotion": turn.emotion,
                        "text": context_text
                    }
                }
            ])
            
        except Exception as e:
            logger.error(f"❌ Error updating Pinecone: {e}")
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simple embedding (temporary implementation for sentence transformer)"""
        # In production, use sentence-transformers
        # For now, create a simple hash-based embedding
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 384-dimensional vector
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        
        return embedding
    
    def get_relevant_context(self, session_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Get relevant conversation context using Pinecone"""
        if not self.pinecone_index:
            # Fallback to recent conversation history
            session = self.get_session(session_id)
            if session:
                recent_turns = session.conversation_history[-limit:]
                return [asdict(turn) for turn in recent_turns]
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_simple_embedding(query)
            
            # Search Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=limit,
                filter={"session_id": session_id},
                include_metadata=True
            )
            
            # Extract relevant context
            relevant_context = []
            for match in results.matches:
                relevant_context.append({
                    "turn_id": match.id,
                    "metadata": match.metadata,
                    "score": match.score
                })
            
            return relevant_context
            
        except Exception as e:
            logger.error(f"❌ Error getting relevant context: {e}")
            return []
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            if self.redis_client:
                self.redis_client.delete(f"session:{session_id}")
        
        if expired_sessions:
            logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")

# FastAPI app
app = FastAPI(title="Context Manager Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global context manager
context_manager = ContextManager()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "context-manager",
        "redis_connected": context_manager.redis_client is not None,
        "pinecone_connected": context_manager.pinecone_index is not None,
        "active_sessions": len(context_manager.active_sessions)
    }

@app.post("/context/session/create")
async def create_session(request: dict):
    """Create a new conversation session"""
    user_id = request.get("user_id", "anonymous")
    session_id = context_manager.create_session(user_id)
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": time.time()
    }

@app.get("/context/session/{session_id}")
async def get_session(session_id: str):
    """Get session context"""
    session = context_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at,
        "last_activity": session.last_activity,
        "current_emotion": session.current_emotion,
        "conversation_count": len(session.conversation_history)
    }

@app.post("/context/session/{session_id}/turn")
async def add_turn(session_id: str, request: dict):
    """Add a conversation turn"""
    user_text = request.get("user_text", "")
    ai_text = request.get("ai_text", "")
    emotion = request.get("emotion", "empathetic")
    audio_context = request.get("audio_context")
    
    turn_id = context_manager.add_conversation_turn(
        session_id, user_text, ai_text, emotion, audio_context
    )
    
    return {
        "turn_id": turn_id,
        "session_id": session_id,
        "timestamp": time.time()
    }

@app.get("/context/session/{session_id}/relevant")
async def get_relevant_context(session_id: str, query: str, limit: int = 5):
    """Get relevant conversation context"""
    relevant_context = context_manager.get_relevant_context(session_id, query, limit)
    
    return {
        "session_id": session_id,
        "query": query,
        "relevant_context": relevant_context,
        "count": len(relevant_context)
    }

@app.post("/context/cleanup")
async def cleanup_sessions():
    """Clean up expired sessions"""
    context_manager.cleanup_expired_sessions()
    
    return {
        "status": "cleanup_completed",
        "active_sessions": len(context_manager.active_sessions),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Context Manager Service...")
    uvicorn.run(app, host="0.0.0.0", port=8003)
