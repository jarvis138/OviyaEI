#!/usr/bin/env python3
"""
Oviya Orchestrator Service
Coordinates ASR → LLM → CSM pipeline with real-time streaming
"""
import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp
import websockets
from collections import deque
import uuid
import os
from datetime import datetime
from service_config import get_service_urls

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SessionState:
    """Represents a conversation session state"""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    conversation_history: List[Dict]
    current_emotion: str
    is_ai_speaking: bool
    interrupt_count: int
    total_messages: int

@dataclass
class PipelineRequest:
    """Represents a pipeline request"""
    request_id: str
    session_id: str
    text: str
    emotion: str
    priority: str
    timestamp: float

@dataclass
class PipelineResponse:
    """Represents a pipeline response"""
    request_id: str
    session_id: str
    audio_chunks: List[bytes]
    text: str
    emotion: str
    total_latency_ms: float
    timestamp: float

class InterruptHandler:
    """Handles interrupt detection and cancellation"""
    
    def __init__(self):
        self.active_generations = {}  # request_id -> cancellation_event
        self.interrupt_stats = {
            "total_interrupts": 0,
            "successful_interrupts": 0,
            "avg_interrupt_latency_ms": 0
        }
    
    async def register_generation(self, request_id: str) -> asyncio.Event:
        """Register a generation for potential interruption"""
        cancellation_event = asyncio.Event()
        self.active_generations[request_id] = cancellation_event
        return cancellation_event
    
    async def interrupt_generation(self, request_id: str) -> bool:
        """Interrupt a specific generation"""
        if request_id in self.active_generations:
            self.active_generations[request_id].set()
            self.interrupt_stats["total_interrupts"] += 1
            logger.info(f"Interrupted generation: {request_id}")
            return True
        return False
    
    async def interrupt_all_generations(self) -> int:
        """Interrupt all active generations"""
        interrupted_count = 0
        for request_id, event in self.active_generations.items():
            event.set()
            interrupted_count += 1
        
        self.interrupt_stats["total_interrupts"] += interrupted_count
        logger.info(f"Interrupted {interrupted_count} generations")
        return interrupted_count
    
    def cleanup_generation(self, request_id: str):
        """Clean up completed generation"""
        if request_id in self.active_generations:
            del self.active_generations[request_id]
    
    def get_interrupt_stats(self) -> Dict:
        """Get interrupt statistics"""
        return self.interrupt_stats

class MemoryManager:
    """Manages conversation memory and context"""
    
    def __init__(self, max_history_turns: int = 10, max_tokens: int = 4000):
        self.max_history_turns = max_history_turns
        self.max_tokens = max_tokens
        self.sessions: Dict[str, SessionState] = {}
        
        # Memory statistics
        self.memory_stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_messages": 0,
            "avg_session_duration_minutes": 0
        }
    
    def create_session(self, session_id: str, user_id: str) -> SessionState:
        """Create a new conversation session"""
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            last_activity=time.time(),
            conversation_history=[],
            current_emotion="empathetic",
            is_ai_speaking=False,
            interrupt_count=0,
            total_messages=0
        )
        
        self.sessions[session_id] = session
        self.memory_stats["total_sessions"] += 1
        self.memory_stats["active_sessions"] += 1
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = time.time()
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add message to session history"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        session.conversation_history.append(message)
        session.total_messages += 1
        
        # Limit history length
        if len(session.conversation_history) > self.max_history_turns:
            session.conversation_history = session.conversation_history[-self.max_history_turns:]
        
        self.memory_stats["total_messages"] += 1
        
        logger.debug(f"Added message to session {session_id}: {role}")
    
    def get_conversation_context(self, session_id: str) -> List[Dict]:
        """Get conversation context for LLM"""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        return session.conversation_history
    
    def set_emotion(self, session_id: str, emotion: str):
        """Set session emotion"""
        if session_id in self.sessions:
            self.sessions[session_id].current_emotion = emotion
    
    def set_ai_speaking(self, session_id: str, is_speaking: bool):
        """Set AI speaking state"""
        if session_id in self.sessions:
            self.sessions[session_id].is_ai_speaking = is_speaking
    
    def increment_interrupt_count(self, session_id: str):
        """Increment interrupt count for session"""
        if session_id in self.sessions:
            self.sessions[session_id].interrupt_count += 1
    
    def cleanup_old_sessions(self, max_idle_minutes: int = 30):
        """Clean up old inactive sessions"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            idle_minutes = (current_time - session.last_activity) / 60
            
            if idle_minutes > max_idle_minutes:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            self.memory_stats["active_sessions"] -= 1
            logger.info(f"Cleaned up old session: {session_id}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        return self.memory_stats

class LLMClient:
    """Client for LLM service (Gemini 2.0 Flash)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.session = None
        
        # Performance tracking
        self.llm_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0,
            "success_rate": 0
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("LLM client initialized")
    
    async def generate_response(self, 
                              text: str, 
                              conversation_context: List[Dict],
                              emotion: str = "empathetic") -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        
        start_time = time.time()
        
        try:
            # Prepare conversation context
            messages = []
            
            # Add system message with emotion context
            system_message = f"You are an empathetic AI assistant. Respond with {emotion} tone. Keep responses conversational and under 100 words."
            messages.append({"role": "user", "parts": [{"text": system_message}]})
            
            # Add conversation history
            for msg in conversation_context[-5:]:  # Last 5 messages
                role = "user" if msg["role"] == "0" else "model"
                messages.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            # Add current user message
            messages.append({"role": "user", "parts": [{"text": text}]})
            
            # Prepare request
            request_data = {
                "contents": messages,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200,
                    "stream": True
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key
            }
            
            # Make streaming request
            async with self.session.post(
                self.base_url,
                json=request_data,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    logger.error(f"LLM API error: {response.status}")
                    yield "I'm sorry, I'm having trouble responding right now."
                    return
                
                # Process streaming response
                async for line in response.content:
                    if line:
                        try:
                            # Parse streaming response
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str == '[DONE]':
                                    break
                                
                                data = json.loads(data_str)
                                
                                if 'candidates' in data and len(data['candidates']) > 0:
                                    candidate = data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                yield part['text']
                        
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error parsing LLM response: {e}")
                            continue
            
            # Update statistics
            latency_ms = (time.time() - start_time) * 1000
            self.llm_stats["total_requests"] += 1
            self.llm_stats["avg_latency_ms"] = (
                (self.llm_stats["avg_latency_ms"] * (self.llm_stats["total_requests"] - 1) + latency_ms) 
                / self.llm_stats["total_requests"]
            )
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            yield "I'm sorry, I'm having trouble responding right now."
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

class CSMClient:
    """Client for CSM TTS service"""
    
    def __init__(self, csm_service_url: str):
        self.csm_service_url = csm_service_url
        self.session = None
        
        # Performance tracking
        self.csm_stats = {
            "total_requests": 0,
            "total_audio_duration_ms": 0,
            "avg_latency_ms": 0,
            "success_rate": 0
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("CSM client initialized")
    
    async def generate_audio_stream(self, 
                                  text: str, 
                                  emotion: str,
                                  session_context: List[Dict] = None) -> AsyncGenerator[bytes, None]:
        """Generate streaming audio from CSM"""
        
        start_time = time.time()
        
        try:
            # Prepare request
            request_data = {
                "text": text,
                "emotion": emotion,
                "session_id": "orchestrator",
                "priority": "normal"
            }
            
            # Make request to CSM service
            async with self.session.post(
                f"{self.csm_service_url}/tts",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    logger.error(f"CSM service error: {response.status}")
                    return
                
                # Process response
                response_data = await response.json()
                
                if 'audio_base64' in response_data:
                    # Decode base64 audio and yield it
                    import base64
                    audio_data = base64.b64decode(response_data['audio_base64'])
                    yield audio_data
            
            # Update statistics
            latency_ms = (time.time() - start_time) * 1000
            self.csm_stats["total_requests"] += 1
            self.csm_stats["avg_latency_ms"] = (
                (self.csm_stats["avg_latency_ms"] * (self.csm_stats["total_requests"] - 1) + latency_ms) 
                / self.csm_stats["total_requests"]
            )
            
        except Exception as e:
            logger.error(f"Error in CSM generation: {e}")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

class OrchestratorPipeline:
    """Main orchestrator pipeline"""
    
    def __init__(self, 
                 llm_api_key: str,
                 csm_service_url: str = None,
                 asr_service_url: str = None):
        
        # Get service URLs from configuration
        if csm_service_url is None or asr_service_url is None:
            urls = get_service_urls()
            csm_service_url = csm_service_url or urls["csm_url"]
            asr_service_url = asr_service_url or urls["asr_url"]
        
        # Service URLs
        self.csm_service_url = csm_service_url
        self.asr_service_url = asr_service_url
        
        # Initialize clients
        self.llm_client = LLMClient(llm_api_key)
        self.csm_client = CSMClient(csm_service_url)
        self.memory_manager = MemoryManager()
        self.interrupt_handler = InterruptHandler()
        
        # Pipeline statistics
        self.pipeline_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_end_to_end_latency_ms": 0
        }
    
    async def initialize(self):
        """Initialize orchestrator pipeline"""
        logger.info("Initializing Orchestrator Pipeline...")
        
        await self.llm_client.initialize()
        await self.csm_client.initialize()
        
        logger.info("Orchestrator Pipeline initialized")
    
    async def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """Process a complete pipeline request"""
        
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Register for potential interruption
            cancellation_event = await self.interrupt_handler.register_generation(request_id)
            
            # Get session
            session = self.memory_manager.get_session(request.session_id)
            if not session:
                session = self.memory_manager.create_session(request.session_id, "anonymous")
            
            # Update session activity
            self.memory_manager.update_session_activity(request.session_id)
            
            # Add user message to memory
            self.memory_manager.add_message(
                request.session_id,
                "user",
                request.text
            )
            
            # Generate LLM response
            logger.info(f"Generating LLM response for: {request.text[:50]}...")
            
            llm_response_text = ""
            async for text_chunk in self.llm_client.generate_response(
                request.text,
                self.memory_manager.get_conversation_context(request.session_id),
                request.emotion
            ):
                # Check for interruption
                if cancellation_event.is_set():
                    logger.info(f"LLM generation interrupted: {request_id}")
                    return PipelineResponse(
                        request_id=request_id,
                        session_id=request.session_id,
                        audio_chunks=[],
                        text="",
                        emotion=request.emotion,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        timestamp=time.time()
                    )
                
                llm_response_text += text_chunk
            
            if not llm_response_text.strip():
                llm_response_text = "I understand. How can I help you?"
            
            # Add AI response to memory
            self.memory_manager.add_message(
                request.session_id,
                "assistant",
                llm_response_text
            )
            
            # Generate CSM audio
            logger.info(f"Generating CSM audio for: {llm_response_text[:50]}...")
            
            audio_chunks = []
            async for audio_chunk in self.csm_client.generate_audio_stream(
                llm_response_text,
                request.emotion,
                self.memory_manager.get_conversation_context(request.session_id)
            ):
                # Check for interruption
                if cancellation_event.is_set():
                    logger.info(f"CSM generation interrupted: {request_id}")
                    break
                
                audio_chunks.append(audio_chunk)
            
            # Clean up generation
            self.interrupt_handler.cleanup_generation(request_id)
            
            # Update statistics
            total_latency_ms = (time.time() - start_time) * 1000
            self.pipeline_stats["total_requests"] += 1
            self.pipeline_stats["successful_requests"] += 1
            self.pipeline_stats["avg_end_to_end_latency_ms"] = (
                (self.pipeline_stats["avg_end_to_end_latency_ms"] * (self.pipeline_stats["total_requests"] - 1) + total_latency_ms) 
                / self.pipeline_stats["total_requests"]
            )
            
            return PipelineResponse(
                request_id=request_id,
                session_id=request.session_id,
                audio_chunks=audio_chunks,
                text=llm_response_text,
                emotion=request.emotion,
                total_latency_ms=total_latency_ms,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            
            # Clean up generation
            self.interrupt_handler.cleanup_generation(request_id)
            
            # Update statistics
            self.pipeline_stats["total_requests"] += 1
            self.pipeline_stats["failed_requests"] += 1
            
            return PipelineResponse(
                request_id=request_id,
                session_id=request.session_id,
                audio_chunks=[],
                text="I'm sorry, I'm having trouble responding right now.",
                emotion=request.emotion,
                total_latency_ms=(time.time() - start_time) * 1000,
                timestamp=time.time()
            )
    
    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt all generations for a session"""
        session = self.memory_manager.get_session(session_id)
        if not session:
            return False
        
        # Interrupt all active generations
        interrupted_count = await self.interrupt_handler.interrupt_all_generations()
        
        if interrupted_count > 0:
            self.memory_manager.increment_interrupt_count(session_id)
            self.memory_manager.set_ai_speaking(session_id, False)
            logger.info(f"Interrupted {interrupted_count} generations for session {session_id}")
        
        return interrupted_count > 0
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "pipeline_stats": self.pipeline_stats,
            "memory_stats": self.memory_manager.get_memory_stats(),
            "interrupt_stats": self.interrupt_handler.get_interrupt_stats(),
            "llm_stats": self.llm_client.llm_stats,
            "csm_stats": self.csm_client.csm_stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.llm_client.close()
        await self.csm_client.close()
        self.memory_manager.cleanup_old_sessions()

# FastAPI Application
app = FastAPI(title="Oviya Orchestrator Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global pipeline
    
    logger.info("Starting Orchestrator Service...")
    
    # Get API key from environment
    llm_api_key = os.getenv("GEMINI_API_KEY")
    if not llm_api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise RuntimeError("GEMINI_API_KEY not configured")
    
    pipeline = OrchestratorPipeline(llm_api_key)
    await pipeline.initialize()
    
    logger.info("Orchestrator Service ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pipeline
    if pipeline:
        await pipeline.cleanup()
    logger.info("Orchestrator Service stopped")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "timestamp": time.time(),
        "active_sessions": pipeline.memory_manager.memory_stats["active_sessions"]
    }

@app.post("/session/create")
async def create_session(request: Dict):
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    user_id = request.get("user_id", "anonymous")
    
    session = pipeline.memory_manager.create_session(session_id, user_id)
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": session.created_at,
        "emotion": session.current_emotion
    }

@app.post("/session/{session_id}/message")
async def send_message(session_id: str, request: Dict):
    """Send a message and get response"""
    
    pipeline_request = PipelineRequest(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        text=request.get("text", ""),
        emotion=request.get("emotion", "empathetic"),
        priority=request.get("priority", "normal"),
        timestamp=time.time()
    )
    
    if not pipeline_request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Process request
    response = await pipeline.process_request(pipeline_request)
    
    return {
        "request_id": response.request_id,
        "text": response.text,
        "emotion": response.emotion,
        "total_latency_ms": response.total_latency_ms,
        "audio_chunks": [chunk.hex() for chunk in response.audio_chunks]
    }

@app.post("/session/{session_id}/interrupt")
async def interrupt_session(session_id: str):
    """Interrupt current generation for session"""
    
    success = await pipeline.interrupt_session(session_id)
    
    return {
        "session_id": session_id,
        "interrupted": success,
        "timestamp": time.time()
    }

@app.websocket("/session/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str):
    """WebSocket streaming session"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                # Process message
                pipeline_request = PipelineRequest(
                    request_id=str(uuid.uuid4()),
                    session_id=session_id,
                    text=data.get("text", ""),
                    emotion=data.get("emotion", "empathetic"),
                    priority=data.get("priority", "normal"),
                    timestamp=time.time()
                )
                
                if not pipeline_request.text:
                    await websocket.send_json({"error": "Text is required"})
                    continue
                
                # Process request
                response = await pipeline.process_request(pipeline_request)
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "request_id": response.request_id,
                    "text": response.text,
                    "emotion": response.emotion,
                    "total_latency_ms": response.total_latency_ms,
                    "audio_chunks": [chunk.hex() for chunk in response.audio_chunks]
                })
            
            elif data.get("type") == "interrupt":
                # Handle interrupt
                success = await pipeline.interrupt_session(session_id)
                await websocket.send_json({
                    "type": "interrupt_result",
                    "interrupted": success,
                    "timestamp": time.time()
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket stream: {e}")
        await websocket.send_json({"error": str(e)})

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    return pipeline.get_pipeline_stats()

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
