#!/usr/bin/env python3
"""
Oviya MCP Memory Integration System
Combines multiple memory sources for persistent user context
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class OviyaMemorySystem:
    """
    Multi-layered memory system for persistent user context

    Layers:
    1. OpenMemory - Semantic conversation memory
    2. ChromaDB - Vector embeddings for personality evolution
    3. Local cache - Fast access to recent conversations
    """

    def __init__(self):
        # Initialize ChromaDB for personality vectors
        self.chroma_client = chromadb.PersistentClient(
            path=str(project_root / "mcp-ecosystem" / "data" / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collections
        self.personality_collection = self.chroma_client.get_or_create_collection(
            name="oviya_personality_vectors",
            metadata={"description": "User personality vector evolution over time"}
        )

        self.conversation_collection = self.chroma_client.get_or_create_collection(
            name="oviya_conversations",
            metadata={"description": "Semantic conversation memory"}
        )

        # Local cache for recent conversations
        self.recent_cache = {}
        self.cache_size = 100

        # MCP client placeholders (will be replaced with actual MCP clients when available)
        self.openmemory_client = None
        self.rag_memory_client = None

    async def initialize_mcp_clients(self):
        """Initialize MCP clients for external memory systems"""
        try:
            # These would be real MCP clients when the SDK is available
            self.openmemory_client = MockMCPClient("openmemory")
            self.rag_memory_client = MockMCPClient("rag-memory")
        except Exception as e:
            print(f"MCP client initialization failed: {e}")

    async def store_conversation_memory(self, user_id: str, conversation_data: Dict[str, Any]) -> bool:
        """
        Store conversation with multiple memory layers

        Args:
            user_id: Unique user identifier
            conversation_data: {
                "user_input": str,
                "response": str,
                "timestamp": float,
                "emotion": str,
                "personality_vector": Dict[str, float],
                "emotion_context": Dict,
                "session_id": str
            }
        """

        try:
            # 1. Store in ChromaDB personality collection
            vector_json = json.dumps(conversation_data["personality_vector"])
            self.personality_collection.add(
                documents=[vector_json],
                metadatas=[{
                    "user_id": user_id,
                    "timestamp": conversation_data["timestamp"],
                    "emotion": conversation_data["emotion"],
                    "session_id": conversation_data.get("session_id", "")
                }],
                ids=[f"{user_id}_{conversation_data['timestamp']}"]
            )

            # 2. Store semantic conversation memory
            conversation_text = f"User: {conversation_data['user_input']} | Oviya: {conversation_data['response']}"
            self.conversation_collection.add(
                documents=[conversation_text],
                metadatas=[{
                    "user_id": user_id,
                    "timestamp": conversation_data["timestamp"],
                    "emotion": conversation_data["emotion"],
                    "personality_vector": vector_json
                }],
                ids=[f"conv_{user_id}_{conversation_data['timestamp']}"]
            )

            # 3. Update local cache
            if user_id not in self.recent_cache:
                self.recent_cache[user_id] = []
            self.recent_cache[user_id].append(conversation_data)
            if len(self.recent_cache[user_id]) > self.cache_size:
                self.recent_cache[user_id].pop(0)

            # 4. Store in external memory systems (when available)
            if self.openmemory_client:
                await self._store_openmemory(user_id, conversation_data)
            if self.rag_memory_client:
                await self._store_rag_memory(user_id, conversation_data)

            return True

        except Exception as e:
            print(f"Memory storage failed: {e}")
            return False

    async def retrieve_relevant_memories(self, user_id: str, current_context: str,
                                       limit: int = 5, days_back: int = 30) -> Dict[str, Any]:
        """
        Retrieve relevant memories from all layers

        Returns:
            {
                "personality_evolution": [...],
                "conversation_history": [...],
                "semantic_matches": [...],
                "recent_context": [...]
            }
        """

        try:
            # Calculate timestamp threshold
            cutoff_date = datetime.now() - timedelta(days=days_back)
            cutoff_timestamp = cutoff_date.timestamp()

            # 1. Get personality evolution
            personality_results = self.personality_collection.query(
                query_texts=[current_context],
                n_results=limit,
                where={"user_id": user_id, "timestamp": {"$gte": cutoff_timestamp}}
            )

            personality_evolution = []
            for i, doc in enumerate(personality_results["documents"]):
                try:
                    vector = json.loads(doc)
                    metadata = personality_results["metadatas"][i]
                    personality_evolution.append({
                        "vector": vector,
                        "timestamp": metadata["timestamp"],
                        "emotion": metadata["emotion"]
                    })
                except json.JSONDecodeError:
                    continue

            # 2. Get conversation history
            conversation_results = self.conversation_collection.query(
                query_texts=[current_context],
                n_results=limit * 2,  # Get more for filtering
                where={"user_id": user_id, "timestamp": {"$gte": cutoff_timestamp}}
            )

            conversation_history = []
            for i, doc in enumerate(conversation_results["documents"][:limit]):
                metadata = conversation_results["metadatas"][i]
                conversation_history.append({
                    "text": doc,
                    "timestamp": metadata["timestamp"],
                    "emotion": metadata["emotion"]
                })

            # 3. Get recent context from cache
            recent_context = self.recent_cache.get(user_id, [])[-limit:]

            # 4. Get semantic matches from external systems
            semantic_matches = []
            if self.openmemory_client:
                semantic_matches.extend(await self._query_openmemory(user_id, current_context, limit))
            if self.rag_memory_client:
                semantic_matches.extend(await self._query_rag_memory(user_id, current_context, limit))

            return {
                "personality_evolution": personality_evolution,
                "conversation_history": conversation_history,
                "semantic_matches": semantic_matches[:limit],
                "recent_context": recent_context,
                "total_memories": len(personality_evolution) + len(conversation_history)
            }

        except Exception as e:
            print(f"Memory retrieval failed: {e}")
            return {
                "personality_evolution": [],
                "conversation_history": [],
                "semantic_matches": [],
                "recent_context": self.recent_cache.get(user_id, [])[-limit:],
                "error": str(e)
            }

    async def get_personality_trajectory(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get personality vector evolution over time"""

        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()

            results = self.personality_collection.query(
                query_texts=["personality evolution"],
                n_results=100,  # Get all recent vectors
                where={"user_id": user_id, "timestamp": {"$gte": cutoff_timestamp}}
            )

            trajectory = []
            for i, doc in enumerate(results["documents"]):
                try:
                    vector = json.loads(doc)
                    metadata = results["metadatas"][i]
                    trajectory.append({
                        "timestamp": metadata["timestamp"],
                        "vector": vector,
                        "emotion": metadata["emotion"]
                    })
                except json.JSONDecodeError:
                    continue

            # Sort by timestamp
            trajectory.sort(key=lambda x: x["timestamp"])

            return {
                "trajectory": trajectory,
                "total_points": len(trajectory),
                "date_range": {
                    "start": trajectory[0]["timestamp"] if trajectory else None,
                    "end": trajectory[-1]["timestamp"] if trajectory else None
                }
            }

        except Exception as e:
            print(f"Personality trajectory retrieval failed: {e}")
            return {"trajectory": [], "error": str(e)}

    async def _store_openmemory(self, user_id: str, conversation_data: Dict[str, Any]):
        """Store in OpenMemory MCP server"""
        if not self.openmemory_client:
            return

        memory_text = f"User {user_id}: {conversation_data['user_input']} | Oviya: {conversation_data['response']}"

        await self.openmemory_client.call_tool("add_memories", {
            "memories": [{
                "text": memory_text,
                "metadata": {
                    "user_id": user_id,
                    "timestamp": conversation_data["timestamp"],
                    "emotion": conversation_data["emotion"],
                    "personality_vector": conversation_data["personality_vector"]
                }
            }]
        })

    async def _store_rag_memory(self, user_id: str, conversation_data: Dict[str, Any]):
        """Store in RAG Memory MCP server"""
        if not self.rag_memory_client:
            return

        # RAG memory would handle vector embeddings and retrieval
        await self.rag_memory_client.call_tool("store_memory", {
            "user_id": user_id,
            "content": conversation_data,
            "embedding_type": "personality_vector"
        })

    async def _query_openmemory(self, user_id: str, query: str, limit: int) -> List[Dict]:
        """Query OpenMemory for semantic matches"""
        if not self.openmemory_client:
            return []

        try:
            results = await self.openmemory_client.call_tool("search_memory", {
                "query": query,
                "user_id": user_id,
                "limit": limit
            })
            return results.get("results", [])
        except Exception as e:
            print(f"OpenMemory query failed: {e}")
            return []

    async def _query_rag_memory(self, user_id: str, query: str, limit: int) -> List[Dict]:
        """Query RAG Memory for semantic matches"""
        if not self.rag_memory_client:
            return []

        try:
            results = await self.rag_memory_client.call_tool("semantic_search", {
                "query": query,
                "user_id": user_id,
                "limit": limit,
                "collection": "conversations"
            })
            return results.get("matches", [])
        except Exception as e:
            print(f"RAG Memory query failed: {e}")
            return []

class MockMCPClient:
    """Mock MCP client until real SDK is available"""

    def __init__(self, server_name: str):
        self.server_name = server_name
        print(f"Mock MCP client initialized for {server_name}")

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool call - would be replaced with real MCP client"""
        print(f"Mock {self.server_name} tool call: {tool_name} with params: {params}")
        return {"mock": True, "tool": tool_name, "params": params}

# Global instance
memory_system = OviyaMemorySystem()

async def initialize_memory_system():
    """Initialize the memory system"""
    await memory_system.initialize_mcp_clients()
    print("Oviya Memory System initialized")

if __name__ == "__main__":
    # Initialize memory system
    asyncio.run(initialize_memory_system())
