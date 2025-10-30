#!/usr/bin/env python3
"""
Comprehensive test suite for Oviya MCP integrations
Tests all MCP systems: memory, crisis detection, empathic thinking, personality
"""

import asyncio
import pytest
import json
import time
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from production.brain.mcp_memory_integration import OviyaMemorySystem
from production.brain.crisis_detection import CrisisDetectionSystem
from production.brain.empathic_thinking import EmpathicThinkingEngine


class TestOviyaMCPIntegrations:
    """Test suite for all MCP integrations"""

    @pytest.fixture
    async def memory_system(self):
        """Initialize memory system for testing"""
        system = OviyaMemorySystem()
        await system.initialize_mcp_clients()
        yield system
        # Cleanup if needed

    @pytest.fixture
    async def crisis_detector(self):
        """Initialize crisis detection system"""
        detector = CrisisDetectionSystem()
        yield detector

    @pytest.fixture
    async def empathy_engine(self):
        """Initialize empathic thinking engine"""
        engine = EmpathicThinkingEngine()
        yield engine

    @pytest.mark.asyncio
    async def test_memory_system_initialization(self, memory_system):
        """Test that memory system initializes correctly"""
        assert memory_system.chroma_client is not None
        assert memory_system.conversation_collection is not None
        assert memory_system.personality_collection is not None

    @pytest.mark.asyncio
    async def test_conversation_memory_storage(self, memory_system):
        """Test storing conversation memory"""
        test_user = "test_user_123"
        test_data = {
            "user_input": "I'm feeling really anxious about work",
            "response": "I hear your anxiety about work. That sounds really challenging.",
            "timestamp": time.time(),
            "emotion": "anxious",
            "personality_vector": {
                "Ma": 0.3, "Ahimsa": 0.4, "Jeong": 0.15, "Logos": 0.1, "Lagom": 0.05
            },
            "session_id": "test_session_001"
        }

        success = await memory_system.store_conversation_memory(test_user, test_data)
        assert success == True

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory_system):
        """Test retrieving relevant memories"""
        test_user = "test_user_123"
        query = "feeling anxious about work"

        memories = await memory_system.retrieve_relevant_memories(test_user, query, limit=3)

        assert isinstance(memories, dict)
        assert "conversation_history" in memories
        assert "personality_evolution" in memories
        assert "semantic_matches" in memories
        assert "recent_context" in memories

    @pytest.mark.asyncio
    async def test_crisis_detection_basic(self, crisis_detector):
        """Test basic crisis detection functionality"""
        # Test non-crisis message
        normal_message = "I'm feeling a bit tired today"
        history = ["Hello", "How are you doing?"]

        assessment = await crisis_detector.assess_crisis_risk(normal_message, history)

        assert assessment["crisis_detected"] == False
        assert assessment["risk_level"] in ["low", "none"]
        assert "recommended_actions" in assessment

    @pytest.mark.asyncio
    async def test_crisis_detection_suicidal(self, crisis_detector):
        """Test crisis detection for suicidal ideation"""
        crisis_message = "I don't want to live anymore"
        history = ["I've been really struggling lately"]

        assessment = await crisis_detector.assess_crisis_risk(crisis_message, history)

        assert assessment["crisis_detected"] == True
        assert assessment["risk_level"] in ["high", "critical"]
        assert assessment["escalation_needed"] == True
        assert "immediate_response" in assessment

    @pytest.mark.asyncio
    async def test_empathic_thinking_basic(self, empathy_engine):
        """Test basic empathic thinking functionality"""
        user_input = "I'm feeling really overwhelmed"
        personality_vector = {
            "Ma": 0.3, "Ahimsa": 0.4, "Jeong": 0.15, "Logos": 0.1, "Lagom": 0.05
        }
        emotion_context = {
            "emotion": "anxious",
            "intensity": 0.7,
            "patterns": [],
            "conflicts": []
        }

        response = await empathy_engine.generate_empathic_response(
            user_input, personality_vector, emotion_context
        )

        assert "response" in response
        assert "thinking_modes_used" in response
        assert "pillar_weights" in response
        assert "emotional_insights" in response
        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_empathic_thinking_modes(self, empathy_engine):
        """Test that multiple thinking modes are used"""
        user_input = "I'm both excited and scared about this new job"
        personality_vector = {
            "Ma": 0.3, "Ahimsa": 0.4, "Jeong": 0.15, "Logos": 0.1, "Lagom": 0.05
        }
        emotion_context = {
            "emotion": "mixed",
            "intensity": 0.6,
            "patterns": ["mixed emotions", "career changes"],
            "conflicts": ["excited", "scared"]
        }

        response = await empathy_engine.generate_empathic_response(
            user_input, personality_vector, emotion_context
        )

        assert "thinking_modes_used" in response
        # Should use multiple modes for complex emotions
        assert len(response["thinking_modes_used"]) >= 2

    @pytest.mark.asyncio
    async def test_personality_vector_computation(self):
        """Test personality vector computation"""
        from production.websocket_server import ConversationSession

        session = ConversationSession("test_user")
        vector = await session._compute_personality_vector("I'm feeling sad", "sad")

        assert isinstance(vector, dict)
        assert all(key in vector for key in ["Ma", "Ahimsa", "Jeong", "Logos", "Lagom"])
        assert all(isinstance(v, float) for v in vector.values())
        assert all(0 <= v <= 1 for v in vector.values())

        # Check that sad emotion increases certain pillars
        assert vector["Jeong"] > 0.15  # Base value
        assert vector["Ahimsa"] > 0.4   # Base value

    @pytest.mark.asyncio
    async def test_emergency_resources(self, crisis_detector):
        """Test emergency resource retrieval"""
        resources = await crisis_detector.get_emergency_resources("us")

        assert isinstance(resources, dict)
        assert "hotlines" in resources or "resources" in resources

    @pytest.mark.asyncio
    async def test_memory_persistence(self, memory_system):
        """Test that memories persist across sessions"""
        test_user = "persistence_test_user"
        test_data = {
            "user_input": "I need help with my anxiety",
            "response": "I'm here to support you through this anxiety.",
            "timestamp": time.time(),
            "emotion": "anxious",
            "personality_vector": {
                "Ma": 0.3, "Ahimsa": 0.5, "Jeong": 0.1, "Logos": 0.05, "Lagom": 0.05
            },
            "session_id": "persistence_session"
        }

        # Store memory
        await memory_system.store_conversation_memory(test_user, test_data)

        # Retrieve memory
        memories = await memory_system.retrieve_relevant_memories(
            test_user, "anxiety help", limit=5
        )

        assert len(memories.get("conversation_history", [])) >= 1

        # Check that personality evolution is tracked
        personality_evolution = memories.get("personality_evolution", [])
        assert len(personality_evolution) >= 1

    @pytest.mark.asyncio
    async def test_cognitive_depth_calculation(self, empathy_engine):
        """Test cognitive depth scoring"""
        # Test different combinations of thinking modes
        test_cases = [
            (["empathetic"], "focused_empathy"),
            (["empathetic", "reflective"], "dual_awareness"),
            (["empathetic", "reflective", "metacognitive"], "multi_perspective"),
            (["empathetic", "dialectical", "reflective", "creative"], "deep_integration")
        ]

        for modes, expected_depth in test_cases:
            # Access private method for testing
            depth = empathy_engine._calculate_cognitive_depth(modes)
            assert depth == expected_depth

    @pytest.mark.asyncio
    async def test_end_to_end_conversation_flow(self, memory_system, crisis_detector, empathy_engine):
        """Test complete conversation flow with all MCP systems"""
        user_id = "e2e_test_user"
        user_input = "I'm feeling really down and don't know what to do"

        # Step 1: Crisis assessment
        crisis_assessment = await crisis_detector.assess_crisis_risk(
            user_input, ["I've been struggling lately"]
        )

        # Should detect as moderate risk, not critical
        assert crisis_assessment["risk_level"] in ["moderate", "low"]
        assert crisis_assessment["escalation_needed"] == False

        # Step 2: Retrieve relevant memories (none initially)
        memories = await memory_system.retrieve_relevant_memories(user_id, user_input)

        # Step 3: Generate empathic response
        personality_vector = {
            "Ma": 0.3, "Ahimsa": 0.4, "Jeong": 0.2, "Logos": 0.05, "Lagom": 0.05
        }
        emotion_context = {
            "emotion": "sad",
            "intensity": 0.7,
            "patterns": [],
            "conflicts": []
        }

        response = await empathy_engine.generate_empathic_response(
            user_input, personality_vector, emotion_context
        )

        assert "response" in response
        assert len(response["response"]) > 0
        assert "thinking_modes_used" in response

        # Step 4: Store conversation memory
        conversation_data = {
            "user_input": user_input,
            "response": response["response"],
            "timestamp": time.time(),
            "emotion": "sad",
            "personality_vector": personality_vector,
            "session_id": "e2e_test_session"
        }

        success = await memory_system.store_conversation_memory(user_id, conversation_data)
        assert success == True

        # Step 5: Verify memory retrieval
        retrieved = await memory_system.retrieve_relevant_memories(
            user_id, "feeling down", limit=3
        )

        assert len(retrieved.get("conversation_history", [])) >= 1

    @pytest.mark.asyncio
    async def test_error_handling(self, memory_system, crisis_detector):
        """Test error handling in MCP systems"""
        # Test memory system with invalid data
        invalid_data = {
            "user_input": None,  # Invalid
            "response": "test",
            "timestamp": time.time(),
            "emotion": "neutral",
            "personality_vector": {}
        }

        # Should handle gracefully
        success = await memory_system.store_conversation_memory("test_user", invalid_data)
        # May succeed or fail gracefully - both are acceptable
        assert isinstance(success, bool)

        # Test crisis detection with empty inputs
        assessment = await crisis_detector.assess_crisis_risk("", [])
        assert "risk_level" in assessment
        assert assessment["risk_level"] == "none"


if __name__ == "__main__":
    # Run tests
    asyncio.run(asyncio.gather(*[
        test_memory_system_initialization(None),
        test_crisis_detection_basic(None),
        test_empathic_thinking_basic(None),
    ]))
    print("MCP integration tests completed!")
