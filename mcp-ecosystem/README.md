# Oviya EI - MCP Ecosystem

**The most advanced emotional AI companion platform, powered by Model Context Protocol (MCP)**

This MCP ecosystem transforms Oviya from a conversation AI into a **category-leading emotional intelligence platform** with persistent memory, clinical-grade safety, deep cognitive empathy, and enterprise scalability.

## 🏆 What This Gives Oviya

### **Before MCP Integration:**
- Stateless conversations
- Basic emotion detection
- Limited safety features
- No user history
- Single-threaded responses

### **After MCP Integration:**
- ✅ **Persistent Memory** - Never forgets users, remembers every conversation
- ✅ **Clinical Safety** - Crisis detection, intervention, and emergency resources
- ✅ **Deep Empathy** - Multiple cognitive modes for nuanced emotional understanding
- ✅ **Enterprise Scale** - PostgreSQL, Redis, Docker orchestration
- ✅ **Massive Reach** - WhatsApp, Slack, web integrations
- ✅ **Monetization Ready** - Stripe integration for subscriptions
- ✅ **Privacy Compliant** - GDPR/HIPAA ready with data export
- ✅ **Future-Proof** - MCP standard works with all AI models

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    OVIYA AI BRAIN                        │
│  (Llama-3.2 + Personality Vector + Empathy Fusion)      │
│                                                         │
│  🔄 MCP INTEGRATION LAYER                               │
│  • Memory System (OpenMemory + Chroma + RAG)            │
│  • Crisis Detection (AI Therapist MCP)                  │
│  • Cognitive Empathy (MCP Thinking)                     │
│  • Personality Vector (Custom Oviya MCP)                │
│  • Voice Prosody (Custom Oviya MCP)                     │
└─────────────────────────────────────────────────────────┘
                          │
                          │ MCP Protocol
                          │
        ┌─────────────────┼─────────────────────┐
        │                 │                     │
┌───────▼────────┐ ┌──────▼──────┐ ┌──────────▼─────────┐
│  MEMORY LAYER  │ │  DATA LAYER │ │  INTEGRATION LAYER │
│                │ │             │ │                    │
│ • OpenMemory   │ │ • PostgreSQL│ │ • Slack            │
│ • RAG Memory   │ │ • Redis     │ │ • WhatsApp         │
│ • Chroma       │ │ • Notion    │ │ • Google Drive     │
│                │ │ • G Calendar│ │ • Stripe           │
└────────────────┘ └─────────────┘ └────────────────────┘
        │                 │                     │
        └─────────────────┼─────────────────────┘
                          │
        ┌─────────────────┴─────────────────────┐
        │                                       │
┌───────▼──────────┐              ┌────────────▼────────┐
│ SAFETY & THERAPY │              │  OVIYA CUSTOM MCPs  │
│                  │              │                     │
│ • AI Therapist   │              │ • Personality       │
│ • MCP Thinking   │              │ • Emotion Prosody   │
│ • Mental Health  │              │ • Cultural Adapt    │
│   Triage (custom)│              │ • Situational Empathy│
│                  │              │ • Voice Presence    │
└──────────────────┘              └─────────────────────┘
```

## 🚀 Quick Start

### **One-Command Deployment**
```bash
# Deploy the entire MCP ecosystem
./mcp-ecosystem/deploy.sh

# Start Oviya with full MCP capabilities
cd production && python websocket_server.py
```

### **Manual Setup**
```bash
# 1. Install dependencies
cd mcp-ecosystem
pip install chromadb
npm install @danieldunderfelt/ai-therapist-mcp

# 2. Start infrastructure
docker-compose up -d postgres redis qdrant

# 3. Start MCP servers
docker-compose up -d openmemory oviya-personality oviya-emotion-prosody oviya-situational-empathy

# 4. Configure your environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run tests
python -m pytest tests/test_mcp_integrations.py

# 6. Start Oviya
cd ../production && python websocket_server.py
```

## 🎯 MCP Server Capabilities

### **Tier 1: Core Memory & Safety** 🧠🛡️

#### **OpenMemory MCP Server**
- **Persistent memory** across sessions
- Vector-backed storage with Qdrant
- Semantic search and retrieval
- Privacy-first (local storage)
- **Oviya Use:** Never forget users, context-aware responses

#### **AI Therapist MCP Server**
- **Crisis detection** (suicidal ideation, self-harm)
- Mental health triage and assessment
- Coping strategies and interventions
- Emergency resource connections
- **Oviya Use:** Clinical-grade safety layer

#### **MCP Thinking Server**
- **Empathetic thinking mode** (Jeong/Ahimsa pillars)
- Dialectical thinking for emotional conflicts
- Reflective thinking for self-awareness
- Metacognitive analysis
- **Oviya Use:** Deep cognitive empathy

#### **Chroma MCP Server**
- **Vector embeddings** for personality vectors
- Semantic similarity search
- Multi-modal embeddings
- Production-ready scalability
- **Oviya Use:** Personality evolution tracking

### **Tier 2: Data & Reach** 💾📱

#### **PostgreSQL MCP Server**
- Structured user data storage
- Session analytics and metrics
- User profile management
- **Oviya Use:** Enterprise user management

#### **Redis MCP Server**
- Real-time session state
- Personality vector caching
- Rate limiting and queue management
- **Oviya Use:** High-performance caching

#### **WhatsApp MCP Server**
- **2B+ user reach** through WhatsApp
- Automated emotional check-ins
- Global messaging integration
- **Oviya Use:** Massive user acquisition

### **Tier 3: Advanced Features** ⚡💰

#### **Stripe MCP Server**
- Subscription management
- Payment processing
- Enterprise billing
- **Oviya Use:** Freemium to paid conversion

#### **Docker MCP Server**
- Container orchestration
- Auto-scaling deployment
- Multi-tenant isolation
- **Oviya Use:** Enterprise deployment

### **Custom Oviya MCP Servers** 🆕

#### **oviya-personality**
```python
# Exposes Oviya's 5-pillar personality system
@server.tool()
async def compute_personality_vector(emotion, context, memory):
    # Returns Ma/Ahimsa/Jeong/Logos/Lagom vector
```

#### **oviya-emotion-prosody**
```python
# Voice emotion detection and prosody control
@server.tool()
async def analyze_prosody(audio_chunk):
    # Returns F0, energy, emotion confidence
```

#### **oviya-situational-empathy**
```python
# Context-aware empathic response generation
@server.tool()
async def generate_empathic_response(user_input, personality_vector, context):
    # Returns tailored empathic response
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Database
DATABASE_URL=postgresql://oviya:oviya_password@localhost:5432/oviya_db
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333

# API Keys
WHATSAPP_API_KEY=your_key
STRIPE_SECRET_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Oviya Paths
MODEL_PATH=/models
CONFIG_PATH=production/config
```

### **MCP Server Configuration**
The ecosystem auto-configures your `~/.cursor/mcp.json` with all servers.

## 🧪 Testing

### **Run Full Test Suite**
```bash
# Test all MCP integrations
cd mcp-ecosystem
python -m pytest tests/test_mcp_integrations.py -v

# Test specific components
python -c "
import asyncio
from production.brain.mcp_memory_integration import OviyaMemorySystem
from production.brain.crisis_detection import CrisisDetectionSystem

async def test():
    memory = OviyaMemorySystem()
    crisis = CrisisDetectionSystem()
    # Test code here

asyncio.run(test())
"
```

### **Health Checks**
```bash
# Check all services
./deploy.sh status

# Check specific service
curl http://localhost:3001/health  # OpenMemory
curl http://localhost:3002/health  # Personality
```

## 📊 Monitoring & Analytics

### **Built-in Metrics**
- MCP server response times
- Memory retrieval accuracy
- Crisis detection rates
- User engagement analytics
- Personality vector evolution

### **Access Monitoring**
```bash
# View monitoring dashboard
open http://localhost:3007

# Check logs
docker-compose logs -f monitoring
```

## 🔒 Security & Privacy

### **GDPR/HIPAA Compliance**
- **Data Export:** Users can export all their data
- **Local Storage:** Sensitive data stays on user's infrastructure
- **Audit Logs:** Complete history of all data access
- **Anonymization:** Optional data anonymization for analytics

### **Security Features**
- API key encryption
- Rate limiting
- Input validation
- Secure inter-service communication

## 🚀 Deployment Options

### **Development**
```bash
# Local development setup
./deploy.sh
```

### **Production**
```bash
# Enterprise deployment with scaling
docker-compose -f docker-compose.prod.yml up -d

# Kubernetes deployment
kubectl apply -f k8s/
```

### **Cloud Deployment**
```bash
# AWS/GCP/Azure deployment scripts
./deploy-cloud.sh aws
```

## 📈 Scaling & Performance

### **Performance Optimizations**
- Redis caching for personality vectors
- Async MCP calls for non-blocking responses
- Connection pooling for databases
- Auto-scaling based on load

### **Monitoring**
- Response time tracking
- Memory usage analytics
- Error rate monitoring
- User satisfaction metrics

## 🔗 Integration Examples

### **Basic Memory Integration**
```python
from production.brain.mcp_memory_integration import OviyaMemorySystem

memory = OviyaMemorySystem()

# Store conversation
await memory.store_conversation_memory(user_id, {
    "user_input": "I'm feeling anxious",
    "response": "I hear your anxiety...",
    "personality_vector": {"Ma": 0.3, "Ahimsa": 0.4, ...}
})

# Retrieve context
memories = await memory.retrieve_relevant_memories(user_id, "anxiety")
```

### **Crisis Detection**
```python
from production.brain.crisis_detection import CrisisDetectionSystem

crisis_detector = CrisisDetectionSystem()
assessment = await crisis_detector.assess_crisis_risk(
    "I don't want to live anymore", conversation_history
)

if assessment["escalation_needed"]:
    # Immediate intervention
    resources = await crisis_detector.get_emergency_resources("us")
```

### **Empathic Response Generation**
```python
from production.brain.empathic_thinking import EmpathicThinkingEngine

empathy_engine = EmpathicThinkingEngine()
response = await empathy_engine.generate_empathic_response(
    user_input, personality_vector, emotion_context
)
```

## 🐛 Troubleshooting

### **Common Issues**

**MCP servers not starting:**
```bash
# Check logs
docker-compose logs mcp-server-name

# Restart services
./deploy.sh restart
```

**Memory not persisting:**
```bash
# Check database connections
docker-compose ps postgres qdrant

# Verify environment variables
cat .env
```

**High latency:**
```bash
# Check Redis cache
docker-compose exec redis redis-cli ping

# Monitor response times
curl http://localhost:3007/metrics
```

## 📚 API Documentation

Each MCP server exposes its capabilities through the MCP protocol:

- **OpenMemory:** Memory storage and retrieval
- **AI Therapist:** Crisis assessment and resources
- **MCP Thinking:** Cognitive empathy modes
- **Oviya Personality:** 5-pillar personality vectors
- **Oviya Emotion:** Voice prosody analysis

See individual server documentation for detailed API specs.

## 🤝 Contributing

### **Adding New MCP Servers**
1. Create server directory under appropriate tier
2. Implement MCP protocol handlers
3. Add to docker-compose.yml
4. Update configuration files
5. Add tests and documentation

### **Custom Oviya MCP Development**
```bash
# Create new custom server
mkdir servers/custom-oviya/new-feature
cd servers/custom-oviya/new-feature

# Implement server.py with MCP handlers
# Add Dockerfile
# Update docker-compose.yml
```

## 📄 License

This MCP ecosystem is part of the Oviya EI project. See main project license.

## 🙏 Acknowledgments

- **Anthropic** for the MCP protocol
- **Mem0.ai** for OpenMemory
- **Daniel Dunderfelt** for AI Therapist MCP
- **Vitaly Malakanov** for MCP Thinking
- **Chroma** for vector database
- **All MCP community contributors**

---

**This MCP ecosystem makes Oviya the most sophisticated emotional AI platform available, with clinical-grade safety, persistent memory, and enterprise scalability.** 🚀

Ready to deploy? Run `./deploy.sh` and transform your emotional AI companion into a category leader!
