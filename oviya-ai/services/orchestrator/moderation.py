#!/usr/bin/env python3
"""
Oviya Content Moderation System
Epic 6: Critical production safety component
"""
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ModerationAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    INTERVENE = "intervene"  # Show crisis resources
    FLAG = "flag"  # Log for review

@dataclass
class ModerationResult:
    action: ModerationAction
    reason: str
    confidence: float
    categories: List[str]
    crisis_resources: Optional[Dict] = None

class ContentModerator:
    """Main content moderation system"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.crisis_detector = CrisisDetector()
        self.pii_detector = PIIDetector()
        self.openai_moderator = OpenAIModerator(openai_api_key)
        
    async def moderate_input(self, text: str, user_id: str) -> ModerationResult:
        """Main moderation pipeline"""
        
        # 1. OpenAI Moderation API
        openai_result = await self.openai_moderator.check(text)
        if openai_result.flagged:
            return ModerationResult(
                action=ModerationAction.BLOCK,
                reason="Content policy violation",
                confidence=openai_result.confidence,
                categories=openai_result.categories
            )
        
        # 2. Crisis Detection
        crisis_result = self.crisis_detector.is_crisis(text)
        if crisis_result["is_crisis"]:
            return ModerationResult(
                action=ModerationAction.INTERVENE,
                reason="Crisis detected",
                confidence=crisis_result["confidence"],
                categories=["crisis"],
                crisis_resources=self.crisis_detector.get_resources()
            )
        
        # 3. PII Detection
        pii_result = self.pii_detector.scan(text)
        if pii_result["found"]:
            # Log PII detection but don't block
            print(f"⚠️ PII detected for user {user_id}: {pii_result['types']}")
            return ModerationResult(
                action=ModerationAction.FLAG,
                reason="PII detected",
                confidence=pii_result["confidence"],
                categories=pii_result["types"]
            )
        
        # 4. Custom Rules
        custom_result = self._check_custom_rules(text)
        if custom_result["violation"]:
            return ModerationResult(
                action=ModerationAction.BLOCK,
                reason=custom_result["reason"],
                confidence=custom_result["confidence"],
                categories=["custom_rule"]
            )
        
        return ModerationResult(
            action=ModerationAction.ALLOW,
            reason="Content approved",
            confidence=1.0,
            categories=[]
        )
    
    def _check_custom_rules(self, text: str) -> Dict:
        """Custom moderation rules"""
        text_lower = text.lower()
        
        # Spam patterns
        spam_patterns = [
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            r"@\w+",  # @mentions
            r"#\w+",  # Hashtags
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text):
                return {
                    "violation": True,
                    "reason": "Spam pattern detected",
                    "confidence": 0.8
                }
        
        # Excessive repetition
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:  # 30% repetition
                return {
                    "violation": True,
                    "reason": "Excessive repetition",
                    "confidence": 0.7
                }
        
        return {"violation": False}

class CrisisDetector:
    """Detects crisis situations and provides resources"""
    
    CRISIS_KEYWORDS = [
        "kill myself", "end my life", "suicide", "want to die",
        "hurt myself", "self harm", "cut myself", "overdose",
        "jump off", "hang myself", "shoot myself", "poison myself",
        "not worth living", "better off dead", "end it all",
        "final goodbye", "last message", "goodbye forever"
    ]
    
    CRISIS_PHRASES = [
        "i want to die", "i don't want to live", "life is not worth it",
        "everyone would be better off", "i'm a burden", "i can't go on",
        "i give up", "i'm done", "i can't take it anymore"
    ]
    
    def __init__(self):
        self.resources = {
            "US": {
                "name": "National Suicide Prevention Lifeline",
                "phone": "988",
                "text": "Text HOME to 741741",
                "website": "https://suicidepreventionlifeline.org"
            },
            "UK": {
                "name": "Samaritans",
                "phone": "116 123",
                "website": "https://www.samaritans.org"
            },
            "India": {
                "name": "AASRA",
                "phone": "91-9820466726",
                "website": "https://www.aasra.info"
            },
            "Global": {
                "name": "International Association for Suicide Prevention",
                "website": "https://www.iasp.info/resources/Crisis_Centres/"
            }
        }
    
    def is_crisis(self, text: str) -> Dict:
        """Detect crisis situations"""
        text_lower = text.lower()
        
        # Check for crisis keywords
        keyword_matches = sum(1 for keyword in self.CRISIS_KEYWORDS if keyword in text_lower)
        
        # Check for crisis phrases
        phrase_matches = sum(1 for phrase in self.CRISIS_PHRASES if phrase in text_lower)
        
        # Calculate confidence
        total_matches = keyword_matches + phrase_matches
        confidence = min(total_matches * 0.3, 1.0)
        
        return {
            "is_crisis": total_matches > 0,
            "confidence": confidence,
            "keyword_matches": keyword_matches,
            "phrase_matches": phrase_matches
        }
    
    def get_resources(self) -> Dict:
        """Get crisis resources"""
        return self.resources

class PIIDetector:
    """Detects Personally Identifiable Information"""
    
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
        "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "address": r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    }
    
    def scan(self, text: str) -> Dict:
        """Scan text for PII"""
        found_types = []
        confidence = 0.0
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found_types.append(pii_type)
                confidence += 0.2
        
        return {
            "found": len(found_types) > 0,
            "types": found_types,
            "confidence": min(confidence, 1.0)
        }

class OpenAIModerator:
    """OpenAI Moderation API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/moderations"
    
    async def check(self, text: str) -> Dict:
        """Check text with OpenAI Moderation API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "input": text,
                    "model": "text-moderation-latest"
                }
                
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        moderation = result["results"][0]
                        
                        return {
                            "flagged": moderation["flagged"],
                            "categories": moderation["categories"],
                            "category_scores": moderation["category_scores"],
                            "confidence": max(moderation["category_scores"].values()) if moderation["flagged"] else 0.0
                        }
                    else:
                        print(f"OpenAI Moderation API error: {response.status}")
                        return {"flagged": False, "confidence": 0.0}
        
        except Exception as e:
            print(f"Error calling OpenAI Moderation API: {e}")
            return {"flagged": False, "confidence": 0.0}

class CrisisInterventionUI:
    """UI components for crisis intervention"""
    
    @staticmethod
    def get_crisis_message(resources: Dict) -> str:
        """Generate crisis intervention message"""
        return f"""
🚨 I'm concerned about what you've shared. Your life has value, and there are people who want to help.

**Immediate Help:**
• Call {resources['US']['phone']} (US) or {resources['UK']['phone']} (UK)
• Text {resources['US']['text']} (US)
• Visit {resources['US']['website']}

**You're not alone.** These feelings are temporary, even when they don't feel that way. Please reach out to someone who can help.

Would you like me to help you find local resources or just listen?
        """.strip()
    
    @staticmethod
    def get_blocked_message(reason: str) -> str:
        """Generate content blocked message"""
        return f"""
❌ I can't respond to that message. Let's talk about something else.

If you're going through a difficult time, I'm here to listen and support you in a healthy way.

What's on your mind today?
        """.strip()

# Usage example
async def main():
    """Test the moderation system"""
    moderator = ContentModerator("your-openai-api-key")
    
    test_cases = [
        "Hello, how are you today?",
        "I want to kill myself",
        "My email is john@example.com",
        "Visit my website at https://spam.com",
        "I'm feeling really sad and hopeless"
    ]
    
    for text in test_cases:
        result = await moderator.moderate_input(text, "test_user")
        print(f"Text: {text}")
        print(f"Action: {result.action.value}")
        print(f"Reason: {result.reason}")
        print(f"Confidence: {result.confidence}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main())


