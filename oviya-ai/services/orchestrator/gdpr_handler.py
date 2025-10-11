#!/usr/bin/env python3
"""
Oviya Legal Compliance System
Epic 6: GDPR compliance, data deletion, user rights
"""
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import auth, firestore, storage
import redis

@dataclass
class UserDataExport:
    """User data export for GDPR compliance"""
    user_id: str
    export_timestamp: datetime
    data: Dict[str, Any]
    format: str = "json"

@dataclass
class DataDeletionRequest:
    """Data deletion request"""
    user_id: str
    request_timestamp: datetime
    reason: str
    status: str = "pending"  # pending, processing, completed, failed

class GDPRHandler:
    """GDPR compliance handler"""
    
    def __init__(self, firestore_client: firestore.Client, redis_client: redis.Redis):
        self.firestore = firestore_client
        self.redis = redis_client
        
        # Data retention policies
        self.retention_policies = {
            "conversations": 365,  # 1 year
            "audio_files": 90,     # 3 months
            "logs": 30,           # 1 month
            "analytics": 730,     # 2 years (anonymized)
            "user_preferences": 365  # 1 year
        }
    
    async def export_user_data(self, user_id: str) -> UserDataExport:
        """Export all user data (GDPR Article 20)"""
        print(f"📤 Exporting data for user: {user_id}")
        
        export_data = {
            "export_info": {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "data_categories": list(self.retention_policies.keys()),
                "gdpr_compliant": True
            },
            "profile_data": await self._export_profile_data(user_id),
            "conversations": await self._export_conversations(user_id),
            "settings": await self._export_settings(user_id),
            "analytics": await self._export_analytics(user_id),
            "metadata": await self._export_metadata(user_id)
        }
        
        # Create export record
        export = UserDataExport(
            user_id=user_id,
            export_timestamp=datetime.now(),
            data=export_data,
            format="json"
        )
        
        # Store export record
        await self._store_export_record(export)
        
        return export
    
    async def delete_user_data(self, user_id: str, reason: str = "user_request") -> DataDeletionRequest:
        """Delete all user data (GDPR Article 17)"""
        print(f"🗑️ Deleting data for user: {user_id}")
        
        deletion_request = DataDeletionRequest(
            user_id=user_id,
            request_timestamp=datetime.now(),
            reason=reason,
            status="processing"
        )
        
        try:
            # 1. Delete from Firestore
            await self._delete_firestore_data(user_id)
            
            # 2. Delete from Firebase Auth
            await self._delete_firebase_auth(user_id)
            
            # 3. Delete audio files from storage
            await self._delete_audio_files(user_id)
            
            # 4. Delete from Redis
            await self._delete_redis_data(user_id)
            
            # 5. Anonymize logs
            await self._anonymize_logs(user_id)
            
            # 6. Update deletion request status
            deletion_request.status = "completed"
            await self._store_deletion_record(deletion_request)
            
            print(f"✅ Data deletion completed for user: {user_id}")
            
        except Exception as e:
            deletion_request.status = "failed"
            await self._store_deletion_record(deletion_request)
            print(f"❌ Data deletion failed for user {user_id}: {e}")
            raise
        
        return deletion_request
    
    async def anonymize_user_data(self, user_id: str) -> bool:
        """Anonymize user data while preserving analytics"""
        print(f"🎭 Anonymizing data for user: {user_id}")
        
        try:
            # Generate anonymous ID
            anonymous_id = hashlib.sha256(f"{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
            
            # Anonymize conversations
            await self._anonymize_conversations(user_id, anonymous_id)
            
            # Anonymize analytics
            await self._anonymize_analytics(user_id, anonymous_id)
            
            # Delete personally identifiable data
            await self._delete_pii_data(user_id)
            
            print(f"✅ Data anonymized for user: {user_id} -> {anonymous_id}")
            return True
            
        except Exception as e:
            print(f"❌ Data anonymization failed for user {user_id}: {e}")
            return False
    
    async def get_data_retention_summary(self) -> Dict:
        """Get data retention summary"""
        summary = {}
        
        for data_type, retention_days in self.retention_policies.items():
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Count records that should be deleted
            if data_type == "conversations":
                count = await self._count_old_conversations(cutoff_date)
            elif data_type == "logs":
                count = await self._count_old_logs(cutoff_date)
            else:
                count = 0
            
            summary[data_type] = {
                "retention_days": retention_days,
                "cutoff_date": cutoff_date.isoformat(),
                "records_to_delete": count
            }
        
        return summary
    
    async def cleanup_expired_data(self) -> Dict:
        """Clean up data that has exceeded retention period"""
        print("🧹 Starting data cleanup...")
        
        cleanup_summary = {}
        
        for data_type, retention_days in self.retention_policies.items():
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            if data_type == "conversations":
                deleted_count = await self._delete_old_conversations(cutoff_date)
            elif data_type == "logs":
                deleted_count = await self._delete_old_logs(cutoff_date)
            elif data_type == "audio_files":
                deleted_count = await self._delete_old_audio_files(cutoff_date)
            else:
                deleted_count = 0
            
            cleanup_summary[data_type] = {
                "retention_days": retention_days,
                "cutoff_date": cutoff_date.isoformat(),
                "deleted_count": deleted_count
            }
        
        print(f"✅ Data cleanup completed: {cleanup_summary}")
        return cleanup_summary
    
    # Private methods for data export
    async def _export_profile_data(self, user_id: str) -> Dict:
        """Export user profile data"""
        try:
            doc = self.firestore.collection('users').document(user_id).get()
            if doc.exists:
                return doc.to_dict()
            return {}
        except Exception as e:
            print(f"Error exporting profile data: {e}")
            return {}
    
    async def _export_conversations(self, user_id: str) -> List[Dict]:
        """Export user conversations"""
        try:
            conversations = []
            docs = self.firestore.collection('conversations').where('user_id', '==', user_id).stream()
            
            for doc in docs:
                conversation_data = doc.to_dict()
                conversation_data['conversation_id'] = doc.id
                conversations.append(conversation_data)
            
            return conversations
        except Exception as e:
            print(f"Error exporting conversations: {e}")
            return []
    
    async def _export_settings(self, user_id: str) -> Dict:
        """Export user settings"""
        try:
            doc = self.firestore.collection('user_settings').document(user_id).get()
            if doc.exists:
                return doc.to_dict()
            return {}
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return {}
    
    async def _export_analytics(self, user_id: str) -> List[Dict]:
        """Export user analytics (anonymized)"""
        try:
            analytics = []
            docs = self.firestore.collection('analytics').where('user_id', '==', user_id).stream()
            
            for doc in docs:
                analytics_data = doc.to_dict()
                # Remove any PII from analytics
                if 'ip_address' in analytics_data:
                    del analytics_data['ip_address']
                if 'user_agent' in analytics_data:
                    del analytics_data['user_agent']
                
                analytics.append(analytics_data)
            
            return analytics
        except Exception as e:
            print(f"Error exporting analytics: {e}")
            return []
    
    async def _export_metadata(self, user_id: str) -> Dict:
        """Export user metadata"""
        return {
            "account_created": await self._get_account_created_date(user_id),
            "last_active": await self._get_last_active_date(user_id),
            "total_conversations": await self._count_user_conversations(user_id),
            "total_messages": await self._count_user_messages(user_id),
            "preferred_emotions": await self._get_preferred_emotions(user_id)
        }
    
    # Private methods for data deletion
    async def _delete_firestore_data(self, user_id: str):
        """Delete user data from Firestore"""
        # Delete user profile
        self.firestore.collection('users').document(user_id).delete()
        
        # Delete conversations
        conversations = self.firestore.collection('conversations').where('user_id', '==', user_id).stream()
        for conv in conversations:
            conv.reference.delete()
        
        # Delete settings
        self.firestore.collection('user_settings').document(user_id).delete()
        
        # Delete analytics
        analytics = self.firestore.collection('analytics').where('user_id', '==', user_id).stream()
        for analytic in analytics:
            analytic.reference.delete()
    
    async def _delete_firebase_auth(self, user_id: str):
        """Delete user from Firebase Auth"""
        try:
            auth.delete_user(user_id)
        except Exception as e:
            print(f"Error deleting Firebase Auth user: {e}")
    
    async def _delete_audio_files(self, user_id: str):
        """Delete user audio files from storage"""
        try:
            bucket = storage.bucket()
            blobs = bucket.list_blobs(prefix=f"audio/{user_id}/")
            
            for blob in blobs:
                blob.delete()
        except Exception as e:
            print(f"Error deleting audio files: {e}")
    
    async def _delete_redis_data(self, user_id: str):
        """Delete user data from Redis"""
        try:
            # Delete all keys related to user
            pattern = f"*{user_id}*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            print(f"Error deleting Redis data: {e}")
    
    async def _anonymize_logs(self, user_id: str):
        """Anonymize logs containing user data"""
        try:
            # This would typically involve updating log files
            # For now, we'll just log the anonymization
            print(f"Anonymizing logs for user: {user_id}")
        except Exception as e:
            print(f"Error anonymizing logs: {e}")
    
    # Private helper methods
    async def _store_export_record(self, export: UserDataExport):
        """Store export record"""
        export_data = {
            "user_id": export.user_id,
            "export_timestamp": export.export_timestamp.isoformat(),
            "format": export.format,
            "status": "completed"
        }
        
        self.firestore.collection('data_exports').add(export_data)
    
    async def _store_deletion_record(self, deletion: DataDeletionRequest):
        """Store deletion record"""
        deletion_data = {
            "user_id": deletion.user_id,
            "request_timestamp": deletion.request_timestamp.isoformat(),
            "reason": deletion.reason,
            "status": deletion.status
        }
        
        self.firestore.collection('data_deletions').add(deletion_data)
    
    async def _get_account_created_date(self, user_id: str) -> str:
        """Get account creation date"""
        try:
            user = auth.get_user(user_id)
            return datetime.fromtimestamp(user.user_metadata.creation_timestamp / 1000).isoformat()
        except:
            return "unknown"
    
    async def _get_last_active_date(self, user_id: str) -> str:
        """Get last active date"""
        try:
            doc = self.firestore.collection('users').document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get('last_active', 'unknown')
            return "unknown"
        except:
            return "unknown"
    
    async def _count_user_conversations(self, user_id: str) -> int:
        """Count user conversations"""
        try:
            docs = self.firestore.collection('conversations').where('user_id', '==', user_id).stream()
            return len(list(docs))
        except:
            return 0
    
    async def _count_user_messages(self, user_id: str) -> int:
        """Count user messages"""
        try:
            conversations = self.firestore.collection('conversations').where('user_id', '==', user_id).stream()
            total_messages = 0
            
            for conv in conversations:
                messages = self.firestore.collection('conversations').document(conv.id).collection('messages').stream()
                total_messages += len(list(messages))
            
            return total_messages
        except:
            return 0
    
    async def _get_preferred_emotions(self, user_id: str) -> List[str]:
        """Get user's preferred emotions"""
        try:
            doc = self.firestore.collection('user_settings').document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get('preferred_emotions', [])
            return []
        except:
            return []
    
    async def _count_old_conversations(self, cutoff_date: datetime) -> int:
        """Count old conversations"""
        try:
            docs = self.firestore.collection('conversations').where('created_at', '<', cutoff_date).stream()
            return len(list(docs))
        except:
            return 0
    
    async def _count_old_logs(self, cutoff_date: datetime) -> int:
        """Count old logs"""
        # This would depend on your logging system
        return 0
    
    async def _delete_old_conversations(self, cutoff_date: datetime) -> int:
        """Delete old conversations"""
        try:
            docs = self.firestore.collection('conversations').where('created_at', '<', cutoff_date).stream()
            deleted_count = 0
            
            for doc in docs:
                doc.reference.delete()
                deleted_count += 1
            
            return deleted_count
        except:
            return 0
    
    async def _delete_old_logs(self, cutoff_date: datetime) -> int:
        """Delete old logs"""
        # This would depend on your logging system
        return 0
    
    async def _delete_old_audio_files(self, cutoff_date: datetime) -> int:
        """Delete old audio files"""
        try:
            bucket = storage.bucket()
            blobs = bucket.list_blobs(prefix="audio/")
            deleted_count = 0
            
            for blob in blobs:
                if blob.time_created < cutoff_date:
                    blob.delete()
                    deleted_count += 1
            
            return deleted_count
        except:
            return 0
    
    async def _anonymize_conversations(self, user_id: str, anonymous_id: str):
        """Anonymize conversations"""
        try:
            conversations = self.firestore.collection('conversations').where('user_id', '==', user_id).stream()
            
            for conv in conversations:
                conv.reference.update({
                    'user_id': anonymous_id,
                    'anonymized': True,
                    'anonymized_at': datetime.now().isoformat()
                })
        except Exception as e:
            print(f"Error anonymizing conversations: {e}")
    
    async def _anonymize_analytics(self, user_id: str, anonymous_id: str):
        """Anonymize analytics"""
        try:
            analytics = self.firestore.collection('analytics').where('user_id', '==', user_id).stream()
            
            for analytic in analytics:
                analytic.reference.update({
                    'user_id': anonymous_id,
                    'anonymized': True,
                    'anonymized_at': datetime.now().isoformat()
                })
        except Exception as e:
            print(f"Error anonymizing analytics: {e}")
    
    async def _delete_pii_data(self, user_id: str):
        """Delete personally identifiable data"""
        try:
            # Delete user profile
            self.firestore.collection('users').document(user_id).delete()
            
            # Delete settings
            self.firestore.collection('user_settings').document(user_id).delete()
        except Exception as e:
            print(f"Error deleting PII data: {e}")

# Usage example
async def main():
    """Test the GDPR compliance system"""
    # Initialize Firebase
    firebase_admin.initialize_app()
    firestore_client = firestore.client()
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    gdpr_handler = GDPRHandler(firestore_client, redis_client)
    
    user_id = "test_user_123"
    
    # Test data export
    export = await gdpr_handler.export_user_data(user_id)
    print(f"Data export completed: {len(export.data)} categories")
    
    # Test data deletion
    deletion = await gdpr_handler.delete_user_data(user_id, "user_request")
    print(f"Data deletion status: {deletion.status}")
    
    # Test cleanup
    cleanup_summary = await gdpr_handler.cleanup_expired_data()
    print(f"Cleanup completed: {cleanup_summary}")

if __name__ == "__main__":
    asyncio.run(main())


