from fastapi import HTTPException
from supabase import create_client, Client
import os
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class SupabaseMemory:
    def __init__(self):
        self.client = supabase

    async def start_session(self, session_id: str):
        """Start a new session."""
        try:
            # Check if session already exists
            response = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
            if response.data:
                return False
            
            # Start new session
            supabase.table("sessions").insert({
                "session_id": session_id
            }).execute()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


    async def fetch_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch the most recent conversation history for a session."""
        try:
            response = supabase.table("messages") \
                .select("*") \
                .eq("session_id", session_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            # Convert to list and reverse to get chronological order
            messages = response.data[::-1]
            return messages
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

    async def store_message(self, session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
        """Store a message in the Supabase messages table."""
        message_obj = {
            "type": message_type,
            "content": content
        }
        if data:
            message_obj["data"] = data

        try:
            supabase.table("messages").insert({
                "session_id": session_id,
                "message": message_obj
            }).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")
            
    async def store_plan(self, query: str, workflow: Dict, results: Dict):
        """Store execution plan with embedding in Supabase"""
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Store in Supabase
            self.supabase.table("plans").insert({
                "query": query,
                "query_embedding": query_embedding,
                "workflow": json.dumps(workflow),
                "results": json.dumps(results),
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing plan: {str(e)}")
            
    async def find_similar_plans(self, query: str, similarity_threshold: float = 0.8, limit: int = 5):
        """Find similar plans using embedding-based similarity search"""
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Search for similar plans using cosine similarity
            response = self.supabase.rpc(
                'match_plans_by_embedding',
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': similarity_threshold,
                    'match_limit': limit
                }
            ).execute()
            
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error finding similar plans: {str(e)}")