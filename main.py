from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from db.supabase_client import SupabaseMemory
from agents.models import AgentRequest, AgentResponse
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_memory = SupabaseMemory()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True


@app.post("/api/mini-jarvis", response_model=AgentResponse)
async def mini_jarvis(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        is_new_session = await supabase_memory.start_session(request.session_id)
        # Fetch conversation history from the DB
        if not is_new_session:
            conversation_history = await supabase_memory.fetch_conversation_history(request.session_id)
        else:
            conversation_history = []
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]
            msg = {"role": msg_type, "content": msg_content}
            messages.append(msg)

        # Store user's query
        await supabase_memory.store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query
        )            

        
        agent_response = "This is a sample agent response..."

        # Store agent's response
        await supabase_memory.store_message(
            session_id=request.session_id,
            message_type="ai",
            content=agent_response,
            data={"request_id": request.request_id}
        )

        return AgentResponse(success=True)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Store error message in conversation
        await supabase_memory.store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False)

if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
