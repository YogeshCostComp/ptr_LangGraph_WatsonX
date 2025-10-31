"""
Run the LangGraph ReAct Agent API Server
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🚀 Starting LangGraph ReAct Agent API Server")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📚 API documentation at: http://{host}:{port}/docs")
    print(f"🔗 Chat endpoint: http://{host}:{port}/v1/chat/completions")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
