"""FastAPI wrapper for the LangGraph ReAct Agent.

This module provides REST API endpoints to interact with the agent,
including a chat completion endpoint compatible with watsonx Orchestrate.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from react_agent.context import Context
from react_agent.graph import graph


# Pydantic models for API requests and responses
class Message(BaseModel):
    """A chat message."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion endpoint."""
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(default="anthropic/claude-sonnet-4-20250514", description="Model to use")
    max_tokens: Optional[int] = Field(default=1024, description="Maximum tokens in response")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion endpoint."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used")
    choices: List[dict] = Field(..., description="List of completion choices")
    usage: dict = Field(..., description="Token usage information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


class AgentRequest(BaseModel):
    """Request model for direct agent invocation."""
    query: str = Field(..., description="User query to send to the agent")
    context_override: Optional[dict] = Field(default=None, description="Optional context overrides")


class AgentResponse(BaseModel):
    """Response model for direct agent invocation."""
    response: str = Field(..., description="Agent's response")
    messages: List[dict] = Field(..., description="Full message history")
    metadata: dict = Field(..., description="Additional metadata")


# Initialize FastAPI app
app = FastAPI(
    title="LangGraph ReAct Agent API",
    description="REST API for interacting with the LangGraph ReAct Agent",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Chat completion endpoint compatible with watsonx Orchestrate and OpenAI API format.
    
    This endpoint accepts chat messages and returns agent responses in a format
    compatible with standard chat completion APIs.
    """
    try:
        # Convert request messages to LangChain format
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        langchain_messages = []
        for msg in request.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
        
        # Prepare input state
        input_state = {
            "messages": langchain_messages
        }
        
        # Create context
        context = Context(
            model=request.model or "anthropic/claude-sonnet-4-20250514",
            max_search_results=10
        )
        
        # Invoke the graph with context parameter
        result = await graph.ainvoke(input_state, context=context)
        
        # Extract the final response
        final_messages = result.get("messages", [])
        if not final_messages:
            raise HTTPException(status_code=500, detail="No response from agent")
        
        # Get the last assistant message
        last_message = final_messages[-1]
        response_content = ""
        
        if hasattr(last_message, 'content'):
            response_content = last_message.content
        elif isinstance(last_message, dict):
            response_content = last_message.get('content', '')
        
        # Format response in OpenAI-compatible format
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Not calculated in this implementation
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """Direct agent invocation endpoint.
    
    This endpoint provides a simpler interface for invoking the agent
    with a single query.
    """
    try:
        # Create context
        context_dict = request.context_override or {}
        context = Context(**context_dict)
        
        # Prepare input state
        input_state = {
            "messages": [{"role": "user", "content": request.query}]
        }
        
        # Invoke the graph
        result = await graph.ainvoke(
            input_state,
            config={"configurable": context.__dict__}
        )
        
        # Extract response
        final_messages = result.get("messages", [])
        if not final_messages:
            raise HTTPException(status_code=500, detail="No response from agent")
        
        last_message = final_messages[-1]
        response_content = ""
        
        if hasattr(last_message, 'content'):
            response_content = last_message.content
        elif isinstance(last_message, dict):
            response_content = last_message.get('content', '')
        
        # Convert messages to serializable format
        serializable_messages = []
        for msg in final_messages:
            if hasattr(msg, 'content'):
                serializable_messages.append({
                    "role": getattr(msg, 'type', 'unknown'),
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                serializable_messages.append(msg)
        
        return {
            "response": response_content,
            "messages": serializable_messages,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "message_count": len(final_messages)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")


@app.post("/agent/stream")
async def stream_agent(request: AgentRequest):
    """Streaming agent invocation endpoint.
    
    This endpoint streams agent responses for real-time interaction.
    Note: Streaming implementation requires additional setup.
    """
    return {
        "error": "Streaming not yet implemented",
        "message": "Use /agent/invoke or /v1/chat/completions for now"
    }


class WatsonXToolRequest(BaseModel):
    """Request model for watsonx tool integration."""
    input: str = Field(..., description="The user's query or question", example="What is the weather in San Francisco?")
    
    class Config:
        schema_extra = {
            "example": {
                "input": "What is the current weather in San Francisco?"
            }
        }


class WatsonXToolResponse(BaseModel):
    """Response model for watsonx tool integration."""
    output: str = Field(..., description="The agent's response to the query")
    success: bool = Field(default=True, description="Whether the request was successful")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata about the response")
    
    class Config:
        schema_extra = {
            "example": {
                "output": "According to current weather data, San Francisco is experiencing partly cloudy conditions with a temperature of 65°F (18°C).",
                "success": True,
                "metadata": {
                    "timestamp": "2025-10-31T12:00:00Z",
                    "tools_used": ["tavily_search"],
                    "model": "claude-sonnet-4-20250514"
                }
            }
        }


@app.post("/watsonx/tool", response_model=WatsonXToolResponse, 
          summary="watsonx Tool Endpoint",
          description="Endpoint designed for IBM watsonx Orchestrate tool integration. Accepts a user query and returns an AI-generated response using web search capabilities.")
async def watsonx_tool(request: WatsonXToolRequest):
    """watsonx Orchestrate compatible tool endpoint.
    
    This endpoint is specifically designed for integration with IBM watsonx Orchestrate.
    It accepts a simple text input and returns a formatted response.
    
    **Input JSON Schema:**
    ```json
    {
        "input": "Your question here"
    }
    ```
    
    **Output JSON Schema:**
    ```json
    {
        "output": "AI response here",
        "success": true,
        "metadata": {
            "timestamp": "ISO timestamp",
            "tools_used": ["list of tools"],
            "model": "model name"
        }
    }
    ```
    
    **Example Usage:**
    ```bash
    curl -X POST "https://ptr-langgraph-watsonx-api.onrender.com/watsonx/tool" \\
         -H "Content-Type: application/json" \\
         -d '{"input": "What is the weather in San Francisco?"}'
    ```
    """
    try:
        from langchain_core.messages import HumanMessage
        
        # Create context
        context = Context(
            model="anthropic/claude-sonnet-4-20250514",
            max_search_results=10
        )
        
        # Prepare input state
        input_state = {
            "messages": [HumanMessage(content=request.input)]
        }
        
        # Invoke the graph with context parameter
        result = await graph.ainvoke(input_state, context=context)
        
        # Extract the final response
        final_messages = result.get("messages", [])
        if not final_messages:
            return WatsonXToolResponse(
                output="I apologize, but I encountered an issue processing your request.",
                success=False,
                metadata={"error": "No response from agent"}
            )
        
        # Get the last assistant message
        last_message = final_messages[-1]
        response_content = ""
        
        if hasattr(last_message, 'content'):
            response_content = last_message.content
        elif isinstance(last_message, dict):
            response_content = last_message.get('content', '')
        
        # Detect which tools were used
        tools_used = []
        for msg in final_messages:
            if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                for tool_call in msg.additional_kwargs['tool_calls']:
                    tool_name = tool_call.get('function', {}).get('name', 'unknown')
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
        
        return WatsonXToolResponse(
            output=response_content,
            success=True,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "tools_used": tools_used if tools_used else ["direct_response"],
                "model": "claude-sonnet-4-20250514",
                "message_count": len(final_messages)
            }
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}"
        print(f"watsonx tool error: {error_detail}\n{traceback.format_exc()}")
        
        return WatsonXToolResponse(
            output=f"I apologize, but I encountered an error processing your request: {str(e)}",
            success=False,
            metadata={"error": error_detail, "timestamp": datetime.utcnow().isoformat()}
        )


@app.get("/watsonx/schema")
async def watsonx_schema():
    """Get the JSON schema for the watsonx tool endpoint.
    
    This endpoint returns the OpenAPI schema information that can be used
    to configure the tool in watsonx Orchestrate.
    """
    return {
        "tool_name": "ptr-langgraph-watsonx",
        "description": "A LangGraph ReAct agent powered by Claude Sonnet that can search the web using Tavily and provide intelligent responses to user queries.",
        "endpoint": "/watsonx/tool",
        "method": "POST",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The user's query or question",
                    "example": "What is the current weather in San Francisco?"
                }
            },
            "required": ["input"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The agent's response to the query"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the request was successful"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata about the response",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "tools_used": {"type": "array", "items": {"type": "string"}},
                        "model": {"type": "string"}
                    }
                }
            },
            "required": ["output", "success"]
        },
        "example_request": {
            "input": "What is the weather in San Francisco?"
        },
        "example_response": {
            "output": "According to current weather data, San Francisco is experiencing partly cloudy conditions with a temperature of 65°F (18°C).",
            "success": True,
            "metadata": {
                "timestamp": "2025-10-31T12:00:00Z",
                "tools_used": ["tavily_search"],
                "model": "claude-sonnet-4-20250514"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
