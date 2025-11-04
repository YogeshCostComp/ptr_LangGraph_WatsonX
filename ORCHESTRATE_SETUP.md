# watsonx Orchestrate Integration Setup

## Overview
This guide helps you integrate your LangGraph agent with watsonx Orchestrate as an external agent.

## 1. Set API Key in Render

Your API already has the `/v1/chat/completions` endpoint with SSE streaming support. Now you need to configure the API key:

### Steps:
1. **Go to Render Dashboard**: https://dashboard.render.com/
2. **Select your service**: `ptr-langgraph-watsonx-api`
3. **Go to Environment tab**
4. **Add new environment variable**:
   - **Key**: `ORCHESTRATE_API_KEY`
   - **Value**: Generate a secure API key (e.g., use a password generator or run this command):
     ```bash
     python -c "import secrets; print(secrets.token_urlsafe(32))"
     ```
   - Example value: `xK9mP2nQ5rT8wV1yC4fH7jL0dG3bN6sM8pR4tW9zA2`

5. **Save Changes** - Render will automatically redeploy your service

### Copy the API Key
Once you set the API key in Render, **copy it** - you'll need it for the Orchestrate configuration.

---

## 2. Test the Endpoint

After the Render deployment completes, test the streaming endpoint:

### Using curl (PowerShell):
```powershell
$headers = @{
    "Content-Type" = "application/json"
    "x-api-key" = "YOUR_API_KEY_HERE"
}

$body = @{
    messages = @(
        @{
            role = "user"
            content = "What is the weather in San Francisco?"
        }
    )
    stream = $true
    model = "anthropic/claude-sonnet-4-20250514"
} | ConvertTo-Json

Invoke-WebRequest -Uri "https://ptr-langgraph-watsonx-api.onrender.com/v1/chat/completions" `
    -Method POST `
    -Headers $headers `
    -Body $body
```

### Using Python:
```python
import requests
import json

url = "https://ptr-langgraph-watsonx-api.onrender.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "YOUR_API_KEY_HERE"
}

data = {
    "messages": [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "stream": True,
    "model": "anthropic/claude-sonnet-4-20250514"
}

response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

---

## 3. Install watsonx Orchestrate ADK

```bash
pip install ibm-watsonx-orchestrate
```

### Configure ADK Connection:
```bash
orchestrate configure
```

Follow the prompts to connect to your watsonx Orchestrate environment.

---

## 4. Update Agent Configuration

Edit `orchestrate_agent_config.json` and replace `${ORCHESTRATE_API_KEY}` with your actual API key from Render:

```json
{
  "api_url": "https://ptr-langgraph-watsonx-api.onrender.com/v1/chat/completions",
  "auth_config": {
    "header_name": "x-api-key",
    "api_key": "xK9mP2nQ5rT8wV1yC4fH7jL0dG3bN6sM8pR4tW9zA2"
  }
}
```

---

## 5. Import Agent to Orchestrate

```bash
orchestrate agents import -f orchestrate_agent_config.json
```

This will register your external agent in watsonx Orchestrate.

---

## 6. Create Native Agent with Collaborator

External agents can't be used directly - they must be attached to a native agent:

```bash
orchestrate agents create \
  --name langgraph_host_agent \
  --kind native \
  --description "Native agent that routes queries to the LangGraph external agent" \
  --llm watsonx/ibm/granite-3-8b-instruct \
  --style default \
  --collaborators langgraph_watsonx_agent \
  --output "langgraph_host_agent.yaml"
```

Then import the native agent:

```bash
orchestrate agents import -f langgraph_host_agent.yaml
```

---

## 7. Test in Orchestrate

### Option A: Test in Agent Builder UI
1. Log in to watsonx Orchestrate
2. Open Agent Builder
3. Find your native agent (`langgraph_host_agent`)
4. Use the test chat to verify it calls your external agent

### Option B: Test with ADK CLI (requires Developer Edition)
```bash
orchestrate chat start
```

---

## 8. Validate and Package (Optional - for Partner Onboarding)

### Create TSV test file:
```tsv
What is the weather in San Francisco?	[Expected response about weather]
```

### Run validation:
```bash
orchestrate evaluations validate-external \
  --tsv ./evaluations/test.tsv \
  --external-agent-config ./orchestrate_agent_config.json \
  --credential "YOUR_API_KEY_HERE"
```

### Package the offering:
```bash
orchestrate partners offering create \
  --offering langgraph_watsonx \
  --publisher Capgemini \
  --type external \
  --agent-name langgraph_watsonx_agent

orchestrate partners offering package \
  --offering langgraph_watsonx \
  --folder .
```

---

## API Endpoint Details

### Endpoint: `/v1/chat/completions`
- **Method**: POST
- **Authentication**: x-api-key header
- **Content-Type**: application/json

### Request Format:
```json
{
  "messages": [
    {"role": "user", "content": "Your question here"}
  ],
  "stream": true,
  "model": "anthropic/claude-sonnet-4-20250514",
  "temperature": 0.7,
  "max_tokens": 2000
}
```

### Response Format (Streaming SSE):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730764800,"model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730764800,"model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730764800,"model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"content":"world!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730764800,"model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Troubleshooting

### Issue: 401 Unauthorized
- **Cause**: Missing or incorrect x-api-key header
- **Fix**: Ensure ORCHESTRATE_API_KEY is set in Render and matches your config

### Issue: 400 Bad Request
- **Cause**: stream parameter not set to true
- **Fix**: watsonx Orchestrate always requires `"stream": true`

### Issue: Connection timeout
- **Cause**: Render service may be sleeping (free tier)
- **Fix**: Make a test request to wake it up, or upgrade to paid tier

### Issue: Agent not responding
- **Cause**: Native agent not configured to call collaborator
- **Fix**: Check native agent instructions include when to call the external agent

---

## Environment Variables Summary

### Required in Render:
- `ORCHESTRATE_API_KEY` - API key for x-api-key authentication
- `ANTHROPIC_API_KEY` - Your Anthropic API key (already set)
- `TAVILY_API_KEY` - Your Tavily search API key (already set)
- `WATSONX_PROJECT_ID` - watsonx project ID (already set)
- `WATSONX_APIKEY` - watsonx API key (already set)

---

## Next Steps

1. ✅ Set `ORCHESTRATE_API_KEY` in Render
2. ✅ Test the `/v1/chat/completions` endpoint
3. ⬜ Install watsonx Orchestrate ADK
4. ⬜ Import agent to Orchestrate
5. ⬜ Create native agent with collaborator
6. ⬜ Test in Orchestrate Agent Builder
7. ⬜ (Optional) Package for partner submission
