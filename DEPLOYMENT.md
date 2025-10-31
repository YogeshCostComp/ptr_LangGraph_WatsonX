# Deployment Guide

## Deploy to Render (Free)

### 1. Deploy API (Backend)

1. **Create Web Service on Render**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - Name: `ptr-langgraph-watsonx-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables**
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `TAVILY_API_KEY`: Your Tavily API key
   - `MODEL`: `anthropic/claude-sonnet-4-20250514`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (takes ~5 minutes)
   - Note your API URL: `https://ptr-langgraph-watsonx-api.onrender.com`

### 2. Deploy React UI (Frontend)

1. **Create Static Site on Render**
   - Click "New +" → "Static Site"
   - Connect same GitHub repository

2. **Configure Site**
   - Name: `ptr-langgraph-watsonx-ui`
   - Build Command: `cd chat-ui-react && npm install && npm run build`
   - Publish Directory: `chat-ui-react/dist`

3. **Update API URL**
   - Before deploying, update `chat-ui-react/src/App.jsx`
   - Change `http://localhost:8000` to your deployed API URL
   - Commit and push changes

4. **Deploy**
   - Click "Create Static Site"
   - Wait for deployment
   - Your app will be live at: `https://ptr-langgraph-watsonx-ui.onrender.com`

## Alternative: Deploy to Railway

### API Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### Environment Variables
Set in Railway dashboard:
- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY`
- `MODEL`

## Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   cd chat-ui-react && npm install
   ```

2. **Set Environment Variables**
   - Copy `.env.example` to `.env`
   - Add your API keys

3. **Run Services**
   ```bash
   # Option 1: Use startup script
   .\start.ps1

   # Option 2: Manual start
   # Terminal 1 - API
   python run_api.py

   # Terminal 2 - UI
   cd chat-ui-react
   npm run dev
   ```

## API Endpoints

- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `GET /health` - Health check
- `POST /agent/invoke` - Direct agent invocation

## Environment Variables

- `ANTHROPIC_API_KEY` (required): Your Anthropic API key
- `TAVILY_API_KEY` (required): Your Tavily API key for web search
- `MODEL` (optional): Model to use, default: `anthropic/claude-sonnet-4-20250514`
- `API_HOST` (optional): Host to bind, default: `0.0.0.0`
- `API_PORT` (optional): Port to bind, default: `8000`
