import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '👋 Hello! I\'m your AI assistant powered by LangGraph. I can search the web using Tavily and help answer your questions. What would you like to know?' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    checkConnection()
    const interval = setInterval(checkConnection, 30000)
    return () => clearInterval(interval)
  }, [])

  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_URL}/health`)
      setIsConnected(response.ok)
    } catch (error) {
      setIsConnected(false)
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const conversationHistory = [...messages, userMessage]
      
      const response = await fetch(`${API_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: conversationHistory.map(m => ({ role: m.role, content: m.content })),
          model: 'anthropic/claude-sonnet-4-20250514',
          max_tokens: 2048,
          temperature: 0.7
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const assistantMessage = {
        role: 'assistant',
        content: data.choices[0].message.content
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error: ${error.message}`
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div>
          🤖 LangGraph ReAct Agent
          <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            ● {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <small>Powered by Claude & Tavily Search</small>
      </div>

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-content">
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-container" onSubmit={sendMessage}>
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message here..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="send-button"
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default App
