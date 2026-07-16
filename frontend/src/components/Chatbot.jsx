import { useState, useEffect, useRef, useCallback } from 'react';
import {
  getSessionId,
  fetchConversations,
  createConversation,
  deleteConversation,
  fetchChatMessages,
  sendChatMessage,
} from '../api';

const SUGGESTED_QUESTIONS = [
  'What is the normal Flash Point spec range?',
  'Explain the model accuracy and metrics',
  'What is the 15-minute process lag?',
  'How does stripping steam affect Flash Point?',
];

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState('');
  const [showConvList, setShowConvList] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const sessionId = useRef(getSessionId());

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && !showConvList) {
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen, showConvList]);

  // Load conversations when chat opens
  const selectConversation = useCallback(async (convId) => {
    setActiveConvId(convId);
    setShowConvList(false);
    setError('');
    try {
      const res = await fetchChatMessages(convId);
      setMessages(
        (res.messages || []).map((m) => ({
          role: m.role,
          content: m.content,
          timestamp: m.created_at,
        }))
      );
    } catch (e) {
      console.error('Failed to load messages:', e);
      setMessages([]);
    }
  }, []);

  const loadConversations = useCallback(async () => {
    try {
      const res = await fetchConversations(sessionId.current);
      const convs = res.conversations || [];
      setConversations(convs);
      if (convs.length > 0 && !activeConvId) {
        selectConversation(convs[0].id);
      }
    } catch (e) {
      console.error('Failed to load conversations:', e);
      setError('Failed to load conversations.');
    }
  }, [activeConvId, selectConversation]);

  useEffect(() => {
    if (!isOpen) return;
    loadConversations();
  }, [isOpen, loadConversations]);

  const handleNewConversation = async () => {
    setIsInitializing(true);
    try {
      const res = await createConversation(sessionId.current);
      await loadConversations();
      selectConversation(res.conversation_id);
    } catch (e) {
      console.error('Failed to create conversation:', e);
      setError('Failed to create conversation.');
    } finally {
      setIsInitializing(false);
    }
  };

  const handleDeleteConversation = async (convId, e) => {
    e.stopPropagation();
    try {
      await deleteConversation(convId);
      if (activeConvId === convId) {
        setActiveConvId(null);
        setMessages([]);
      }
      await loadConversations();
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      setError('Failed to delete conversation.');
    }
  };

  const handleSend = async (text) => {
    const msg = (text || input).trim();
    if (!msg || isStreaming) return;

    // Auto-create conversation if none exists
    let convId = activeConvId;
    if (!convId) {
      try {
        const res = await createConversation(sessionId.current);
        convId = res.conversation_id;
        setActiveConvId(convId);
        await loadConversations();
      } catch {
        setError('Failed to start a conversation.');
        return;
      }
    }

    setInput('');
    setError('');
    setIsStreaming(true);

    // Add user message
    const userMsg = { role: 'user', content: msg, timestamp: new Date().toISOString() };
    const assistantMsg = { role: 'assistant', content: '', timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    abortRef.current = sendChatMessage(
      convId,
      msg,
      (chunk) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, content: last.content + chunk };
          }
          return updated;
        });
      },
      () => setIsStreaming(false),
      (errMsg) => {
        setError(errMsg);
        setIsStreaming(false);
      }
    );
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleAbort = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      setIsStreaming(false);
    }
  };

  const formatTime = (ts) => {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  return (
    <>
      {/* ── Floating Action Button ── */}
      <button
        id="chatbot-toggle"
        className={`chatbot-fab ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* ── Chat Window ── */}
      {isOpen && (
        <div className="chatbot-window" role="dialog" aria-label="AI Assistant Chat">
          {/* Header */}
          <div className="chatbot-header">
            <div className="chatbot-header-left">
              <button
                className="chatbot-menu-btn"
                onClick={() => setShowConvList(!showConvList)}
                title="Conversations"
                aria-label={showConvList ? 'Close conversation list' : 'Open conversation list'}
              >
                ☰
              </button>
              <div>
                <div className="chatbot-header-title">🔥 HY Kero Assistant</div>
                <div className="chatbot-header-sub">CDU Process • ML Model AI</div>
              </div>
            </div>
            <button className="chatbot-new-btn" onClick={handleNewConversation} disabled={isInitializing} title="New chat" aria-label="Start new conversation">
              +
            </button>
          </div>

          {/* Conversations List Panel */}
          {showConvList && (
            <div className="chatbot-convlist">
              <div className="chatbot-convlist-title">Conversations</div>
              {conversations.length === 0 ? (
                <div className="chatbot-convlist-empty">No conversations yet</div>
              ) : (
                conversations.map((c) => (
                  <div
                    key={c.id}
                    className={`chatbot-conv-item ${activeConvId === c.id ? 'active' : ''}`}
                    onClick={() => selectConversation(c.id)}
                  >
                    <span className="chatbot-conv-title">{c.title}</span>
                    <button
                      className="chatbot-conv-delete"
                      onClick={(e) => handleDeleteConversation(c.id, e)}
                      title="Delete"
                      aria-label={`Delete conversation: ${c.title}`}
                    >
                      🗑
                    </button>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Messages Area */}
          <div className="chatbot-messages">
            {messages.length === 0 && !showConvList && (
              <div className="chatbot-welcome">
                <div className="chatbot-welcome-icon">🤖</div>
                <div className="chatbot-welcome-text">
                  Ask me anything about the HY Kero Flash Point model, CDU operations, or process data.
                </div>
                <div className="chatbot-suggestions">
                  {SUGGESTED_QUESTIONS.map((q, i) => (
                    <button key={i} className="chatbot-suggestion" onClick={() => handleSend(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={`${m.role}-${i}`} className={`chatbot-msg ${m.role}`}>
                <div className="chatbot-msg-avatar">{m.role === 'user' ? '👤' : '🤖'}</div>
                <div className="chatbot-msg-body">
                  <div className="chatbot-msg-content">{m.content}</div>
                  {/* Typing indicator for streaming assistant message */}
                  {isStreaming && i === messages.length - 1 && m.role === 'assistant' && (
                    <div className="chatbot-typing">
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </div>
                  )}
                  <div className="chatbot-msg-time">{formatTime(m.timestamp)}</div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Error Banner */}
          {error && (
            <div className="chatbot-error">
              ⚠️ {error}
              <button onClick={() => setError('')}>✕</button>
            </div>
          )}

          {/* Input Area */}
          <div className="chatbot-input-area">
            <textarea
              ref={inputRef}
              className="chatbot-input"
              rows={1}
              placeholder="Ask about Flash Point, CDU ops, ML model..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              maxLength={1000}
              aria-label="Type your message to the AI assistant"
            />
            {isStreaming ? (
              <button className="chatbot-send-btn stop" onClick={handleAbort} title="Stop generating" aria-label="Stop generating response">
                ■
              </button>
            ) : (
              <button
                className="chatbot-send-btn"
                onClick={() => handleSend()}
                disabled={!input.trim()}
                title="Send message"
              >
                ➤
              </button>
            )}
          </div>
        </div>
      )}
    </>
  );
}
