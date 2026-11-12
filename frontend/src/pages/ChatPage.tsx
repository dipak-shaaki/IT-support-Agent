import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import {
  ArrowUp,
  ArrowUpRight,
  Check,
  ChevronDown,
  FileText,
  Globe2,
  LoaderCircle,
  WandSparkles,
} from 'lucide-react'
import { getConversation, streamChat } from '../api/client'
import { clearConversationId, readConversationId, saveConversationId } from '../api/browserStorage'
import { Markdown } from '../components/Markdown'
import { Sidebar } from '../components/Sidebar'
import type { ChatEvent, ConversationMessage } from '../types'

const SUGGESTIONS = [
  { icon: '⌁', text: 'I can’t connect to the company VPN' },
  { icon: '↻', text: 'Help me reset my password' },
  { icon: '◉', text: 'My MFA code isn’t working' },
  { icon: '⌕', text: 'Is Microsoft Teams having an outage?' },
]

const SOURCE_LABELS: Record<string, string> = {
  private_kb: 'Company knowledge base',
  web_search: 'Public web search',
  direct: 'Direct response',
  insufficient_evidence: 'Insufficient evidence',
}

export function ChatPage() {
  const [conversationId, setConversationId] = useState(
    readConversationId,
  )
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [activity, setActivity] = useState<string[]>([])
  const [source, setSource] = useState('')
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [restoring, setRestoring] = useState(Boolean(conversationId))
  const [historyError, setHistoryError] = useState('')
  const endOfMessages = useRef<HTMLDivElement>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const savedId = readConversationId()
    if (!savedId) return
    let active = true
    getConversation(savedId)
      .then((response) => {
        if (active) setMessages(response.messages)
      })
      .catch((error: unknown) => {
        if (active) {
          setHistoryError(
            error instanceof Error ? error.message : 'Conversation could not be restored.',
          )
        }
      })
      .finally(() => {
        if (active) setRestoring(false)
      })
    return () => { active = false }
  }, [])

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    endOfMessages.current?.scrollIntoView({
      behavior: reducedMotion ? 'auto' : 'smooth',
      block: 'end',
    })
  }, [messages])

  function startNewChat() {
    if (busy || restoring) return
    clearConversationId()
    setConversationId(null)
    setMessages([])
    setActivity([])
    setSource('')
    setHistoryError('')
    setQuestion('')
    textarea.current?.focus()
  }

  function addActivity(text: string) {
    setActivity((items) => [...items, text])
  }

  async function sendQuestion(text: string) {
    const trimmed = text.trim()
    if (!trimmed || busy || restoring) return

    setQuestion('')
    if (textarea.current) textarea.current.style.height = 'auto'
    setHistoryError('')
    setSource('Working')
    setActivity([])
    setBusy(true)
    setMessages((items) => [
      ...items,
      { role: 'user', content: trimmed },
      { role: 'assistant', content: '', pending: true },
    ])

    let answer = ''
    let completed = false
    try {
      for await (const event of streamChat(trimmed, conversationId)) {
        handleEvent(event, {
          onToken: (token) => {
            answer += token
            setMessages((items) => {
              const next = [...items]
              const last = next[next.length - 1]
              if (last?.role === 'assistant') {
                next[next.length - 1] = { ...last, content: answer }
              }
              return next
            })
          },
          onTrace: addActivity,
          onDone: (eventDone) => {
            completed = true
            if (saveConversationId(eventDone.conversation_id)) {
              setConversationId(eventDone.conversation_id)
            }
            setSource(eventDone.answer
              ? SOURCE_LABELS[eventDone.source_used] ?? eventDone.source_used
              : '')
            setMessages((items) => {
              const next = [...items]
              const last = next[next.length - 1]
              if (last?.role === 'assistant') {
                if (!eventDone.answer) return next.slice(0, -1)
                next[next.length - 1] = {
                  ...last,
                  content: eventDone.answer,
                  source: eventDone.source_used,
                  citations: eventDone.citations,
                  pending: false,
                }
              }
              return next
            })
          },
        })
      }
      if (!completed) throw new Error('The response stream ended before an answer arrived.')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Something went wrong.'
      setSource('Could not complete')
      setMessages((items) => {
        const next = [...items]
        const last = next[next.length - 1]
        if (last?.role === 'assistant') {
          next[next.length - 1] = {
            ...last,
            content: `I couldn’t complete that request. ${message}`,
            pending: false,
          }
        }
        return next
      })
      addActivity('Request failed')
    } finally {
      setBusy(false)
      textarea.current?.focus()
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void sendQuestion(question)
  }

  function handleInput(event: ChangeEvent<HTMLTextAreaElement>) {
    setQuestion(event.target.value)
    event.target.style.height = 'auto'
    event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`
  }

  return (
    <div className="app-shell">
      <Sidebar active="chat" />
      <main className="main-panel">
        <header className="topbar">
          <div className="page-title">IT support</div>
          <div className="topbar-actions">
            <button className="button button-outline topbar-new" type="button" onClick={startNewChat} disabled={busy || restoring}>
              New chat
            </button>
          </div>
        </header>

        <div className={`chat-workspace${messages.length === 0 ? ' chat-empty' : ''}`}>
          <div className="conversation">
            {messages.length === 0 ? (
              <section className="welcome-panel">
                <h1>What can we help with?</h1>
                <p className="welcome-description">
                  Tell us what’s not working. We’ll check your company’s support guides first.
                </p>
                <div className="suggestion-label">Common requests</div>
                <div className="suggestion-grid">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      className="suggestion-card"
                      type="button"
                      key={suggestion.text}
                      disabled={busy || restoring}
                      onClick={() => void sendQuestion(suggestion.text)}
                    >
                      <span className="suggestion-icon">{suggestion.icon}</span>
                      <span>{suggestion.text}</span>
                      <ArrowUpRight className="suggestion-arrow" size={15} />
                    </button>
                  ))}
                </div>
              </section>
            ) : (
              <section className="message-list" aria-live="polite" aria-label="Conversation">
                {restoring && <div className="restore-notice"><LoaderCircle className="spin" size={15} /> Restoring your conversation…</div>}
                {messages.map((message, index) => (
                  <article className={`message-row ${message.role}`} key={`${index}-${message.role}`}>
                    <div className="message-content">
                      {message.role === 'assistant' && <div className="message-author">Orbit</div>}
                      <div className={`message-bubble ${message.role}`}>
                        {message.pending && !message.content ? (
                          <div className="thinking-indicator"><span /><span /><span /><small>Thinking through this…</small></div>
                        ) : (
                          <Markdown>{message.content}</Markdown>
                        )}
                      </div>
                      {message.role === 'assistant' && !message.pending && (
                        <>
                          {message.source && (
                            <div className="answer-meta">
                              <span className={`source-pill ${message.source}`}>
                                {message.source === 'web_search' ? <Globe2 size={12} /> : <FileText size={12} />}
                                {SOURCE_LABELS[message.source] ?? message.source}
                              </span>
                            </div>
                          )}
                          {message.citations && message.citations.length > 0 && (
                            <div className="citation-list" aria-label="Sources">
                              {message.citations.map((citation, citationIndex) => (
                                citation.url ? (
                                  <a
                                    className="citation-chip"
                                    href={citation.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    key={`${citation.url}-${citationIndex}`}
                                  >
                                    <Globe2 size={12} /> {citation.title}
                                  </a>
                                ) : (
                                  <span className="citation-chip" key={`${citation.title}-${citationIndex}`}>
                                    <FileText size={12} /> {citation.title}
                                  </span>
                                )
                              ))}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </article>
                ))}
                {historyError && <output className="history-error" aria-live="polite">{historyError}</output>}
                <div ref={endOfMessages} />
              </section>
            )}
          </div>

          <div className="composer-wrap">
            {activity.length > 0 && (
              <details className="activity-panel">
                <summary><WandSparkles size={14} /> Agent activity <span className="activity-count">{activity.length}</span><ChevronDown size={14} /></summary>
                <div className="activity-content">
                  {activity.map((line, index) => <div className="activity-line" key={`${index}-${line}`}><Check size={13} />{line}</div>)}
                </div>
              </details>
            )}
            {historyError && messages.length === 0 && (
              <output className="history-error" aria-live="polite">{historyError}</output>
            )}
            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                ref={textarea}
                value={question}
                onChange={handleInput}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    event.currentTarget.form?.requestSubmit()
                  }
                }}
                rows={1}
                maxLength={12000}
                placeholder="Tell me what’s going on…"
                aria-label="Your IT support question"
                aria-describedby="question-limit"
                disabled={busy || restoring}
              />
              <div className="composer-bottom">
                <span className="composer-hint" id="question-limit">
                  Enter to send <span className="hint-divider">·</span> Shift + Enter for a line break
                </span>
                <button className="send-button" type="submit" disabled={!question.trim() || busy || restoring} aria-label="Send message">
                  {busy ? <LoaderCircle className="spin" size={18} /> : <ArrowUp size={19} />}
                </button>
              </div>
            </form>
            <div className="composer-disclaimer">
              <span>{source ? `Last response · ${source}` : 'Messages are limited to 400 tokens.'}</span>
              <span>Confirm critical changes with IT.</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function handleEvent(
  event: ChatEvent,
  handlers: {
    onToken: (text: string) => void
    onTrace: (text: string) => void
    onDone: (event: Extract<ChatEvent, { type: 'done' }>) => void
  },
) {
  if (event.type === 'token') handlers.onToken(event.text)
  else if (event.type === 'trace') handlers.onTrace(event.text)
  else if (event.type === 'done') handlers.onDone(event)
  else if (event.type === 'error') throw new Error(event.message)
}
