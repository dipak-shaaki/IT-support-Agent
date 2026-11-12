import type {
  ChatEvent,
  ConversationResponse,
  DashboardStats,
  IngestJob,
  IngestResponse,
} from '../types'

const apiBase = (() => {
  const configured = (import.meta.env.VITE_API_BASE_URL ?? '').trim()
  if (!configured) return ''

  let apiOrigin: URL
  try {
    apiOrigin = new URL(configured)
  } catch {
    throw new Error('VITE_API_BASE_URL must be an absolute HTTP(S) origin.')
  }
  if (
    !['http:', 'https:'].includes(apiOrigin.protocol)
    || apiOrigin.username
    || apiOrigin.password
    || apiOrigin.pathname !== '/'
    || apiOrigin.search
    || apiOrigin.hash
  ) {
    throw new Error('VITE_API_BASE_URL must be a trusted HTTP(S) origin without a path.')
  }
  return apiOrigin.origin
})()

export function apiUrl(path: string): string {
  if (!path.startsWith('/api/') || path.startsWith('//')) {
    throw new Error('API requests must use a controlled /api/ path.')
  }
  return apiBase ? new URL(path, apiBase).toString() : path
}

async function responseError(response: Response): Promise<Error> {
  let message = `Request failed (${response.status})`
  try {
    const body: unknown = await response.json()
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const detail = body.detail
      if (typeof detail === 'string') message = detail
      else if (Array.isArray(detail)) {
        const messages = detail.flatMap((item: unknown) => {
          if (typeof item !== 'object' || item === null || !('msg' in item)) return []
          return typeof item.msg === 'string' ? [item.msg] : []
        })
        if (messages.length) message = messages.join('; ')
      }
    }
  } catch {
    // Keep the HTTP status as the useful error when the response isn't JSON.
  }
  return new Error(message)
}

export async function* streamChat(
  question: string,
  conversationId: string | null,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(apiUrl('/api/chat/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      ...(conversationId ? { conversation_id: conversationId } : {}),
    }),
  })

  if (!response.ok) throw await responseError(response)
  if (!response.body) throw new Error('Streaming is not supported by this browser')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const parseBlock = (block: string): ChatEvent | null => {
    const data = block
      .split(/\r?\n/)
      .find((line) => line.startsWith('data:'))
      ?.slice(5)
      .trim()
    return data ? JSON.parse(data) as ChatEvent : null
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ''

      for (const block of blocks) {
        const event = parseBlock(block)
        if (event) yield event
      }

      if (done) break
    }

    if (buffer.trim()) {
      const event = parseBlock(buffer)
      if (event) yield event
    }
  } finally {
    reader.releaseLock()
  }
}

export async function getConversation(id: string): Promise<ConversationResponse> {
  const response = await fetch(apiUrl(`/api/conversations/${encodeURIComponent(id)}`))
  if (!response.ok) throw await responseError(response)
  return response.json() as Promise<ConversationResponse>
}

export async function getDashboardStats(adminKey: string): Promise<DashboardStats> {
  const response = await fetch(apiUrl('/api/stats'), {
    headers: { 'X-Admin-Key': adminKey },
  })
  if (!response.ok) throw await responseError(response)
  return response.json() as Promise<DashboardStats>
}

export async function ingestDocument(file: File, adminKey: string): Promise<IngestResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(apiUrl('/api/ingest'), {
    method: 'POST',
    headers: { 'X-Admin-Key': adminKey },
    body: formData,
  })
  if (!response.ok) throw await responseError(response)
  return response.json() as Promise<IngestResponse>
}

export async function getIngestJob(jobId: string, adminKey: string): Promise<IngestJob> {
  const response = await fetch(apiUrl(`/api/ingest/jobs/${encodeURIComponent(jobId)}`), {
    headers: { 'X-Admin-Key': adminKey },
  })
  if (!response.ok) throw await responseError(response)
  return response.json() as Promise<IngestJob>
}
