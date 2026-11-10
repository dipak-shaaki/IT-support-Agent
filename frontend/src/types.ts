export type Citation = {
  title: string
  url: string
  type: 'private_kb' | 'web' | string
}

export type ConversationMessage = {
  role: 'user' | 'assistant'
  content: string
  source?: string
  citations?: Citation[]
  pending?: boolean
}

export type ChatEvent =
  | { type: 'token'; text: string }
  | { type: 'trace'; text: string; node?: string }
  | {
      type: 'done'
      conversation_id: string
      answer: string
      source_used: string
      citations: Citation[]
      retrieval_query: string
    }
  | { type: 'error'; message: string }

export type SourceCount = { source: string; count: number }
export type DailyCount = { day: string; count: number }
export type KnowledgeGap = { question: string; count: number; last_seen: string }
export type RecentQuery = { created_at: string; question: string; source: string }

export type DashboardStats = {
  total: number
  sources: SourceCount[]
  daily: DailyCount[]
  gaps: KnowledgeGap[]
  gaps_total: number
  recent: RecentQuery[]
}

export type ConversationResponse = {
  conversation_id: string
  messages: ConversationMessage[]
}

export type IngestResponse =
  | { message: string; file: string; chunks: number; ids_created: string[] }
  | { job_id: string; status: 'queued'; file: string }

export type IngestJob = {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  file: string
  chunks: number
  ids_created: number
  error: string
}
