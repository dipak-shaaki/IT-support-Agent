const CONVERSATION_STORAGE = 'enterprise-it-support-conversation-id'
const LEGACY_ADMIN_KEY_STORAGE = 'it_copilot_admin_key'
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

let inMemoryAdminKey = ''

export function readAdminKey(): string {
  sessionStorage.removeItem(LEGACY_ADMIN_KEY_STORAGE)
  return inMemoryAdminKey
}

export function saveAdminKey(value: string): void {
  sessionStorage.removeItem(LEGACY_ADMIN_KEY_STORAGE)
  inMemoryAdminKey = value.trim()
}

export function readConversationId(): string | null {
  const value = localStorage.getItem(CONVERSATION_STORAGE)
  if (value && UUID_PATTERN.test(value)) return value
  if (value !== null) localStorage.removeItem(CONVERSATION_STORAGE)
  return null
}

export function saveConversationId(value: string): boolean {
  if (!UUID_PATTERN.test(value)) return false
  localStorage.setItem(CONVERSATION_STORAGE, value)
  return true
}

export function clearConversationId(): void {
  localStorage.removeItem(CONVERSATION_STORAGE)
}
