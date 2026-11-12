const ADMIN_KEY_STORAGE = 'it_copilot_admin_key'
const CONVERSATION_STORAGE = 'enterprise-it-support-conversation-id'
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

function isSafeAdminKey(value: string): boolean {
  if (!value.length || value.length > 512) return false
  return Array.from(value).every((character) => {
    const code = character.codePointAt(0) ?? 0
    return code >= 32 && code !== 127
  })
}

export function readAdminKey(): string {
  const value = sessionStorage.getItem(ADMIN_KEY_STORAGE)
  if (value && isSafeAdminKey(value)) return value
  if (value !== null) sessionStorage.removeItem(ADMIN_KEY_STORAGE)
  return ''
}

export function saveAdminKey(value: string): void {
  const normalized = value.trim()
  if (isSafeAdminKey(normalized)) sessionStorage.setItem(ADMIN_KEY_STORAGE, normalized)
  else sessionStorage.removeItem(ADMIN_KEY_STORAGE)
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
