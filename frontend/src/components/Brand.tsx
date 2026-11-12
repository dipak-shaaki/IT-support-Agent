import { LifeBuoy } from 'lucide-react'

export function Brand({ compact = false }: Readonly<{ compact?: boolean }>) {
  return (
    <a className={`brand${compact ? ' brand-compact' : ''}`} href="/" aria-label="IT Support home">
      <span className="brand-mark"><LifeBuoy size={19} strokeWidth={2.2} /></span>
      <span className="brand-name">Orbit<span>Support</span></span>
    </a>
  )
}
