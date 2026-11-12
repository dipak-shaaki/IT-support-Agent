import { useEffect, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { ReactLenis } from 'lenis/react'
import 'lenis/dist/lenis.css'
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  CircleHelp,
  Clock3,
  Database,
  LoaderCircle,
  LockKeyhole,
  MessageSquareText,
  Search,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { getDashboardStats } from '../api/client'
import { readAdminKey, saveAdminKey } from '../api/browserStorage'
import { Sidebar } from '../components/Sidebar'
import type { DashboardStats } from '../types'

const SOURCE_LABELS: Record<string, string> = {
  private_kb: 'Private knowledge base',
  web_search: 'Web fallback',
  insufficient_evidence: 'Insufficient evidence',
  direct: 'Direct reply',
}

export function DashboardPage() {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const [adminKey, setAdminKey] = useState(readAdminKey)
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const savedKey = readAdminKey()
    if (!savedKey) return
    let active = true

    getDashboardStats(savedKey)
      .then((data) => {
        if (active) {
          setStats(data)
          setStatus(`Updated ${new Date().toLocaleTimeString()}`)
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setStatus(error instanceof Error ? error.message : 'Dashboard data could not be loaded.')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => { active = false }
  }, [])

  async function loadStats(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault()
    const key = adminKey.trim()
    if (!key) {
      setStatus('Enter the admin API key to continue.')
      return
    }

    setLoading(true)
    setStatus('')
    saveAdminKey(key)
    try {
      setStats(await getDashboardStats(key))
      setStatus(`Updated ${new Date().toLocaleTimeString()}`)
    } catch (error) {
      setStats(null)
      setStatus(error instanceof Error ? error.message : 'Dashboard data could not be loaded.')
    } finally {
      setLoading(false)
    }
  }

  const sourceCounts = Object.fromEntries((stats?.sources ?? []).map(({ source, count }) => [source, count]))
  const kbAnswers = sourceCounts.private_kb ?? 0
  const webAnswers = sourceCounts.web_search ?? 0
  const unanswered = stats?.gaps_total ?? 0
  const total = stats?.total ?? 0

  return (
    <ReactLenis
      root
      options={{
        autoRaf: true,
        smoothWheel: !reduceMotion,
        lerp: reduceMotion ? 1 : 0.1,
      }}
    >
    <div className="app-shell dashboard-shell">
      <Sidebar active="dashboard" />
      <main className="main-panel dashboard-main">
        <header className="topbar">
          <div className="page-title">Operations</div>
          <a className="button button-outline" href="/"><MessageSquareText size={15} /> Back to support</a>
        </header>

        <div className="dashboard-content">
          <section className="dashboard-heading">
            <div>
              <div className="welcome-kicker"><span className="kicker-sparkle"><Activity size={13} /></span> SERVICE OVERVIEW</div>
              <h1>Support operations</h1>
              <p>Understand what employees need and how well the knowledge base is helping.</p>
            </div>
            <div className="dashboard-access">
              <form className="admin-key-form" onSubmit={(event) => void loadStats(event)}>
                <div className="input-with-icon">
                  <LockKeyhole size={15} />
                  <input
                    type="password"
                    placeholder="Admin API key"
                    aria-label="Admin API key"
                    autoComplete="current-password"
                    value={adminKey}
                    onChange={(event) => setAdminKey(event.target.value)}
                  />
                </div>
                <button className="button button-primary" type="submit" disabled={loading}>
                  {loading ? <LoaderCircle className="spin" size={15} /> : <Search size={15} />}
                  {loading ? 'Loading' : 'Load stats'}
                </button>
              </form>
              <span className={`dashboard-status${status && !stats ? ' error' : ''}`} role="status">{status || 'Admin access required'}</span>
            </div>
          </section>

          {!stats ? (
            <section className="dashboard-empty">
              <div className="empty-illustration"><ShieldCheck size={27} /></div>
              <h2>Operational insights, at a glance.</h2>
              <p>Enter your admin key to see knowledge coverage, answer sources, and recent support questions.</p>
            </section>
          ) : (
            <>
              <section className="stat-grid" aria-label="Key metrics">
                <MetricCard icon={<MessageSquareText size={17} />} label="Questions logged" value={total.toLocaleString()} note="All-time audit records" accent="blue" />
                <MetricCard icon={<BookOpen size={17} />} label="Private KB answers" value={`${percent(kbAnswers, total)}%`} note={`${kbAnswers.toLocaleString()} answered internally`} accent="green" />
                <MetricCard icon={<ArrowUpRight size={17} />} label="Web fallback rate" value={`${percent(webAnswers, total)}%`} note={`${webAnswers.toLocaleString()} used public search`} accent="violet" />
                <MetricCard icon={<CircleHelp size={17} />} label="Knowledge gaps" value={unanswered.toLocaleString()} note="Needs documentation or follow-up" accent="amber" />
              </section>

              <section className="dashboard-grid">
                <div className="surface-card source-card">
                  <SectionHeading icon={<BarChart3 size={16} />} title="Answer sources" subtitle="Where the assistant found its response" />
                  <div className="bar-list">
                    {stats.sources.length ? stats.sources.map((item) => {
                      const color = item.source === 'private_kb' ? 'green' : item.source === 'web_search' ? 'violet' : 'amber'
                      return (
                        <div className="bar-row" key={item.source}>
                          <div className="bar-label-wrap"><span className={`bar-dot ${color}`} /><span>{SOURCE_LABELS[item.source] ?? item.source}</span></div>
                          <div className="bar-track"><span className={`bar-fill ${color}`} style={{ width: `${percent(item.count, total)}%` }} /></div>
                          <strong>{item.count}</strong>
                        </div>
                      )
                    }) : <EmptyState text="No source activity yet." />}
                  </div>
                </div>
                <div className="surface-card activity-card">
                  <SectionHeading icon={<Clock3 size={16} />} title="Questions per day" subtitle="Recent activity over the last two weeks" />
                  {stats.daily.length ? (
                    <div className="daily-chart" aria-label="Daily question counts">
                      {stats.daily.slice().reverse().map((item) => {
                        const max = Math.max(...stats.daily.map((day) => day.count), 1)
                        return (
                          <div className="daily-column" key={item.day} title={`${item.count} questions on ${item.day}`}>
                            <span className="daily-count">{item.count || ''}</span>
                            <div className="daily-bar-track"><span style={{ height: `${Math.max(4, item.count / max * 100)}%` }} /></div>
                            <span className="daily-label">{item.day.slice(5)}</span>
                          </div>
                        )
                      })}
                    </div>
                  ) : <EmptyState text="No daily activity yet." />}
                </div>
              </section>

              <section className="surface-card table-card">
                <SectionHeading icon={<Database size={16} />} title="Knowledge base opportunities" subtitle="Repeated questions that may need clearer documentation" />
                {stats.gaps.length ? (
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead><tr><th>Question</th><th>Asks</th><th>Last seen</th></tr></thead>
                      <tbody>
                        {stats.gaps.map((gap, index) => (
                          <tr key={`${gap.question}-${index}`}>
                            <td><span className="table-question"><CircleHelp size={14} />{gap.question}</span></td>
                            <td><span className="count-badge">{gap.count}</span></td>
                            <td>{formatDate(gap.last_seen)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : <EmptyState text="No knowledge gaps recorded." />}
              </section>

              <section className="surface-card table-card">
                <SectionHeading icon={<ArrowDownToLine size={16} />} title="Recent questions" subtitle="The latest requests handled by the support agent" />
                {stats.recent.length ? (
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead><tr><th>Question</th><th>Source</th><th>When</th></tr></thead>
                      <tbody>
                        {stats.recent.map((item, index) => (
                          <tr key={`${item.created_at}-${index}`}>
                            <td><span className="table-question"><Sparkles size={14} />{item.question}</span></td>
                            <td><span className={`table-source ${item.source}`}>{SOURCE_LABELS[item.source] ?? item.source}</span></td>
                            <td>{formatDate(item.created_at)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : <EmptyState text="No recent questions." />}
              </section>
            </>
          )}
          <footer className="dashboard-footer"><span><span className="online-dot" /> Data is private to your workspace</span><span>Orbit Support · Operations</span></footer>
        </div>
      </main>
    </div>
    </ReactLenis>
  )
}

function MetricCard({
  icon,
  label,
  value,
  note,
  accent,
}: {
  icon: ReactNode
  label: string
  value: string
  note: string
  accent: string
}) {
  return (
    <article className="metric-card">
      <div className={`metric-icon ${accent}`}>{icon}</div>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-note">{note}</span>
    </article>
  )
}

function SectionHeading({
  icon,
  title,
  subtitle,
}: {
  icon: ReactNode
  title: string
  subtitle: string
}) {
  return (
    <div className="section-heading">
      <div className="section-heading-icon">{icon}</div>
      <div><h2>{title}</h2><p>{subtitle}</p></div>
    </div>
  )
}

function EmptyState({ text }: { text: string }) {
  return <div className="chart-empty">{text}</div>
}

function percent(value: number, total: number) {
  return total ? Math.round(value / total * 100) : 0
}

function formatDate(value: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}
