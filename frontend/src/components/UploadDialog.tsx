import { useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { CheckCircle2, FileUp, LoaderCircle, LockKeyhole, X } from 'lucide-react'
import { getIngestJob, ingestDocument } from '../api/client'

const ADMIN_KEY_STORAGE = 'it_copilot_admin_key'
const JOB_POLL_INTERVAL_MS = 2000
const JOB_POLL_TIMEOUT_MS = 5 * 60 * 1000

async function waitForJob(jobId: string, adminKey: string) {
  const deadline = Date.now() + JOB_POLL_TIMEOUT_MS
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL_MS))
    const job = await getIngestJob(jobId, adminKey)
    if (job.status === 'done' || job.status === 'error') return job
  }
  throw new Error('Indexing is still running. Check back in a few minutes.')
}

export function UploadDialog({ onClose }: { onClose: () => void }) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [adminKey, setAdminKey] = useState(() => sessionStorage.getItem(ADMIN_KEY_STORAGE) ?? '')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<{ kind: 'success' | 'error' | 'info'; text: string } | null>(null)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const file = fileInput.current?.files?.[0]
    if (!file) {
      setStatus({ kind: 'error', text: 'Choose a document to upload.' })
      return
    }
    if (!adminKey.trim()) {
      setStatus({ kind: 'error', text: 'Enter the admin API key.' })
      return
    }

    setBusy(true)
    setStatus(null)
    sessionStorage.setItem(ADMIN_KEY_STORAGE, adminKey.trim())
    try {
      const result = await ingestDocument(file, adminKey.trim())
      if ('job_id' in result) {
        setStatus({ kind: 'info', text: `Queued ${result.file}. Indexing…` })
        const job = await waitForJob(result.job_id, adminKey.trim())
        if (job.status === 'error') throw new Error(job.error || 'Indexing failed.')
        setStatus({
          kind: 'success',
          text: `Indexed ${job.file}: ${job.chunks} chunks added.`,
        })
      } else {
        setStatus({
          kind: 'success',
          text: `Indexed ${result.file}: ${result.chunks} chunks added.`,
        })
      }
      if (fileInput.current) fileInput.current.value = ''
    } catch (error) {
      setStatus({
        kind: 'error',
        text: error instanceof Error ? error.message : 'Document upload failed.',
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !busy) onClose()
    }}>
      <section className="upload-dialog" role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <button className="icon-button dialog-close" type="button" onClick={onClose} aria-label="Close upload dialog">
          <X size={18} />
        </button>
        <div className="dialog-symbol"><FileUp size={22} /></div>
        <span className="eyebrow">KNOWLEDGE BASE</span>
        <h2 id="upload-title">Add a support document</h2>
        <p className="dialog-description">Upload a PDF, text, Markdown, or Word document. It will be indexed for future support answers.</p>
        <form className="upload-form" onSubmit={submit}>
          <label htmlFor="upload-admin-key">Admin API key</label>
          <div className="input-with-icon">
            <LockKeyhole size={16} />
            <input
              id="upload-admin-key"
              type="password"
              autoComplete="current-password"
              value={adminKey}
              onChange={(event) => setAdminKey(event.target.value)}
              required
            />
          </div>
          <label htmlFor="upload-file">Choose a document</label>
          <input
            ref={fileInput}
            id="upload-file"
            className="file-input"
            type="file"
            accept=".pdf,.txt,.md,.docx"
            required
          />
          {status && (
            <div className={`form-status ${status.kind}`} role="status">
              {status.kind === 'success' && <CheckCircle2 size={16} />}
              <span>{status.text}</span>
            </div>
          )}
          <button className="button button-primary upload-submit" type="submit" disabled={busy}>
            {busy ? <><LoaderCircle className="spin" size={16} /> Indexing…</> : 'Upload and index'}
          </button>
        </form>
        <p className="dialog-footnote">The admin key is stored only for this browser session.</p>
      </section>
    </div>
  )
}
