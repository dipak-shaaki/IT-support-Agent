import { useState } from 'react'
import { Activity, FilePlus2, Sparkles, Upload } from 'lucide-react'
import { Brand } from './Brand'
import { UploadDialog } from './UploadDialog'

export function Sidebar({ active = 'chat' }: { active?: 'chat' | 'dashboard' }) {
  const [uploadOpen, setUploadOpen] = useState(false)

  return (
    <>
      <aside className="sidebar">
        <Brand />
        <nav className="mobile-sidebar-actions" aria-label="Quick actions">
          <button type="button" onClick={() => setUploadOpen(true)} aria-label="Add a support document">
            <Upload size={16} />
          </button>
          <a href="/dashboard" aria-label="Operations dashboard"><Activity size={16} /></a>
        </nav>
        <nav className="side-nav" aria-label="Workspace">
          <a className={`nav-item${active === 'chat' ? ' active' : ''}`} href="/">
            <Sparkles size={17} /><span>Support chat</span>
          </a>
          <a className={`nav-item${active === 'dashboard' ? ' active' : ''}`} href="/dashboard">
            <Activity size={17} /><span>Operations</span>
          </a>
          <button className="nav-item" type="button" onClick={() => setUploadOpen(true)}>
            <FilePlus2 size={17} /><span>Knowledge base</span>
          </button>
        </nav>

      </aside>
      {uploadOpen && <UploadDialog onClose={() => setUploadOpen(false)} />}
    </>
  )
}
