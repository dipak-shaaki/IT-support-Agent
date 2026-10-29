const KEY_STORAGE = 'it_copilot_admin_key';
const $ = (id) => document.getElementById(id);
const SOURCE_LABELS = {
  private_kb: 'Private KB',
  web_search: 'Web fallback',
  insufficient_evidence: 'Insufficient evidence',
  direct: 'Direct reply',
};

function escapeHtml(s = '') {
  return String(s).replace(/[&<>'"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}
function pct(part, total) { return total ? Math.round((part / total) * 100) : 0; }
function when(iso) { return iso ? escapeHtml(iso.slice(0, 16).replace('T', ' ')) : '—'; }
function plural(n, word) { return `${n} ${word}${n === 1 ? '' : 's'}`; }

function kpi(label, value, hint) {
  return `<div class="kpi"><div class="kpi-label">${escapeHtml(label)}</div>
    <div class="kpi-value">${escapeHtml(value)}</div>
    <div class="kpi-hint">${escapeHtml(hint)}</div></div>`;
}

function bars(items) {
  if (!items.length) return '<div class="empty">No data yet.</div>';
  const max = Math.max(...items.map((i) => i.count));
  return items.map((i) => `<div class="bar-row">
      <span class="bar-label">${escapeHtml(i.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${Math.round((i.count / max) * 100)}%"></span></span>
      <span class="bar-count">${i.count}</span>
    </div>`).join('');
}

function render(data) {
  const counts = Object.fromEntries(data.sources.map((s) => [s.source, s.count]));
  const total = data.total;
  const deflected = counts.private_kb || 0;
  const fallback = counts.web_search || 0;
  const unanswered = counts.insufficient_evidence || 0;

  $('kpis').innerHTML =
    kpi('Total questions', total, 'all time, from the audit log') +
    kpi('Answered from private KB', `${pct(deflected, total)}%`, `${plural(deflected, 'answer')} without leaving the KB`) +
    kpi('KB gap rate', `${pct(fallback + unanswered, total)}%`, 'fell back to web or gave up') +
    kpi('Unanswered', unanswered, 'users told to contact the help desk');

  $('sources').innerHTML = bars(
    data.sources.map((s) => ({ label: SOURCE_LABELS[s.source] || s.source, count: s.count }))
  );

  $('daily').innerHTML = bars(
    [...data.daily].reverse().map((d) => ({ label: d.day, count: d.count }))
  );

  $('gaps').innerHTML = data.gaps.length
    ? data.gaps.map((g) => `<tr><td>${escapeHtml(g.question)}</td><td>${g.count}</td><td>${when(g.last_seen)}</td></tr>`).join('')
    : '<tr><td colspan="3" class="empty">No gaps recorded.</td></tr>';

  $('recent').innerHTML = data.recent.length
    ? data.recent.map((r) => `<tr><td>${when(r.created_at)}</td><td>${escapeHtml(r.question)}</td>
        <td>${escapeHtml(SOURCE_LABELS[r.source] || r.source)}</td></tr>`).join('')
    : '<tr><td colspan="3" class="empty">No data yet.</td></tr>';
}

async function loadStats() {
  const key = $('adminKey').value.trim();
  if (!key) { $('status').textContent = 'Enter the admin API key.'; return; }
  sessionStorage.setItem(KEY_STORAGE, key);
  $('status').textContent = 'Loading…';
  try {
    const res = await fetch('/api/stats', { headers: { 'X-Admin-Key': key } });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    render(data);
    $('status').textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    $('status').textContent = `Error: ${e.message}`;
  }
}

$('keyForm').addEventListener('submit', (e) => { e.preventDefault(); loadStats(); });

const saved = sessionStorage.getItem(KEY_STORAGE);
if (saved) { $('adminKey').value = saved; loadStats(); }
