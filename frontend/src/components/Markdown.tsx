import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ReactNode } from 'react'

function MarkdownLink({
  href,
  children,
}: Readonly<{ href?: string; children?: ReactNode }>) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
}

export function Markdown({ children }: Readonly<{ children: string }>) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: MarkdownLink,
      }}
    >
      {children}
    </ReactMarkdown>
  )
}
