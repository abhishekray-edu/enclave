import { memo, useState, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import hljs from 'highlight.js/lib/common';
import 'highlight.js/styles/atom-one-dark.css';
import 'katex/dist/katex.min.css';
import { normalizeMathDelimiters } from '@/lib/mathText';

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlight(code: string, lang: string): string {
  try {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
    }
    return hljs.highlightAuto(code).value;
  } catch {
    return escapeHtml(code);
  }
}

const LANG_NAMES: Record<string, string> = {
  py: 'Python', python: 'Python', js: 'JavaScript', javascript: 'JavaScript',
  ts: 'TypeScript', typescript: 'TypeScript', jsx: 'JSX', tsx: 'TSX',
  cpp: 'C++', cc: 'C++', c: 'C', cs: 'C#', csharp: 'C#', java: 'Java',
  go: 'Go', rs: 'Rust', rust: 'Rust', rb: 'Ruby', ruby: 'Ruby', php: 'PHP',
  html: 'HTML', css: 'CSS', json: 'JSON', sql: 'SQL', md: 'Markdown',
  sh: 'Shell', bash: 'Bash', shell: 'Shell', yaml: 'YAML', yml: 'YAML',
};
const LANG_EXT: Record<string, string> = {
  python: 'py', javascript: 'js', typescript: 'ts', csharp: 'cs', ruby: 'rb',
  rust: 'rs', shell: 'sh', bash: 'sh', markdown: 'md', yaml: 'yml',
};

function displayLang(lang: string): string {
  const key = lang.toLowerCase();
  return LANG_NAMES[key] ?? (lang ? lang.charAt(0).toUpperCase() + lang.slice(1) : 'Code');
}
function fileExt(lang: string): string {
  const key = lang.toLowerCase();
  return LANG_EXT[key] ?? (key || 'txt');
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function IconBtn({ title, onClick, children }: { title: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 transition hover:bg-white/10 hover:text-zinc-100"
    >
      {children}
    </button>
  );
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const html = highlight(code, lang);

  return (
    <div className="code-block">
      <div className="code-block__header">
        <span className="code-block__lang">{displayLang(lang)}</span>
        <div className="flex items-center gap-0.5">
          <IconBtn title="Download" onClick={() => downloadText(`snippet.${fileExt(lang)}`, code)}>
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3v12" />
              <path d="m7 11 5 5 5-5" />
              <path d="M5 21h14" />
            </svg>
          </IconBtn>
          <IconBtn
            title={copied ? 'Copied' : 'Copy'}
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(code);
                setCopied(true);
                setTimeout(() => setCopied(false), 1200);
              } catch {
                /* clipboard unavailable */
              }
            }}
          >
            {copied ? (
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m20 6-11 11-5-5" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="11" height="11" rx="2" />
                <path d="M5 15V5a2 2 0 0 1 2-2h10" />
              </svg>
            )}
          </IconBtn>
        </div>
      </div>
      <pre className="code-block__body">
        <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}

const components: Components = {
  code({ className, children }) {
    const match = /language-(\w+)/.exec(className ?? '');
    const text = String(children ?? '');
    const isBlock = match !== null || text.includes('\n');
    if (!isBlock) {
      return <code className="inline-code">{children as ReactNode}</code>;
    }
    return <CodeBlock lang={match?.[1] ?? ''} code={text.replace(/\n$/, '')} />;
  },
  // The CodeBlock renders its own <pre>; flatten react-markdown's wrapper.
  pre({ children }) {
    return <>{children}</>;
  },
};

function MarkdownImpl({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        // singleDollarTextMath:false keeps prose like "$5 and $10" from parsing as math; the
        // normalizer converts any real \(...\)/\[...\] to $$...$$ so nothing is lost.
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {normalizeMathDelimiters(content)}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
