import { Readability, isProbablyReaderable } from '@mozilla/readability';
import type { GetPageContentRequest, PageBlock, PageContent } from '@/lib/types';

const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS', 'AUDIO', 'VIDEO']);
const BLOCK_TAGS = new Set([
  'P', 'DIV', 'SECTION', 'ARTICLE', 'HEADER', 'FOOTER', 'MAIN', 'ASIDE', 'NAV',
  'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'UL', 'OL', 'TR', 'TABLE',
  'PRE', 'BLOCKQUOTE', 'BR', 'HR', 'FORM', 'LABEL', 'TEXTAREA',
]);
const MAX_EXTRACTED_CHARS = 60_000;

interface TextBudget {
  chars: number;
  truncated: boolean;
}

function isHidden(el: Element): boolean {
  if (el.hasAttribute('hidden') || el.getAttribute('aria-hidden') === 'true') return true;
  const s = getComputedStyle(el);
  return s.display === 'none' || s.visibility === 'hidden';
}

function pushText(out: string[], budget: TextBudget, text: string): boolean {
  const normalized = text.replace(/\s+/g, ' ');
  if (!normalized.trim()) return true;
  const remaining = MAX_EXTRACTED_CHARS - budget.chars;
  if (remaining <= 0) {
    budget.truncated = true;
    return false;
  }
  if (normalized.length > remaining) {
    out.push(normalized.slice(0, remaining));
    budget.chars += remaining;
    budget.truncated = true;
    return false;
  }
  out.push(normalized);
  budget.chars += normalized.length;
  return true;
}

const HEADING_TAGS = new Map<string, number>([
  ['H1', 1], ['H2', 2], ['H3', 3], ['H4', 4], ['H5', 5], ['H6', 6],
]);

/** Accumulates document structure (headings + text blocks) in parallel with the flat text. */
interface BlockCollector {
  blocks: PageBlock[];
  /** Pending text for the current (not-yet-emitted) text block. */
  buf: string[];
}

/** Emit the buffered text as one 'text' block (skipped if empty), then reset the buffer. */
function flushTextBlock(bc: BlockCollector) {
  const text = bc.buf.join(' ').replace(/\s+/g, ' ').trim();
  bc.buf.length = 0;
  if (text) bc.blocks.push({ type: 'text', text });
}

/**
 * innerText-style extraction that ALSO descends into open shadow roots and
 * same-origin iframes — the places normal innerText/Readability silently miss.
 * Used for app-like pages (SPAs, dashboards, coding sites) where article
 * extraction loses the real content. Simultaneously records coarse structure
 * (headings + per-block text) into `bc` for chunking and structure-aware truncation.
 * `suppress` keeps a heading's own text out of the block buffer (it's already its
 * own heading block) while still letting it land in the flat `out` text.
 */
function deepText(root: Node, out: string[], budget: TextBudget, bc: BlockCollector, suppress: boolean, depth = 0) {
  if (budget.truncated || depth > 50) return;

  if (root.nodeType === Node.TEXT_NODE) {
    const t = root.textContent;
    if (t) {
      pushText(out, budget, t);
      if (!suppress) {
        const n = t.replace(/\s+/g, ' ');
        if (n.trim()) bc.buf.push(n);
      }
    }
    return;
  }
  if (root.nodeType !== Node.ELEMENT_NODE) return;

  const el = root as Element;
  if (SKIP_TAGS.has(el.tagName) || isHidden(el)) return;

  // Headings become their own structural block (and a section boundary).
  const headingLevel = HEADING_TAGS.get(el.tagName);
  if (headingLevel && !suppress) {
    flushTextBlock(bc);
    const htext = (el.textContent ?? '').replace(/\s+/g, ' ').trim();
    if (htext) bc.blocks.push({ type: 'heading', level: headingLevel, text: htext });
    // Walk children suppressed so their text still reaches `out` but isn't double-counted.
    for (const c of el.childNodes) {
      if (budget.truncated) return;
      deepText(c, out, budget, bc, true, depth + 1);
    }
    if (!budget.truncated) out.push('\n');
    return;
  }

  // Form field values aren't in the text tree — capture them explicitly.
  if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
    const v = (el as HTMLInputElement | HTMLTextAreaElement).value;
    if (v) {
      const trimmed = v.trim();
      if (trimmed && !suppress) bc.buf.push(trimmed);
      if (v && !pushText(out, budget, v.trim())) return;
    }
  }

  // Descend into shadow DOM (web components).
  const shadow = (el as Element & { shadowRoot?: ShadowRoot | null }).shadowRoot;
  if (shadow) {
    for (const c of shadow.childNodes) {
      if (budget.truncated) return;
      deepText(c, out, budget, bc, suppress, depth + 1);
    }
  }

  // Descend into same-origin iframes (cross-origin throws and is skipped).
  if (el.tagName === 'IFRAME') {
    try {
      const doc = (el as HTMLIFrameElement).contentDocument;
      if (doc?.body) deepText(doc.body, out, budget, bc, suppress, depth + 1);
    } catch {
      /* cross-origin frame — not readable from here */
    }
    return;
  }

  for (const c of el.childNodes) {
    if (budget.truncated) return;
    deepText(c, out, budget, bc, suppress, depth + 1);
  }
  if (!budget.truncated && BLOCK_TAGS.has(el.tagName)) {
    out.push('\n');
    if (!suppress) flushTextBlock(bc); // block boundary → emit a text block
  }
}

/** Walk a root, returning both flat text and coarse structure. */
function extractStructured(root: Node): { text: string; truncated: boolean; blocks: PageBlock[] } {
  const out: string[] = [];
  const bc: BlockCollector = { blocks: [], buf: [] };
  const budget: TextBudget = { chars: 0, truncated: false };
  deepText(root, out, budget, bc, false);
  flushTextBlock(bc);
  const text = out
    .join(' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/ ?\n ?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return { text, truncated: budget.truncated, blocks: bc.blocks };
}

/** Text of block elements currently intersecting the viewport (for viewport-aware ranking). */
function collectViewport(): string {
  const vh = window.innerHeight || 0;
  if (!vh || !document.body) return '';
  const parts: string[] = [];
  let chars = 0;
  for (const el of document.body.querySelectorAll('p,li,h1,h2,h3,h4,h5,h6,td,blockquote,pre,article,section')) {
    if (chars > 8000) break;
    const r = el.getBoundingClientRect();
    if (r.height > 0 && r.bottom > 0 && r.top < vh) {
      const t = (el.textContent ?? '').replace(/\s+/g, ' ').trim();
      if (t) {
        parts.push(t);
        chars += t.length;
      }
    }
  }
  return parts.join(' ');
}

function extractPage(wantViewport = false): PageContent {
  const selection = window.getSelection()?.toString() ?? '';

  // Comprehensive text first — this is what guarantees we don't drop content.
  const deep = document.body
    ? extractStructured(document.body)
    : { text: '', truncated: false, blocks: [] as PageBlock[] };

  // Readability is only a *cleanup* pass for genuine articles. We use it for
  // metadata always, and for the body ONLY when it captured most of the page
  // (i.e. it's a real article, not a multi-panel app where it would lose content).
  let title = document.title;
  let excerpt = '';
  let byline = '';
  let siteName = '';
  let articleText = '';
  let articleBlocks: PageBlock[] = [];
  try {
    if (isProbablyReaderable(document)) {
      const article = new Readability(document.cloneNode(true) as Document).parse();
      if (article) {
        title = article.title || title;
        excerpt = article.excerpt ?? '';
        byline = article.byline ?? '';
        siteName = article.siteName ?? '';
        // Derive both clean text AND structure from the same cleaned article HTML.
        if (article.content) {
          const parsed = new DOMParser().parseFromString(article.content, 'text/html');
          const res = extractStructured(parsed.body);
          articleText = res.text.trim();
          articleBlocks = res.blocks;
        } else {
          articleText = (article.textContent ?? '').trim();
        }
      }
    }
  } catch {
    /* fall back to deep text */
  }

  const useArticle = articleText.length > 500 && articleText.length >= deep.text.length * 0.5;
  const textContent = useArticle ? articleText : deep.text || articleText;
  const blocks = useArticle ? articleBlocks : deep.blocks;

  return {
    title,
    url: location.href,
    textContent,
    excerpt,
    byline,
    siteName,
    selection,
    sourceTruncated: useArticle ? false : deep.truncated,
    blocks: blocks.length ? blocks : undefined,
    viewportText: wantViewport ? collectViewport() || undefined : undefined,
  };
}

export default defineContentScript({
  matches: ['<all_urls>'],
  main() {
    browser.runtime.onMessage.addListener((message, _sender, sendResponse) => {
      const msg = message as GetPageContentRequest | undefined;
      if (msg?.type === 'GET_PAGE_CONTENT') {
        sendResponse(extractPage(Boolean(msg.wantViewport)));
      }
    });
  },
});
