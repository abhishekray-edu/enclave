import { Readability, isProbablyReaderable } from '@mozilla/readability';
import type { PageContent } from '@/lib/types';

const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS', 'AUDIO', 'VIDEO']);
const BLOCK_TAGS = new Set([
  'P', 'DIV', 'SECTION', 'ARTICLE', 'HEADER', 'FOOTER', 'MAIN', 'ASIDE', 'NAV',
  'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'UL', 'OL', 'TR', 'TABLE',
  'PRE', 'BLOCKQUOTE', 'BR', 'HR', 'FORM', 'LABEL', 'TEXTAREA',
]);

function isHidden(el: Element): boolean {
  if (el.hasAttribute('hidden') || el.getAttribute('aria-hidden') === 'true') return true;
  const s = getComputedStyle(el);
  return s.display === 'none' || s.visibility === 'hidden';
}

/**
 * innerText-style extraction that ALSO descends into open shadow roots and
 * same-origin iframes — the places normal innerText/Readability silently miss.
 * Used for app-like pages (SPAs, dashboards, coding sites) where article
 * extraction loses the real content.
 */
function deepText(root: Node, out: string[], depth = 0) {
  if (depth > 50) return;

  if (root.nodeType === Node.TEXT_NODE) {
    const t = root.textContent;
    if (t && t.trim()) out.push(t.replace(/\s+/g, ' '));
    return;
  }
  if (root.nodeType !== Node.ELEMENT_NODE) return;

  const el = root as Element;
  if (SKIP_TAGS.has(el.tagName) || isHidden(el)) return;

  // Form field values aren't in the text tree — capture them explicitly.
  if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
    const v = (el as HTMLInputElement | HTMLTextAreaElement).value;
    if (v && v.trim()) out.push(v.trim());
  }

  // Descend into shadow DOM (web components).
  const shadow = (el as Element & { shadowRoot?: ShadowRoot | null }).shadowRoot;
  if (shadow) {
    shadow.childNodes.forEach((c) => deepText(c, out, depth + 1));
  }

  // Descend into same-origin iframes (cross-origin throws and is skipped).
  if (el.tagName === 'IFRAME') {
    try {
      const doc = (el as HTMLIFrameElement).contentDocument;
      if (doc?.body) deepText(doc.body, out, depth + 1);
    } catch {
      /* cross-origin frame — not readable from here */
    }
    return;
  }

  el.childNodes.forEach((c) => deepText(c, out, depth + 1));
  if (BLOCK_TAGS.has(el.tagName)) out.push('\n');
}

function collectDeepText(): string {
  const out: string[] = [];
  if (document.body) deepText(document.body, out);
  return out
    .join(' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/ ?\n ?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function extractPage(): PageContent {
  const selection = window.getSelection()?.toString() ?? '';

  // Comprehensive text first — this is what guarantees we don't drop content.
  const deep = collectDeepText();

  // Readability is only a *cleanup* pass for genuine articles. We use it for
  // metadata always, and for the body ONLY when it captured most of the page
  // (i.e. it's a real article, not a multi-panel app where it would lose content).
  let title = document.title;
  let excerpt = '';
  let byline = '';
  let siteName = '';
  let articleText = '';
  try {
    if (isProbablyReaderable(document)) {
      const article = new Readability(document.cloneNode(true) as Document).parse();
      if (article) {
        title = article.title || title;
        excerpt = article.excerpt ?? '';
        byline = article.byline ?? '';
        siteName = article.siteName ?? '';
        articleText = (article.textContent ?? '').trim();
      }
    }
  } catch {
    /* fall back to deep text */
  }

  const useArticle = articleText.length > 500 && articleText.length >= deep.length * 0.5;
  const textContent = useArticle ? articleText : deep || articleText;

  return {
    title,
    url: location.href,
    textContent,
    excerpt,
    byline,
    siteName,
    selection,
  };
}

export default defineContentScript({
  matches: ['<all_urls>'],
  main() {
    browser.runtime.onMessage.addListener((message, _sender, sendResponse) => {
      if (message?.type === 'GET_PAGE_CONTENT') {
        sendResponse(extractPage());
      }
    });
  },
});
