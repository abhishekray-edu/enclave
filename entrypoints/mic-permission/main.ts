// One-time microphone permission page. getUserMedia can't prompt from the side panel or the
// offscreen document, so voice mode opens this page in a normal tab. A click calls
// getUserMedia (triggering Chrome's permission prompt), immediately stops the tracks, tells the
// panel the outcome via a MIC_PERMISSION_RESULT runtime message, and closes the tab. Once the
// extension origin is granted, the offscreen document (USER_MEDIA reason) can capture silently.
import { browser } from 'wxt/browser';

function sendResult(granted: boolean) {
  // The panel may have been closed since it opened this page — a missing listener is fine.
  browser.runtime.sendMessage({ type: 'MIC_PERMISSION_RESULT', granted }).catch(() => {});
}

const app = document.getElementById('app')!;
document.body.style.cssText =
  'margin:0;font-family:system-ui,-apple-system,sans-serif;background:#18181b;color:#f4f4f5;display:flex;min-height:100vh;align-items:center;justify-content:center;';
app.style.cssText = 'max-width:32rem;padding:2rem;text-align:center;';

function render(body: string) {
  app.innerHTML = body;
}

function idle() {
  render(`
    <h1 style="font-size:1.25rem;margin:0 0 .75rem;">Enable voice mode</h1>
    <p style="color:#a1a1aa;line-height:1.5;margin:0 0 1.25rem;">
      Enclave needs access to your microphone to hear you. Audio is processed entirely on your
      device by the local speech model — it is never recorded, uploaded, or sent anywhere.
    </p>
    <button id="allow" style="cursor:pointer;border:0;border-radius:.5rem;background:#f4f4f5;color:#18181b;font-size:.9rem;font-weight:600;padding:.6rem 1.25rem;">
      Allow microphone
    </button>
    <p id="err" style="color:#fca5a5;font-size:.8rem;margin:1rem 0 0;min-height:1rem;"></p>
  `);
  document.getElementById('allow')!.addEventListener('click', request);
}

async function request() {
  const errEl = document.getElementById('err');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // We only needed the grant — release the mic immediately; the offscreen doc captures for real.
    stream.getTracks().forEach((t) => t.stop());
    sendResult(true);
    render(`
      <h1 style="font-size:1.25rem;margin:0 0 .75rem;">Microphone enabled ✓</h1>
      <p style="color:#a1a1aa;line-height:1.5;margin:0;">
        You can close this tab and start talking to Enclave.
      </p>
    `);
    setTimeout(() => { try { window.close(); } catch { /* tab may not be script-closable */ } }, 1200);
  } catch {
    sendResult(false);
    render(`
      <h1 style="font-size:1.25rem;margin:0 0 .75rem;">Microphone blocked</h1>
      <p style="color:#a1a1aa;line-height:1.5;margin:0 0 1rem;">
        Your browser denied microphone access for Enclave. To enable voice mode, allow the
        microphone for this extension in your browser's site settings, then try again.
      </p>
      <button id="retry" style="cursor:pointer;border:0;border-radius:.5rem;background:#f4f4f5;color:#18181b;font-size:.9rem;font-weight:600;padding:.6rem 1.25rem;">
        Try again
      </button>
    `);
    document.getElementById('retry')?.addEventListener('click', idle);
    if (errEl) errEl.textContent = '';
  }
}

idle();
