// AudioWorklet processor for microphone capture (sibling of pcm-player-worklet.js). The
// AudioContext is created at 16 kHz (see lib/micCapture.ts) so Chrome resamples the mic to the
// rate Silero VAD + Moonshine expect. This processor rings the 128-sample render quanta into
// fixed 512-sample frames (one Silero step) and transfers each finished frame to the main
// thread, which forwards it to stt.worker.ts. Shipped as a real file (not a blob: URL) so it
// loads under the extension CSP (script-src 'self') via
// audioWorklet.addModule(chrome.runtime.getURL('mic-capture-worklet.js')).
//
// Protocol (processor -> main): {type:'frame', data:Float32Array(512)} (transferred)
const FRAME_SAMPLES = 512;

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(FRAME_SAMPLES);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    // No input connected yet (or a silent render) — keep the node alive.
    if (!input || !input[0]) return true;
    const channel = input[0]; // mono (channelCount: 1)
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.filled++] = channel[i];
      if (this.filled === FRAME_SAMPLES) {
        const frame = this.buffer; // hand off this buffer…
        this.port.postMessage({ type: 'frame', data: frame }, [frame.buffer]);
        this.buffer = new Float32Array(FRAME_SAMPLES); // …and start a fresh one
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);
