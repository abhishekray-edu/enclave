// AudioWorklet processor for gapless streamed PCM playback.
// Adapted from KevinAHM/pocket-tts-web (PCMPlayerWorklet.js) — the ring-buffer + capacity
// backpressure protocol is kept faithfully; UI/metrics/logging were stripped. Shipped as a
// real file (not a blob: URL) so it loads under the extension CSP (script-src 'self') via
// audioWorklet.addModule(chrome.runtime.getURL('pcm-player-worklet.js')).
//
// Protocol (main thread <-> processor):
//   in : {type:'audio', data:Float32Array} | {type:'reset'} | {type:'stream-ended'}
//   out: {type:'capacity', buffered, capacity, requestSamples, isPlaying}
//        {type:'playback-started'} | {type:'underrun'} | {type:'playback-complete'}
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Ring buffer holding up to 60s of audio at the context sample rate.
    this.bufferSize = sampleRate * 60;
    this.ringBuffer = new Float32Array(this.bufferSize);
    this.readPos = 0;
    this.writePos = 0;
    this.isPlaying = false;
    // Monotonic count of samples actually played (read out of the ring). The main thread maps it
    // to per-sentence write watermarks to reveal displayed text in step with the audio.
    this.totalReadSamples = 0;

    // Start playback only once ~300ms is buffered; aim to keep ~2x that queued.
    this.minBufferSamples = Math.floor((300 * sampleRate) / 1000);
    this.targetBufferSamples = this.minBufferSamples * 2;

    this.streamEnded = false;
    this.playbackCompleteReported = false;

    this.frameCount = 0;
    // ~32 quanta ≈ 170 ms between capacity reports — frequent enough for approximate text/audio
    // sync without flooding the main thread.
    this.reportInterval = 32;

    this.port.onmessage = (e) => {
      switch (e.data.type) {
        case 'audio':
          this.addAudio(e.data.data);
          break;
        case 'reset':
          this.reset();
          break;
        case 'stream-ended':
          this.streamEnded = true;
          // A very short utterance may never reach minBufferSamples, so playback never
          // auto-started. Force it to drain the tail, or report done if nothing is buffered.
          if (!this.isPlaying) {
            if (this.getBufferedSamples() > 0) {
              this.isPlaying = true;
            } else if (!this.playbackCompleteReported) {
              this.port.postMessage({ type: 'playback-complete' });
              this.playbackCompleteReported = true;
            }
          }
          break;
      }
    };

    this.sendCapacityUpdate();
  }

  addAudio(float32Data) {
    const samples = float32Data.length;
    const available = this.getAvailableSpace();

    if (samples > available) {
      // Backpressure should prevent this; drop oldest data to recover if it happens.
      const overflow = samples - available;
      this.readPos = (this.readPos + overflow) % this.bufferSize;
    }

    if (this.writePos + samples <= this.bufferSize) {
      this.ringBuffer.set(float32Data, this.writePos);
      this.writePos += samples;
      if (this.writePos >= this.bufferSize) this.writePos = 0;
    } else {
      const firstPart = this.bufferSize - this.writePos;
      this.ringBuffer.set(float32Data.slice(0, firstPart), this.writePos);
      this.ringBuffer.set(float32Data.slice(firstPart), 0);
      this.writePos = samples - firstPart;
    }

    const buffered = this.getBufferedSamples();
    if (!this.isPlaying && buffered >= this.minBufferSamples) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'playback-started', buffered });
    }

    this.sendCapacityUpdate();
  }

  getAvailableSpace() {
    return this.bufferSize - this.getBufferedSamples() - 128; // small safety margin
  }

  getBufferedSamples() {
    return this.writePos >= this.readPos
      ? this.writePos - this.readPos
      : this.bufferSize - this.readPos + this.writePos;
  }

  sendCapacityUpdate() {
    const buffered = this.getBufferedSamples();
    const capacity = this.getAvailableSpace();
    let requestSamples = 0;
    if (buffered < this.targetBufferSamples) {
      requestSamples = Math.min(capacity, this.targetBufferSamples - buffered);
    }
    this.port.postMessage({
      type: 'capacity',
      buffered,
      capacity,
      requestSamples,
      isPlaying: this.isPlaying,
      totalReadSamples: this.totalReadSamples,
    });
  }

  readInto(outputChannel, count) {
    if (this.readPos + count <= this.bufferSize) {
      outputChannel.set(this.ringBuffer.subarray(this.readPos, this.readPos + count));
      this.readPos += count;
      if (this.readPos >= this.bufferSize) this.readPos = 0;
    } else {
      const firstPart = this.bufferSize - this.readPos;
      outputChannel.set(this.ringBuffer.subarray(this.readPos, this.bufferSize), 0);
      outputChannel.set(this.ringBuffer.subarray(0, count - firstPart), firstPart);
      this.readPos = count - firstPart;
    }
    this.totalReadSamples += count;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) return true;
    const outputChannel = output[0];
    const numSamples = outputChannel.length;

    if (++this.frameCount % this.reportInterval === 0) this.sendCapacityUpdate();

    if (!this.isPlaying) {
      outputChannel.fill(0);
      return true;
    }

    const buffered = this.getBufferedSamples();

    if (buffered < numSamples) {
      // Underrun: play what we have, pad the rest with silence.
      if (buffered > 0) this.readInto(outputChannel, buffered);
      for (let i = buffered; i < numSamples; i++) outputChannel[i] = 0;

      if (this.streamEnded && buffered === 0) {
        if (!this.playbackCompleteReported) {
          this.port.postMessage({ type: 'playback-complete' });
          this.playbackCompleteReported = true;
        }
        this.isPlaying = false;
        this.streamEnded = false;
      } else {
        this.port.postMessage({ type: 'underrun', buffered, needed: numSamples });
        this.sendCapacityUpdate();
      }
    } else {
      this.readInto(outputChannel, numSamples);
    }

    return true;
  }

  reset() {
    this.readPos = 0;
    this.writePos = 0;
    this.ringBuffer.fill(0);
    this.isPlaying = false;
    this.streamEnded = false;
    this.playbackCompleteReported = false;
    this.totalReadSamples = 0;
    this.sendCapacityUpdate();
  }
}

registerProcessor('pcm-processor', PCMProcessor);
