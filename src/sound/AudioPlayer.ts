import {Script} from '../core/Script.js';

import {CategoryVolumes} from './CategoryVolumes';

export interface AudioPlayerOptions {
  sampleRate?: number;
  channelCount?: number;
  category?: string;
}

export class AudioPlayer extends Script {
  private options: AudioPlayerOptions = {};
  private audioContext?: AudioContext;
  private audioQueue: AudioBuffer[] = [];
  private nextStartTime = 0;
  private gainNode?: GainNode;
  private categoryVolumes?: CategoryVolumes;
  private volume = 1.0;
  private category = 'speech';

  constructor(options: AudioPlayerOptions = {}) {
    super();
    this.options = {sampleRate: 24000, channelCount: 1, ...options};
    if (options.category) {
      this.category = options.category;
    }
  }

  /**
   * Sets the CategoryVolumes instance for this player to respect
   * master/category volumes
   */
  setCategoryVolumes(categoryVolumes: CategoryVolumes) {
    this.categoryVolumes = categoryVolumes;
    this.updateGainNodeVolume();
  }

  /**
   * Sets the specific volume for this player (0.0 to 1.0)
   */
  setVolume(level: number) {
    this.volume = Math.max(0, Math.min(1, level));
    this.updateGainNodeVolume();
  }

  /**
   * Updates the gain node volume based on category volumes
   * Public so CoreSound can update it when master volume changes
   */
  updateGainNodeVolume() {
    if (this.gainNode && this.categoryVolumes) {
      const effectiveVolume =
          this.categoryVolumes.getEffectiveVolume(this.category, this.volume);
      this.gainNode.gain.value = effectiveVolume;
    } else if (this.gainNode) {
      this.gainNode.gain.value = this.volume;
    }
  }

  async initializeAudioContext() {
    if (!this.audioContext) {
      this.audioContext =
          new AudioContext({sampleRate: this.options.sampleRate});
      this.nextStartTime = this.audioContext.currentTime;

      // Create gain node for volume control
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
      this.updateGainNodeVolume();
    }

    // Ensure audio context is running (not suspended)
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  async playAudioChunk(base64AudioData: string) {
    if (!base64AudioData) return;

    await this.initializeAudioContext();
    const arrayBuffer = this.base64ToArrayBuffer(base64AudioData);
    const audioBuffer = this.audioContext!.createBuffer(
        this.options.channelCount!, arrayBuffer.byteLength / 2,
        this.options.sampleRate!);

    const channelData = audioBuffer.getChannelData(0);
    const int16View = new Int16Array(arrayBuffer);

    for (let i = 0; i < int16View.length; i++) {
      channelData[i] = int16View[i] / 32768.0;
    }

    this.audioQueue.push(audioBuffer);
    this.scheduleAudioBuffers();
  }

  scheduleAudioBuffers() {
    while (this.audioQueue.length > 0) {
      const audioBuffer = this.audioQueue.shift()!;
      const currentTime = this.audioContext!.currentTime;
      const startTime = Math.max(this.nextStartTime, currentTime);

      const source = this.audioContext!.createBufferSource();
      source.buffer = audioBuffer;

      // Connect through gain node for volume control
      source.connect(this.gainNode || this.audioContext!.destination);

      // Start playback
      source.start(startTime);

      this.nextStartTime = startTime + audioBuffer.duration;
    }
  }

  clearQueue() {
    this.audioQueue = [];
  }

  getIsPlaying() {
    return this.nextStartTime > this.audioContext!.currentTime;
  }

  getQueueLength() {
    return this.audioQueue.length;
  }

  base64ToArrayBuffer(base64: string) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  stop() {
    this.clearQueue();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = undefined;
      this.gainNode = undefined;
      this.nextStartTime = 0;  // Reset timing
    }
  }

  static isSupported() {
    return !!('AudioContext' in window || 'webkitAudioContext' in window);
  }

  dispose() {
    this.stop();
    super.dispose();
  }
}
