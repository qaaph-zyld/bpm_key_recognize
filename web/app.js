import { guess } from 'https://cdn.skypack.dev/web-audio-beat-detector';

const el = {
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  status: document.getElementById('status'),
  statusText: document.getElementById('statusText'),
  results: document.getElementById('results'),
  fileName: document.getElementById('fileName'),
  bpmValue: document.getElementById('bpmValue'),
  bpmDetails: document.getElementById('bpmDetails'),
  keyValue: document.getElementById('keyValue'),
  keyDetails: document.getElementById('keyDetails'),
};

const PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

let audioCtx;
let essentia; // Essentia instance
let extractor; // EssentiaExtractor instance
let essentiaReady = false;

function setStatus(msg) {
  el.statusText.textContent = msg;
}

function showResults(on) {
  el.results.hidden = !on;
}

async function initEssentia() {
  if (essentiaReady) return;
  try {
    setStatus('Loading Essentia.js (WASM)...');
    const wasm = await EssentiaWASM();
    essentia = new Essentia(wasm);
    extractor = new EssentiaExtractor(wasm);
    essentiaReady = true;
    setStatus('Essentia.js loaded. Ready.');
  } catch (err) {
    console.error(err);
    setStatus('Failed to load Essentia.js. Key detection may be unavailable.');
  }
}

function reduceTo12(chroma) {
  const n = chroma.length;
  if (n === 12) return Float32Array.from(chroma);
  const out = new Float32Array(12);
  if (n % 12 === 0) {
    const group = n / 12;
    for (let i = 0; i < 12; i++) {
      let s = 0;
      for (let g = 0; g < group; g++) s += chroma[i * group + g] || 0;
      out[i] = s;
    }
  } else {
    // fold by modulo 12
    for (let i = 0; i < n; i++) out[i % 12] += chroma[i] || 0;
  }
  return out;
}

function normDot(a, b) {
  let sa = 0, sb = 0, s = 0;
  for (let i = 0; i < a.length; i++) { const x=a[i]; const y=b[i]; sa += x*x; sb += y*y; s += x*y; }
  const na = Math.sqrt(sa) + 1e-9; const nb = Math.sqrt(sb) + 1e-9; return s / (na * nb);
}

function rotate12(arr, shift) {
  const out = new Float32Array(12);
  for (let i = 0; i < 12; i++) out[i] = arr[(i + shift) % 12];
  return out;
}

function estimateKeyFromHPCPMean(hpcpMean12) {
  const scores = [];
  for (let s = 0; s < 12; s++) {
    const rot = rotate12(hpcpMean12, (12 - s) % 12);
    scores.push({ score: normDot(rot, KRUMHANSL_MAJOR), mode: 'major', shift: s });
    scores.push({ score: normDot(rot, KRUMHANSL_MINOR), mode: 'minor', shift: s });
  }
  scores.sort((a,b)=>b.score - a.score);
  const best = scores[0];
  const second = scores[1] || { score: 0, shift: 0, mode: 'major' };
  const keyName = `${PITCH_CLASSES[best.shift]} ${best.mode}`;
  const confidence = (best.score - second.score) / (Math.abs(best.score) + 1e-9);
  return { key: keyName, confidence: Math.max(0, confidence), top2: [best, second] };
}

async function runBPM(audioBuffer) {
  try {
    const { bpm, offset, tempo } = await guess(audioBuffer);
    return { bpm: bpm || tempo || null, confidence: 1.0, method: 'web-audio-beat-detector' };
  } catch (err) {
    console.warn('BPM detection failed', err);
    return { bpm: null, confidence: 0, method: 'none', error: String(err) };
  }
}

function runKey(audioBuffer) {
  if (!essentiaReady) return { key: null, confidence: 0, method: 'none', error: 'Essentia not loaded' };
  try {
    const sr = audioBuffer.sampleRate;
    const mono = essentia.audioBufferToMonoSignal(audioBuffer);
    const maxSeconds = 60;
    const maxSamples = Math.min(mono.length, Math.floor(sr * maxSeconds));
    const signal = mono.subarray(0, maxSamples);

    const frameSize = 4096; const hopSize = 2048;
    const frames = essentia.FrameGenerator(signal, frameSize, hopSize);

    let acc = null; let count = 0;
    for (let i = 0; i < frames.size(); i += 2) { // stride to speed up
      const frame = frames.get(i);
      const hpcp = extractor.hpcpExtractor(frame, sr);
      if (!acc) acc = new Float32Array(hpcp.length).fill(0);
      for (let k = 0; k < hpcp.length; k++) acc[k] += hpcp[k] || 0;
      count++;
    }
    if (!acc || count === 0) return { key: null, confidence: 0, method: 'hpcp+krumhansl' };

    for (let i = 0; i < acc.length; i++) acc[i] /= count;
    const hpcp12 = reduceTo12(acc);
    const res = estimateKeyFromHPCPMean(hpcp12);
    return { key: res.key, confidence: res.confidence, method: 'hpcp+krumhansl' };
  } catch (err) {
    console.warn('Key detection failed', err);
    return { key: null, confidence: 0, method: 'hpcp+krumhansl', error: String(err) };
  }
}

async function handleFile(file) {
  try {
    showResults(false);
    setStatus('Decoding audio...');
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    el.fileName.textContent = file.name;

    setStatus('Analyzing BPM...');
    const bpmRes = await runBPM(audioBuffer);
    el.bpmValue.textContent = bpmRes.bpm ? `${bpmRes.bpm.toFixed(2)} BPM` : '—';
    el.bpmDetails.textContent = `method=${bpmRes.method}`;

    setStatus('Analyzing Key...');
    const keyRes = await runKey(audioBuffer);
    el.keyValue.textContent = keyRes.key || '—';
    el.keyDetails.textContent = `method=${keyRes.method}, conf=${(keyRes.confidence||0).toFixed(3)}`;

    setStatus('Done.');
    showResults(true);
  } catch (err) {
    console.error(err);
    setStatus('Failed to analyze audio.');
  }
}

function setupUI() {
  el.fileInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleFile(file);
  });
  const dz = el.dropzone;
  dz.addEventListener('click', () => el.fileInput.click());
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('hover'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault(); dz.classList.remove('hover');
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

(async function main() {
  setupUI();
  await initEssentia();
})();
