// Mouth Groove engine: pure DSP, no DOM. Works in browser and Node.
(function (root) {
  'use strict';

  const CLASSES = ['kick', 'snare', 'hat', 'bass'];
  const DRUM_NOTES = { kick: 36, snare: 38, hat: 42 };

  // ---------- FFT (radix-2, in place) ----------
  function makeFFT(n) {
    const levels = Math.log2(n) | 0;
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let x = i, r = 0;
      for (let j = 0; j < levels; j++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    return function (re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
      }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half;
            const tre = re[l] * cos[k] + im[l] * sin[k];
            const tim = -re[l] * sin[k] + im[l] * cos[k];
            re[l] = re[j] - tre; im[l] = im[j] - tim;
            re[j] += tre; im[j] += tim;
          }
        }
      }
    };
  }

  function hann(n) { const w = new Float32Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)); return w; }

  function biquadLP(x, sr, fc, q) {
    const w0 = 2 * Math.PI * fc / sr, alpha = Math.sin(w0) / (2 * (q || 0.7071)), c = Math.cos(w0);
    const a0 = 1 + alpha;
    const b0 = (1 - c) / 2 / a0, b1 = (1 - c) / a0, b2 = b0, a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
    const y = new Float32Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      y[i] = yi; x2 = x1; x1 = xi; y2 = y1; y1 = yi;
    }
    return y;
  }

  function biquadHP(x, sr, fc, q) {
    const w0 = 2 * Math.PI * fc / sr, alpha = Math.sin(w0) / (2 * (q || 0.7071)), c = Math.cos(w0);
    const a0 = 1 + alpha;
    const b0 = (1 + c) / 2 / a0, b1 = -(1 + c) / a0, b2 = b0, a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
    const y = new Float32Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      y[i] = yi; x2 = x1; x1 = xi; y2 = y1; y1 = yi;
    }
    return y;
  }

  function percentile(arr, p) {
    const a = Array.from(arr).filter(Number.isFinite).sort((a, b) => a - b);
    if (!a.length) return 0;
    return a[Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))))];
  }
  function median(a) { return percentile(a, 0.5); }

  // ---------- Stage 1: heavy analysis (done once per take) ----------
  const BAND_EDGES = [0, 150, 400, 1000, 2500, 5000, 1e9];

  function analyze(samples, sr) {
    // normalise level
    let peak = 0;
    for (let i = 0; i < samples.length; i++) { const a = Math.abs(samples[i]); if (a > peak) peak = a; }
    const g = peak > 0 ? 0.9 / peak : 1;
    const x = new Float32Array(samples.length);
    for (let i = 0; i < x.length; i++) x[i] = samples[i] * g;

    const N = 512, hop = Math.round(sr * 0.004); // ~4 ms hop
    const win = hann(N), fft = makeFFT(N);
    const nb = N / 2 + 1;
    const nFrames = Math.max(1, Math.floor((x.length - N) / hop) + 1);
    const flux = new Float32Array(nFrames);
    const energy = new Float32Array(nFrames);
    let prev = new Float32Array(nb), cur = new Float32Array(nb);
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let f = 0; f < nFrames; f++) {
      const o = f * hop;
      for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
      fft(re, im);
      let fl = 0, en = 0;
      for (let k = 0; k < nb; k++) {
        const p = re[k] * re[k] + im[k] * im[k];
        en += p;
        const l = Math.log1p(30 * Math.sqrt(p));
        cur[k] = l;
        const d = l - prev[k];
        if (d > 0) fl += d;
      }
      flux[f] = f === 0 ? 0 : fl;
      energy[f] = en;
      const t = prev; prev = cur; cur = t;
    }
    // normalise flux by robust max
    const fmax = percentile(flux, 0.98) || 1;
    for (let i = 0; i < nFrames; i++) flux[i] /= fmax;

    // short envelope for precise onset timing (1 ms smoothing)
    const mkEnv = (sig) => {
      const env = new Float32Array(sig.length);
      const a = Math.exp(-1 / (sr * 0.004));
      let e = 0;
      for (let i = 0; i < sig.length; i++) { const v = Math.abs(sig[i]); e = v > e ? v : a * e + (1 - a) * v; env[i] = e; }
      return env;
    };
    const env = mkEnv(x);
    const envHi = mkEnv(biquadHP(biquadHP(x, sr, 1500, 0.7071), sr, 1500, 0.7071));

    // ---- pitch track on low band (downsampled) ----
    const lp = biquadLP(biquadLP(x, sr, 420, 0.7071), sr, 420, 0.7071);
    const ds = Math.max(1, Math.round(sr / 5512));
    const sr2 = sr / ds;
    const y = new Float32Array(Math.floor(lp.length / ds));
    for (let i = 0; i < y.length; i++) y[i] = lp[i * ds];
    const W = Math.round(sr2 * 0.04);             // 40 ms analysis window
    const tauMin = Math.floor(sr2 / 500), tauMax = Math.ceil(sr2 / 38);
    const hop2 = Math.max(1, Math.round(sr2 * 0.008)); // 8 ms
    const nP = Math.max(0, Math.floor((y.length - W - tauMax) / hop2));
    const pTime = new Float32Array(nP), pMidi = new Float32Array(nP), pAp = new Float32Array(nP), pRms = new Float32Array(nP);
    const d = new Float64Array(tauMax + 2);
    for (let j = 0; j < nP; j++) {
      const o = j * hop2;
      let rms = 0;
      for (let i = 0; i < W; i++) rms += y[o + i] * y[o + i];
      rms = Math.sqrt(rms / W);
      d[0] = 1;
      let run = 0;
      for (let tau = 1; tau <= tauMax; tau++) {
        let s = 0;
        for (let i = 0; i < W; i++) { const v = y[o + i] - y[o + i + tau]; s += v * v; }
        run += s;
        d[tau] = run > 0 ? s * tau / run : 1;
      }
      let best = -1;
      for (let tau = tauMin; tau < tauMax; tau++) {
        if (d[tau] < 0.25) { while (tau + 1 < tauMax && d[tau + 1] < d[tau]) tau++; best = tau; break; }
      }
      if (best < 0) { // fall back to global min
        let mv = 1e9;
        for (let tau = tauMin; tau < tauMax; tau++) if (d[tau] < mv) { mv = d[tau]; best = tau; }
      }
      let tauF = best;
      if (best > 1 && best < tauMax) {
        const s0 = d[best - 1], s1 = d[best], s2 = d[best + 1];
        const den = s0 - 2 * s1 + s2;
        if (den !== 0) tauF = best + 0.5 * (s0 - s2) / den;
      }
      const f0 = sr2 / tauF;
      pTime[j] = (o + W / 2) / sr2;
      pMidi[j] = 69 + 12 * Math.log2(f0 / 440);
      pAp[j] = d[best];
      pRms[j] = rms;
    }

    return { sr, x, hop, N, flux, energy, env, envHi, duration: x.length / sr,
      pitch: { time: pTime, midi: pMidi, ap: pAp, rms: pRms, hop: hop2 / sr2, win: W / sr2, y, sr2 } };
  }

  // ---------- onset picking ----------
  function pickOnsets(A, sensitivity) {
    // sensitivity 0..1 (higher = more hits)
    const { flux, hop, sr, N, env, energy } = A;
    const n = flux.length;
    const delta = 0.32 - 0.26 * sensitivity;
    const halfMed = Math.round(0.12 * sr / hop);
    const enFloor = percentile(energy, 0.15) * 4 + 1e-9;
    const minGap = Math.round(0.045 * sr / hop);
    const out = [];
    for (let i = 2; i < n - 2; i++) {
      const v = flux[i];
      if (v < delta) continue;
      let isMax = true;
      for (let k = -3; k <= 3; k++) if (k && flux[i + k] > v) { isMax = false; break; }
      if (!isMax) continue;
      const lo = Math.max(0, i - halfMed), hi = Math.min(n, i + halfMed);
      const local = [];
      for (let k = lo; k < hi; k += 2) local.push(flux[k]);
      if (v < median(local) + delta) continue;
      const enMax = Math.max(energy[i], energy[i + 1] || 0, energy[i + 2] || 0);
      if (enMax < enFloor) continue;
      if (out.length && i - out[out.length - 1].frame < minGap) {
        if (v > out[out.length - 1].strength) out[out.length - 1] = { frame: i, strength: v };
        continue;
      }
      out.push({ frame: i, strength: v });
    }
    // refine timing: sharpest relative rise of the full or high-band envelope near the frame
    const { envHi } = A;
    const lag = Math.round(0.004 * sr), lead = Math.round(0.0015 * sr);
    for (const o of out) {
      const c = o.frame * hop + N / 2;
      const s0 = Math.max(lag + 1, Math.round(c - 0.04 * sr)), s1 = Math.min(env.length - lead - 1, Math.round(c + 0.012 * sr));
      let best = s0, bs = -1e9;
      for (const E of [env, envHi]) {
        let pk = 1e-9;
        for (let s = s0; s < s1; s++) if (E[s] > pk) pk = E[s];
        for (let s = s0; s < s1; s++) {
          const after = E[s + lead];
          if (after < 0.15 * pk) continue;
          const r = Math.log((after + 1e-4) / (E[s - lag] + 1e-4)) * Math.sqrt(after / pk);
          if (r > bs) { bs = r; best = s; }
        }
      }
      o.sample = best; o.t = best / sr;
    }
    return out;
  }

  // ---------- per-hit features ----------
  let _fftF = null, _winF = null;
  function features(A, sample, strength) {
    const { x, sr } = A;
    const N = 1024;
    if (!_fftF) { _fftF = makeFFT(N); _winF = hann(N); }
    const L = Math.round(0.032 * sr);
    const spec = (start) => {
      const re = new Float64Array(N), im = new Float64Array(N);
      for (let i = 0; i < L; i++) {
        const w = 0.5 - 0.5 * Math.cos(Math.PI * 2 * i / (L - 1));
        const s = start + i;
        re[i] = (s >= 0 && s < x.length ? x[s] : 0) * w;
      }
      _fftF(re, im);
      const p = new Float64Array(N / 2 + 1);
      for (let k = 0; k <= N / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k];
      return p;
    };
    // energy that is NEW at the hit: attack spectrum minus what was already sounding
    const pre = spec(sample - Math.round(0.036 * sr));
    const pA = spec(sample - Math.round(0.002 * sr));
    const pB = spec(sample + Math.round(0.05 * sr));
    const bands = new Float64Array(BAND_EDGES.length - 1);
    let tot = 1e-12, cen = 0, logSum = 0, lin = 0, cnt = 0, totB = 1e-12;
    let up = 1e-12, upCen = 0, upB = 1e-12, hi = 1e-12, lo = 1e-12, loB = 1e-12;
    for (let k = 1; k <= N / 2; k++) {
      const p = Math.max(0, pA[k] - 1.5 * pre[k]) + 1e-12;
      const pb = Math.max(0, pB[k] - 1.5 * pre[k]);
      totB += pb;
      const f = k * sr / N;
      if (f >= 400) { up += p; upCen += p * f; upB += pb; }
      if (f >= 1000) hi += p; else if (f < 400) { lo += p; loB += pb; }
      for (let b = 0; b < bands.length; b++) if (f >= BAND_EDGES[b] && f < BAND_EDGES[b + 1]) { bands[b] += p; break; }
      tot += p; cen += p * f;
      logSum += Math.log(p + 1e-9); lin += p + 1e-9; cnt++;
    }
    const centroid = cen / tot;
    const flat = Math.exp(logSum / cnt) / (lin / cnt);
    const eA = Math.sqrt(tot) + 1e-6, eB = Math.sqrt(totB) + 1e-6;
    const upper = { cen: upCen / up, decay: Math.sqrt(upB / up), level: Math.sqrt(hi) };
    // periodicity of low band ~25-65 ms after the hit
    const P = A.pitch;
    let ap = 1, j0 = Math.floor((sample / sr - P.win / 2 + 0.03) / P.hop);
    for (let j = Math.max(0, j0); j < Math.min(P.ap.length, j0 + 4); j++) ap = Math.min(ap, P.ap[j]);
    const f = [];
    for (let b = 0; b < bands.length; b++) f.push(Math.log10(bands[b] / tot + 1e-6));
    f.push(Math.log2(centroid));
    f.push(Math.log10(flat + 1e-9));
    f.push(Math.log2(eB / eA));
    f.push(ap);
    const loEps = 1e-4 * (tot + totB) + 1e-9;
    const lowDecay = Math.min(4, Math.sqrt((loB + loEps) / (lo + loEps)));
    f.push(Math.log2(lowDecay + 0.02));
    return { vec: f, level: eA, upper, lowLevel: Math.sqrt(lo), lowDecay, strength: strength || 1, centroid, low: (bands[0] + bands[1]) / tot, high: (bands[4] + bands[5]) / tot, decay: eB / eA, ap };
  }

  // ---------- classifier ----------
  function upperClass(u, model) {
    if (model && model.up) {
      const z = [Math.log2(u.cen), Math.log2(u.decay + 0.02)];
      const d = c => { const m = model.up[c]; return ((z[0] - m[0]) / 0.5) ** 2 + ((z[1] - m[1]) / 1.5) ** 2; };
      return d('hat') <= d('snare') ? 'hat' : 'snare';
    }
    return (u.cen > 5500 && u.decay < 0.25) || u.cen > 7500 ? 'hat' : 'snare';
  }
  function kickOrBass(ft) {
    return ft.ap < 0.3 && ft.lowDecay > 0.7 && (ft.relLow ?? 0) < 0.6 ? 'bass' : 'kick';
  }
  function defaultClass(ft) {
    if (ft.low > 0.55) return kickOrBass(ft);
    return upperClass(ft.upper, null);
  }

  // Untaught: if the take is mostly clicks (little low thump), sort the clicks into three
  // colours by k-means and name them darkest = kick, middle = snare, brightest = hat.
  function autoSort(raw) {
    const hits = raw.filter(o => o.cls !== 'bass');
    if (hits.length < 12) return;
    const kicks = hits.filter(o => o.cls === 'kick').length;
    if (kicks / hits.length > 0.2) return; // a real low kick exists: trust the default rules
    const X = hits.map(o => {
      const v = o.ft.vec; // first 6 entries: log band shares
      return [v[1], v[2], v[3], v[4], v[5], v[6] * 0.7, Math.log2(o.ft.upper.decay + 0.02) * 0.4];
    });
    const dim = X[0].length, mean = new Array(dim).fill(0), sd = new Array(dim).fill(0);
    for (const x of X) for (let i = 0; i < dim; i++) mean[i] += x[i] / X.length;
    for (const x of X) for (let i = 0; i < dim; i++) sd[i] += (x[i] - mean[i]) ** 2 / X.length;
    const Z = X.map(x => x.map((v, i) => (v - mean[i]) / (Math.sqrt(sd[i]) + 1e-6)));
    // deterministic k-means++-ish init: by brightness percentiles
    const bright = hits.map(o => Math.log2(o.ft.upper.cen));
    const order = bright.map((b, i) => [b, i]).sort((a, b) => a[0] - b[0]);
    let C = [0.15, 0.5, 0.85].map(p => Z[order[Math.floor(p * (order.length - 1))][1]].slice());
    let lab = new Array(Z.length).fill(0);
    for (let it = 0; it < 25; it++) {
      lab = Z.map(z => { let b = 0, bd = 1e9; C.forEach((c, k) => { const d = c.reduce((s, v, i) => s + (v - z[i]) ** 2, 0); if (d < bd) { bd = d; b = k; } }); return b; });
      C = C.map((c, k) => { const m = Z.filter((_, i) => lab[i] === k); if (!m.length) return c; return c.map((_, j) => m.reduce((s, z) => s + z[j], 0) / m.length); });
    }
    const cb = [0, 1, 2].map(k => { const v = hits.map((o, i) => lab[i] === k ? bright[i] : null).filter(v => v != null); return v.length ? median(v) : 0; });
    const rank = [0, 1, 2].sort((a, b) => cb[a] - cb[b]);
    const name = {}; name[rank[0]] = 'kick'; name[rank[1]] = 'snare'; name[rank[2]] = 'hat';
    hits.forEach((o, i) => { o.cls = name[lab[i]]; o.auto = true; });
  }

  function trainModel(calib) {
    // calib: {kick:[vec], snare:[vec], hat:[vec], bass:[vec]}
    const all = [];
    for (const c of CLASSES) for (const s of (calib[c] || [])) all.push({ c, v: s.vec, u: s.upper, h: s.hiLevel });
    const classesWithData = CLASSES.filter(c => (calib[c] || []).length >= 2);
    if (classesWithData.length < 2) return null;
    const dim = all[0].v.length;
    const mean = new Array(dim).fill(0), sd = new Array(dim).fill(0);
    for (const s of all) for (let i = 0; i < dim; i++) mean[i] += s.v[i] / all.length;
    for (const s of all) for (let i = 0; i < dim; i++) sd[i] += (s.v[i] - mean[i]) ** 2 / all.length;
    for (let i = 0; i < dim; i++) sd[i] = Math.sqrt(sd[i]) + 1e-3;
    const pts = all.map(s => ({ c: s.c, z: s.v.map((v, i) => (v - mean[i]) / sd[i]) }));
    let up = null;
    const hs = all.filter(s => s.c === 'hat' && s.u), ss = all.filter(s => s.c === 'snare' && s.u);
    if (hs.length >= 2 && ss.length >= 2) {
      const m = arr => [median(arr.map(s => Math.log2(s.u.cen))), median(arr.map(s => Math.log2(s.u.decay + 0.02)))];
      up = { hat: m(hs), snare: m(ss) };
    }
    return { mean, sd, pts, classes: classesWithData, up };
  }

  function classify(model, ft) {
    if (!model) return defaultClass(ft);
    const c = knn(model, ft);
    // low-end hits overlap with sustained bass a lot; settle kick vs bass on the physics
    if (c === 'kick' || c === 'bass') return kickOrBass(ft);
    return c;
  }
  function knn(model, ft) {
    const z = ft.vec.map((v, i) => (v - model.mean[i]) / model.sd[i]);
    const ds = model.pts.map(p => ({ c: p.c, d: p.z.reduce((s, v, i) => s + (v - z[i]) ** 2, 0) }));
    ds.sort((a, b) => a.d - b.d);
    const k = Math.min(5, ds.length), votes = {};
    for (let i = 0; i < k; i++) votes[ds[i].c] = (votes[ds[i].c] || 0) + 1 / (ds[i].d + 0.5);
    let best = null, bv = -1;
    for (const c in votes) if (votes[c] > bv) { bv = votes[c]; best = c; }
    return best;
  }

  // ---------- bass notes ----------
  function bassNotes(A, opts, onsets) {
    const P = A.pitch;
    const n = P.time.length;
    const rmsRef = percentile(P.rms, 0.95) || 1;
    const gate = rmsRef * (0.25 - 0.2 * (opts.bassSensitivity ?? 0.5));
    const voiced = new Uint8Array(n);
    for (let j = 0; j < n; j++) voiced[j] = P.ap[j] < 0.3 && P.rms[j] > gate && P.midi[j] > 20 && P.midi[j] < 80 ? 1 : 0;
    const splitTimes = onsets.filter(o => o.cls === 'bass').map(o => o.t);
    const kickTimes = onsets.filter(o => o.cls === 'kick').map(o => o.t);
    const notes = [];
    let cur = null, gap = 0;
    const maxGap = Math.round(0.035 / P.hop);
    const close = () => {
      if (cur && cur.frames.length) {
        const ms = cur.frames.map(j => P.midi[j]);
        const m = median(ms);
        const start = P.time[cur.frames[0]] - P.win / 2 + 0.006;
        const end = P.time[cur.frames[cur.frames.length - 1]] + P.win / 2 - 0.012;
        let peak = 0; for (const j of cur.frames) peak = Math.max(peak, P.rms[j]);
        notes.push({ start, end, pitch: m, level: peak });
      }
      cur = null; gap = 0;
    };
    let si = 0, ki = 0;
    for (let j = 0; j < n; j++) {
      const t = P.time[j];
      // re-articulation split
      while (si < splitTimes.length && splitTimes[si] < t - P.win / 2) {
        const st = splitTimes[si++];
        if (cur && st - (P.time[cur.frames[0]] - P.win / 2) > 0.06) close();
      }
      while (ki < kickTimes.length && kickTimes[ki] < t - P.win / 2 - 0.05) ki++;
      if (ki < kickTimes.length && t - P.win / 2 > kickTimes[ki] - 0.005 && t - P.win / 2 < kickTimes[ki] + 0.05) continue; // kick thump: ignore
      if (!voiced[j]) { if (cur && ++gap > maxGap) close(); continue; }
      const m = P.midi[j];
      if (!cur) { cur = { frames: [j] }; gap = 0; continue; }
      const recent = cur.frames.slice(-4).map(k => P.midi[k]);
      const ref = median(recent);
      if (Math.abs(m - ref) < 0.6) { cur.frames.push(j); gap = 0; continue; }
      // look ahead: is this a new stable pitch?
      let agree = 0;
      for (let k = j + 1; k <= j + 2 && k < n; k++) if (voiced[k] && Math.abs(P.midi[k] - m) < 0.6) agree++;
      if (agree >= 2) { close(); cur = { frames: [j] }; }
      // otherwise treat as glitch
    }
    close();
    // snap starts to nearby onsets, apply filters
    const allOn = onsets.map(o => ({ t: o.t, k: o.cls === 'kick' }));
    const out = [];
    const shift = 12 * (opts.bassOctave || 0);
    for (const nt of notes) {
      let best = null, bd = 1;
      for (const o of allOn) {
        const dd = nt.start - o.t; // positive: onset before note
        let win = 0.075;
        if (o.k && dd > 0.05) {
          // kick right before the note: snap only if low-end energy never dipped between kick and note
          let mn = Infinity;
          for (let j = 0; j < n; j++) {
            const tc = P.time[j];
            if (tc < o.t + 0.03 || tc > nt.start + P.win / 2) continue;
            mn = Math.min(mn, P.rms[j]);
          }
          if (mn === Infinity || mn > 0.45 * nt.level) win = 0.13;
        }
        if (dd < win && dd > -0.025 && Math.abs(dd) < bd) { bd = Math.abs(dd); best = o.t; }
      }
      if (best != null) nt.start = best;
      const dur = nt.end - nt.start;
      if (dur < 0.06) continue;
      const nearKick = kickTimes.some(t => Math.abs(t - nt.start) < 0.03);
      if (nearKick && dur < 0.13) continue;
      out.push({ t: nt.start, dur, note: Math.round(nt.pitch) + shift, cents: Math.round((nt.pitch - Math.round(nt.pitch)) * 100), level: nt.level });
    }
    // fix overlaps
    out.sort((a, b) => a.t - b.t);
    for (let i = 0; i + 1 < out.length; i++) if (out[i].t + out[i].dur > out[i + 1].t) out[i].dur = Math.max(0.0, out[i + 1].t - out[i].t - 0.005);
    for (let i = out.length - 1; i >= 0; i--) if (out[i].dur < 0.045) out.splice(i, 1);
    const ref = percentile(out.map(o => o.level), 0.9) || 1;
    for (const o of out) o.vel = Math.max(40, Math.min(127, Math.round(100 + 40 * Math.log10(o.level / ref))));
    return out;
  }

  // ---------- full groove from analysis ----------
  function buildGroove(A, model, opts) {
    const raw = pickOnsets(A, opts.drumSensitivity ?? 0.5);
    for (const o of raw) o.ft = features(A, o.sample, o.strength);
    const lowRef = percentile(raw.map(o => o.ft.lowLevel), 0.9) || 1;
    for (const o of raw) { o.ft.relLow = o.ft.lowLevel / lowRef; o.cls = classify(model, o.ft); }
    if (!model && opts.autoSort !== false) autoSort(raw);
    // overlapping hits: a bass or kick onset can also carry a hat or snare on top
    const ref = c => median(raw.filter(o => o.cls === c).map(o => o.ft.upper.level));
    const hsRef = Math.min(ref('hat') || Infinity, ref('snare') || Infinity);
    const kickRef = ref('kick');
    const thr = Math.max(0.6 * (hsRef || 0), 3 * (kickRef || 0)) * (1.3 - 0.6 * (opts.drumSensitivity ?? 0.5));
    const extra = [];
    if (opts.stack !== false && hsRef > 0 && hsRef < Infinity) for (const o of raw) {
      if ((o.cls === 'bass' || o.cls === 'kick') && o.ft.upper.level > thr) {
        extra.push({ t: o.t, sample: o.sample, strength: o.strength, ft: o.ft, cls: upperClass(o.ft.upper, model), layered: true });
      }
    }
    const bass = bassNotes(A, opts, raw);
    let drums = raw.filter(o => o.cls !== 'bass').concat(extra).sort((a, b) => a.t - b.t);
    // a mouth click often has a second, weaker transient right after it: fold it into one hit
    const mergeWin = opts.mergeMs == null ? 0.065 : opts.mergeMs / 1000;
    if (mergeWin > 0) {
      const kept = [];
      for (const d of drums) {
        const prev = kept.length ? kept[kept.length - 1] : null;
        if (prev && d.t - prev.t < mergeWin && !d.layered && !prev.layered) {
          if (d.ft.level > prev.ft.level * 2) kept[kept.length - 1] = d;
          continue;
        }
        kept.push(d);
      }
      drums = kept;
    }
    const lv = {};
    const lev = d => d.layered || d.cls !== 'kick' ? d.ft.upper.level : d.ft.level;
    for (const c of ['kick', 'snare', 'hat']) lv[c] = percentile(drums.filter(d => d.cls === c).map(lev), 0.9) || 1;
    for (const d of drums) d.vel = Math.max(35, Math.min(127, Math.round(105 + 45 * Math.log10(lev(d) / lv[d.cls]))));
    return { drums: drums.map(d => ({ t: d.t, cls: d.cls, vel: d.vel, ft: d.ft, layered: !!d.layered })), bass, onsets: raw };
  }

  // ---------- bass cleanup: merge slides, snap to a key ----------
  const MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  function detectKey(bass) {
    const h = new Array(12).fill(0);
    for (const b of bass) h[((b.note % 12) + 12) % 12] += Math.min(0.6, b.dur) * (b.vel || 100);
    const corr = (prof, k) => { let mx = 0, my = 0; for (let i = 0; i < 12; i++) { mx += h[i] / 12; my += prof[i] / 12; }
      let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < 12; i++) { const x = h[(i + k) % 12] - mx, y = prof[i] - my; sxy += x * y; sxx += x * x; syy += y * y; } return sxy / Math.sqrt(sxx * syy + 1e-12); };
    let best = { tonic: 0, minor: true, r: -2 };
    for (let k = 0; k < 12; k++) for (const minor of [false, true]) { const r = corr(minor ? MIN : MAJ, k); if (r > best.r) best = { tonic: k, minor, r }; }
    return best;
  }
  function scaleOf(key) {
    const steps = key.minor ? [0, 2, 3, 5, 7, 8, 10] : [0, 2, 4, 5, 7, 9, 11];
    return steps.map(x => (x + key.tonic) % 12);
  }
  function cleanBass(bass, opts) {
    let out = bass.map(b => Object.assign({}, b, { raw: b.note + (b.cents || 0) / 100 }));
    out.sort((a, b) => a.t - b.t);
    // merge slides: a note that runs straight into a neighbour within a semitone is one note
    if (opts.merge !== false) {
      const m = [];
      for (const b of out) {
        const p = m[m.length - 1];
        if (p && b.t - (p.t + p.dur) < 0.04 && Math.abs(b.raw - p.raw) <= 1.3) {
          const end = Math.max(p.t + p.dur, b.t + b.dur);
          if (b.dur > p.dur * 1.5) { p.note = b.note; p.raw = b.raw; }
          p.dur = end - p.t; p.vel = Math.max(p.vel, b.vel); p.level = Math.max(p.level, b.level);
          continue;
        }
        m.push(b);
      }
      out = m;
    }
    let key = null;
    if (opts.key && opts.key !== 'off' && out.length) {
      key = opts.key === 'auto' ? detectKey(out) : opts.key;
      const sc = scaleOf(key);
      for (const b of out) {
        const pc = ((b.note % 12) + 12) % 12;
        if (sc.includes(pc)) continue;
        // out of key: move to the in-key neighbour nearest the sung pitch
        const up = b.note + 1, dn = b.note - 1;
        const upOk = sc.includes(((up % 12) + 12) % 12), dnOk = sc.includes(((dn % 12) + 12) % 12);
        if (upOk && dnOk) b.note = (b.raw >= b.note) ? up : dn;
        else b.note = upOk ? up : dn;
      }
      // after snapping, neighbours may now be the same note: merge again
      if (opts.merge !== false) {
        const m = [];
        for (const b of out) { const p = m[m.length - 1]; if (p && p.note === b.note && b.t - (p.t + p.dur) < 0.04) { p.dur = b.t + b.dur - p.t; continue; } m.push(b); }
        out = m;
      }
    }
    return { bass: out, key };
  }

  // ---------- how well do the taught sounds separate? (leave-one-out) ----------
  function separation(calib) {
    const out = {};
    const cls = ['kick', 'snare', 'hat', 'bass'].filter(c => (calib[c] || []).length);
    for (const c of cls) out[c] = { n: 0, ok: 0, as: {} };
    for (const c of cls) {
      calib[c].forEach((h, i) => {
        const rest = {}; for (const d of cls) rest[d] = calib[d].filter((_, j) => !(d === c && j === i));
        const m = trainModel(rest); if (!m) return;
        const got = knn(m, { vec: h.vec });
        out[c].n++; if (got === c) out[c].ok++; else out[c].as[got] = (out[c].as[got] || 0) + 1;
      });
    }
    return out;
  }

  // ---------- calibration: isolated hits of one class ----------
  function calibFromClip(samples, sr) {
    const A = analyze(samples, sr);
    const on = pickOnsets(A, 0.8);
    const mx = Math.max(0, ...on.map(o => o.strength));
    // isolated hits: keep strong onsets, at least 150 ms apart (strongest wins)
    const strong = on.filter(o => o.strength >= 0.35 * mx).sort((a, b) => b.strength - a.strength);
    const kept = [];
    for (const o of strong) if (kept.every(k => Math.abs(k.t - o.t) > 0.15)) kept.push(o);
    kept.sort((a, b) => a.t - b.t);
    const out = kept.map(o => { const ft = features(A, o.sample, o.strength); return { vec: ft.vec, upper: ft.upper, t: o.t }; });
    return { hits: out, analysis: A };
  }

  // ---------- tempo ----------
  function estimateTempo(A, groove) {
    const hopS = 0.01;
    const n = Math.ceil(A.duration / hopS) + 1;
    const env = new Float32Array(n);
    const add = (t, w) => {
      const c = t / hopS;
      for (let k = -3; k <= 3; k++) { const i = Math.round(c) + k; if (i >= 0 && i < n) env[i] += w * Math.exp(-0.5 * ((i - c) / 1.2) ** 2); }
    };
    for (const d of groove.drums) add(d.t, d.cls === 'kick' ? 1.5 : d.cls === 'snare' ? 1.2 : 0.7);
    for (const b of groove.bass) add(b.t, 0.8);
    let bestBpm = 100, bestScore = -1;
    for (let bpm = 60; bpm <= 190; bpm += 0.5) {
      const lag = 60 / bpm / hopS;
      let s = 0;
      for (const mult of [1, 2, 0.5, 4]) {
        const L = lag * mult, wgt = mult === 1 ? 1 : mult === 2 ? 0.6 : mult === 0.5 ? 0.5 : 0.3;
        const Li = Math.floor(L), fr = L - Li;
        if (Li < 2 || Li >= n) continue;
        let acc = 0;
        for (let i = 0; i + Li + 1 < n; i++) acc += env[i] * (env[i + Li] * (1 - fr) + env[i + Li + 1] * fr);
        s += wgt * acc;
      }
      const prior = Math.exp(-0.5 * (Math.log2(bpm / 105) / 0.6) ** 2);
      s *= prior;
      if (s > bestScore) { bestScore = s; bestBpm = bpm; }
    }
    return bestBpm;
  }

  // grid phase + downbeat
  function alignGrid(groove, bpm, hint) {
    const P = 60 / bpm;
    const ev = [];
    for (const d of groove.drums) ev.push({ t: d.t, w: d.cls === 'hat' ? 0.5 : 1, bw: d.cls === 'hat' ? 0.3 : 1, kick: d.cls === 'kick', snare: d.cls === 'snare' });
    for (const b of groove.bass) ev.push({ t: b.t, w: 0.6, bw: 0.3 });
    if (!ev.length) return 0;
    let best = 0, bs = -1e9;
    const lo = hint != null ? hint - 0.06 : 0, hi = hint != null ? hint + 0.06 : P;
    // 1) fine phase on the 16th/8th lattice
    for (let ph = lo; ph < hi; ph += P / 192) {
      let s = 0;
      for (const e of ev) {
        s += e.w * Math.cos(2 * Math.PI * (e.t - ph) / (P / 4));
        s += 0.8 * e.w * Math.cos(2 * Math.PI * (e.t - ph) / (P / 2));
      }
      if (s > bs) { bs = s; best = ph; }
    }
    // 2) which of the 8th positions is the beat: snares and kicks mostly land on beats
    if (hint == null) {
      let bb = best, bbs = -1e9;
      for (const cand of [best, best + P / 2]) {
        let s = 0;
        for (const e of ev) {
          const near = Math.abs(((e.t - cand) / P) - Math.round((e.t - cand) / P)) < 0.12;
          if (near) s += e.snare ? 1.5 : e.kick ? 1 : 0.25;
        }
        if (s > bbs) { bbs = s; bb = cand; }
      }
      best = bb;
    }
    // downbeat: choose which beat of the bar gets most kick weight minus snare weight
    let db = 0, dbs = -1e9;
    for (let k = 0; k < 4; k++) {
      const off = best + k * P;
      let s = 0;
      for (const e of ev) {
        const pos = (((e.t - off) / P) % 4 + 4) % 4;
        const nb = Math.round(pos) % 4, dist = Math.abs(pos - Math.round(pos));
        if (dist > 0.12) continue;
        if (e.kick && nb === 0) s += 2; if (e.kick && nb === 2) s += 0.7;
        if (e.snare && (nb === 1 || nb === 3)) s += 1.5; if (e.snare && nb === 0) s -= 1;
      }
      if (s > dbs) { dbs = s; db = k; }
    }
    if (hint != null) db = 0;
    let off = best + db * P;
    const first = Math.min(...ev.map(e => e.t));
    const bar = 4 * P;
    // bar 1 starts at the last downbeat at or before the first event (allow small pickup tolerance)
    while (off > first + P / 8) off -= bar;
    while (off + bar <= first - P / 8) off += bar;
    return off;
  }

  // ---------- grid maps: 16th-note index <-> seconds ----------
  // map = { k0, times: Float64Array } where times[i] is the time of 16th index k0 + i; index 0 = bar 1 downbeat
  function fixedMap(bpm, offset, duration) {
    const g = 60 / bpm / 4;
    const k0 = Math.floor((-2 - offset) / g), k1 = Math.ceil((duration + 2 - offset) / g);
    const times = new Float64Array(k1 - k0 + 1);
    for (let i = 0; i < times.length; i++) times[i] = offset + (k0 + i) * g;
    return { k0, times, bpm, mode: 'fixed' };
  }
  function uOf(map, t) {
    const T = map.times, n = T.length;
    if (t <= T[0]) return map.k0 + (t - T[0]) / (T[1] - T[0]);
    if (t >= T[n - 1]) return map.k0 + n - 1 + (t - T[n - 1]) / (T[n - 1] - T[n - 2]);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
    return map.k0 + lo + (t - T[lo]) / (T[hi] - T[lo]);
  }
  function tOf(map, u) {
    const T = map.times, n = T.length, i = u - map.k0;
    if (i <= 0) return T[0] + i * (T[1] - T[0]);
    if (i >= n - 1) return T[n - 1] + (i - n + 1) * (T[n - 1] - T[n - 2]);
    const f = Math.floor(i); return T[f] + (i - f) * (T[f + 1] - T[f]);
  }

  // Follow a drifting human tempo: phase-locked 16th grid, then smoothed into a tempo map
  function trackGrid(groove, bpm, duration) {
    let ev = [];
    for (const d of groove.drums) ev.push({ t: d.t, w: d.cls === 'hat' ? 0.6 : 1, cls: d.cls });
    for (const b of groove.bass) ev.push({ t: b.t, w: 0.5, cls: 'bass' });
    ev.sort((a, b) => a.t - b.t);
    const m = [];
    for (const e of ev) { if (m.length && e.t - m[m.length - 1].t < 0.06) { if (e.w > m[m.length - 1].w) m[m.length - 1] = e; continue; } m.push(e); }
    ev = m;
    if (ev.length < 8) return fixedMap(bpm, ev.length ? ev[0].t : 0, duration);
    const g0 = 60 / bpm / 4;
    // initial phase from the first few seconds
    const first = ev.filter(e => e.t < ev[0].t + 16 * g0);
    let ph = ev[0].t, bs = -1e9;
    for (let p = ev[0].t - g0; p < ev[0].t; p += g0 / 60) {
      let sc = 0; for (const e of first) sc += e.w * Math.cos(2 * Math.PI * (e.t - p) / g0);
      if (sc > bs) { bs = sc; ph = p; }
    }
    // PLL pass
    let g = g0, k = 0, tau = ph;
    const asg = [];
    for (const e of ev) {
      const dk = Math.round((e.t - tau) / g), kk = k + dk;
      const pred = tau + dk * g, err = e.t - pred;
      if (Math.abs(err) < 0.3 * g) {
        tau = pred + 0.35 * err; g = Math.max(g0 * 0.9, Math.min(g0 * 1.1, g + 0.04 * err)); asg.push({ k: kk, t: e.t, w: e.w, ok: true });
      } else { tau = pred; asg.push({ k: kk, t: e.t, w: e.w, ok: false }); }
      k = kk;
    }
    const fit = (A) => {
      const kmin = A[0].k - 8, kmax = A[A.length - 1].k + 8;
      const times = new Float64Array(kmax - kmin + 1);
      const H = 14; // half window in 16ths (about 3.5 beats)
      let j0 = 0;
      for (let kk = kmin; kk <= kmax; kk++) {
        while (j0 < A.length && A[j0].k < kk - H) j0++;
        let sw = 0, sk = 0, st = 0, skk = 0, skt = 0;
        for (let j = j0; j < A.length && A[j].k <= kk + H; j++) {
          if (!A[j].ok) continue;
          const d = Math.abs(A[j].k - kk) / (H + 1), w = A[j].w * (1 - d * d * d) ** 3;
          sw += w; sk += w * A[j].k; st += w * A[j].t; skk += w * A[j].k * A[j].k; skt += w * A[j].k * A[j].t;
        }
        let val = NaN;
        if (sw > 0.5) {
          const mk = sk / sw, mt = st / sw, vk = skk / sw - mk * mk;
          const slope = vk > 1 ? (skt / sw - mk * mt) / vk : g0;
          val = mt + Math.max(g0 * 0.85, Math.min(g0 * 1.15, slope)) * (kk - mk);
        }
        times[kk - kmin] = val;
      }
      // fill gaps and force monotonic spacing
      for (let i = 0; i < times.length; i++) if (!Number.isFinite(times[i])) times[i] = i ? times[i - 1] + g0 : NaN;
      for (let i = times.length - 1; i >= 0; i--) if (!Number.isFinite(times[i])) times[i] = times[i + 1] - g0;
      for (let i = 1; i < times.length; i++) if (times[i] < times[i - 1] + 0.6 * g0) times[i] = times[i - 1] + 0.6 * g0;
      return { k0: kmin, times };
    };
    let map = fit(asg);
    // second pass: reassign every event to its nearest line on the smoothed map
    for (let pass = 0; pass < 2; pass++) {
      for (const a of asg) { const u = uOf(map, a.t); a.k = Math.round(u); a.ok = Math.abs(u - a.k) < 0.3; }
      asg.sort((a, b) => a.k - b.k || a.t - b.t);
      map = fit(asg);
    }
    // bar 1: pick the 16th offset that puts kicks on 1 and snares on 2 and 4
    const evk = ev.map(e => ({ cls: e.cls, w: e.w, k: Math.round(uOf(map, e.t)) }));
    let bestC = 0, bc = -1e9;
    for (let c = 0; c < 16; c++) {
      let sc = 0;
      for (const e of evk) {
        const p = (((e.k - c) % 16) + 16) % 16;
        if (p % 4 === 0) sc += e.cls === 'kick' ? 1.2 : e.cls === 'snare' ? 1 : 0.3;
        if (e.cls === 'kick' && p === 0) sc += 1.5; if (e.cls === 'kick' && p === 8) sc += 0.6;
        if (e.cls === 'snare' && (p === 4 || p === 12)) sc += 1.2; if (e.cls === 'snare' && p === 0) sc -= 0.8;
      }
      if (sc > bc) { bc = sc; bestC = c; }
    }
    const firstK = Math.min(...evk.map(e => e.k));
    let shift = bestC;
    while (firstK - shift < -2) shift -= 16;
    while (firstK - shift >= 14) shift += 16;
    map.k0 -= shift;
    const span = map.times[map.times.length - 9] - map.times[8], nk = map.times.length - 17;
    map.bpm = Math.round(60 / (4 * span / nk) * 10) / 10;
    map.mode = 'follow';
    return map;
  }
  function shiftMap(map, sixteenths) { return Object.assign({}, map, { k0: map.k0 - sixteenths }); }

  function quantizeGroove(groove, map, gridDiv, strength, swing, opts) {
    const step = 16 / gridDiv; // in 16ths
    const qu = u => {
      const k = Math.round(u / step);
      let target = k * step;
      if (swing && (gridDiv === 16 || gridDiv === 8) && (k % 2 !== 0)) target += step * swing * 0.5;
      return u + strength * (target - u);
    };
    let drums = groove.drums.map(d => { const u = uOf(map, d.t), q = qu(u); return Object.assign({}, d, { u, qu: q, qt: tOf(map, q) }); });
    if (opts && opts.onePerStep && strength > 0.5) {
      // keep the loudest hit in each grid step; the others become 'off' so they can be restored by tapping
      const best = new Map();
      for (const d of drums) { if (d.cls === 'off') continue; const k = Math.round(d.u / step); const b = best.get(k); if (!b || (d.edited && !b.edited) || (d.edited === b.edited && d.vel > b.vel)) best.set(k, d); }
      drums = drums.map(d => d.cls === 'off' || d.edited || best.get(Math.round(d.u / step)) === d ? d : Object.assign({}, d, { cls: 'off', dup: true }));
    }
    const bass = groove.bass.map(b => {
      const u = uOf(map, b.t), ue = uOf(map, b.t + b.dur);
      const s = qu(u); let e = qu(ue);
      const minLen = strength > 0.5 ? step * 0.5 : 0.25;
      if (e - s < minLen) e = s + Math.max(minLen, (ue - u) * (1 - strength) + step * strength);
      const qt = tOf(map, s);
      return Object.assign({}, b, { u, qu: s, que: e, qt, qdur: Math.max(0.03, tOf(map, e) - qt) });
    });
    bass.sort((a, b) => a.qu - b.qu || b.dur - a.dur);
    for (let i = bass.length - 1; i > 0; i--) if (Math.abs(bass[i].qu - bass[i - 1].qu) < 1e-6) bass.splice(i, 1); // same start: keep the longer
    for (let i = 0; i + 1 < bass.length; i++) if (bass[i].que > bass[i + 1].qu) { bass[i].que = bass[i + 1].qu; bass[i].qdur = Math.max(0.03, tOf(map, bass[i].que) - bass[i].qt); }
    return { drums, bass };
  }

  // ---------- MIDI file writer ----------
  function vlq(n) { const b = [n & 0x7f]; while ((n >>= 7)) b.unshift((n & 0x7f) | 0x80); return b; }
  function trackChunk(events) { // events: [{tick, bytes}]
    events.sort((a, b) => a.tick - b.tick || a.order - b.order);
    const data = []; let last = 0;
    for (const e of events) { data.push(...vlq(Math.max(0, e.tick - last)), ...e.bytes); last = Math.max(last, e.tick); }
    data.push(0, 0xff, 0x2f, 0);
    const len = data.length;
    return [0x4d, 0x54, 0x72, 0x6b, (len >>> 24) & 255, (len >>> 16) & 255, (len >>> 8) & 255, len & 255, ...data];
  }
  function textMeta(type, s) { const b = Array.from(s).map(c => c.charCodeAt(0) & 0x7f); return [0xff, type, ...vlq(b.length), ...b]; }

  function writeMidi(q, bpm, opts) {
    const PPQ = 480;
    const minU = Math.min(0, ...q.bass.map(b => b.qu), ...q.drums.map(d => d.qu));
    const lead = minU < 0 ? Math.ceil(-minU / 16) * 16 : 0; // pickup: start the file a bar early
    const tickU = u => Math.max(0, Math.round((u + lead) * PPQ / 4));
    const which = opts.which || 'both';
    const tracks = [];
    const us = Math.round(60e6 / bpm);
    tracks.push(trackChunk([
      { tick: 0, order: 0, bytes: textMeta(0x03, opts.title || 'Mouth Groove') },
      { tick: 0, order: 1, bytes: [0xff, 0x51, 3, (us >> 16) & 255, (us >> 8) & 255, us & 255] },
      { tick: 0, order: 2, bytes: [0xff, 0x58, 4, 4, 2, 24, 8] },
    ]));
    if (which !== 'drums') {
      const ev = [
        { tick: 0, order: 0, bytes: textMeta(0x03, 'Mouth Bass') },
        { tick: 0, order: 1, bytes: [0xc0, opts.program ?? 33] },
      ];
      for (const b of q.bass) {
        const s = tickU(b.qu), e = Math.max(s + 1, tickU(b.que));
        const n = Math.max(0, Math.min(127, b.note));
        ev.push({ tick: s, order: 3, bytes: [0x90, n, b.vel] });
        ev.push({ tick: e, order: 2, bytes: [0x80, n, 0] });
      }
      tracks.push(trackChunk(ev));
    }
    if (which !== 'bass') {
      const ev = [{ tick: 0, order: 0, bytes: textMeta(0x03, 'Mouth Drums') }];
      for (const d of q.drums) {
        const n = DRUM_NOTES[d.cls]; if (!n) continue;
        const s = tickU(d.qu);
        ev.push({ tick: s, order: 3, bytes: [0x99, n, d.vel] });
        ev.push({ tick: s + PPQ / 4, order: 2, bytes: [0x89, n, 0] });
      }
      tracks.push(trackChunk(ev));
    }
    const head = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 1, 0, tracks.length, (PPQ >> 8) & 255, PPQ & 255];
    const out = head.concat(...tracks);
    return new Uint8Array(out);
  }

  // ---------- tiny zip (store only) ----------
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(u8) { let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
  function zip(files) { // [{name, data:Uint8Array}]
    const parts = [], central = []; let off = 0;
    const u16 = v => [v & 255, (v >> 8) & 255], u32 = v => [v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255];
    for (const f of files) {
      const name = Array.from(f.name).map(c => c.charCodeAt(0));
      const crc = crc32(f.data), sz = f.data.length;
      const local = [...u32(0x04034b50), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0x21), ...u32(crc), ...u32(sz), ...u32(sz), ...u16(name.length), ...u16(0), ...name];
      parts.push(new Uint8Array(local), f.data);
      central.push(...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0x21), ...u32(crc), ...u32(sz), ...u32(sz), ...u16(name.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(0), ...u32(off), ...name);
      off += local.length + sz;
    }
    const end = [...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length), ...u32(central.length), ...u32(off), ...u16(0)];
    parts.push(new Uint8Array(central), new Uint8Array(end));
    let total = 0; for (const p of parts) total += p.length;
    const out = new Uint8Array(total); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  root.MG = { separation, cleanBass, detectKey, fixedMap, trackGrid, shiftMap, uOf, tOf, biquadLP, calibFromClip, analyze, pickOnsets, features, classify, trainModel, buildGroove, estimateTempo, alignGrid, quantizeGroove, writeMidi, zip, DRUM_NOTES, CLASSES };
})(typeof window !== 'undefined' ? window : globalThis);
