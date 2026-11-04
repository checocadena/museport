// The shots made in Remotion. Every number comes from public/film-data.json, which
// make_film_data.py reads from the committed evidence.
import React from 'react';
import { AbsoluteFill, Easing, Img, interpolate, spring, staticFile, useCurrentFrame,
  useVideoConfig } from 'remotion';
import { C, F } from '../theme';
import { Count, Eyebrow, Rise, Source, Stage } from './common';

type Cue = (re: RegExp, fallback: number, last?: boolean) => number;

type Film = {
  devcuda: { files: number; attempted: number; correct: number; nvidia: number };
  portable_speeds: { device: string; ratio: number; ms: number; cuda_ms: number }[];
  amd: { device: string; before_ms: number; after_ms: number };
  run5: { device: string; ratio: number };
  gpu_calls: { calls: number; usd: number };
  tuning: { ratio: number; pass: boolean; cublas: boolean }[];
  onemath: { pr: string; passed_before: number; passed_after: number };
  j3_test: string[];
  j4_session: string[];
};
export type Data = { film: Film };

const short = (d: string) => d.replace(/^NVIDIA (GeForce )?/, '').replace(/ 80GB HBM3$/, '');

// The GPT-2 trainer's move, drawn as the voice says it: CUDA (NVIDIA only) -> SYCL, the open
// standard behind oneAPI -> NVIDIA, AMD and CPU, each checked off as it is named.
export const Hook: React.FC<{ cue: Cue }> = ({ cue }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const tCuda = cue(/NVIDIA/i, 60);           // "It normally only runs on NVIDIA"
  const tSycl = cue(/SYCL/i, 150);            // "it now runs on SYCL"
  const tApi = cue(/API/i, tSycl + 50);       // "the open standard behind oneAPI"
  const chips = ['NVIDIA', 'AMD', 'CPU'];
  const at = [cue(/NVIDIA/i, tApi + 30, true), cue(/AMD/i, tApi + 55, true), cue(/CPU/i, tApi + 80, true)];
  const pop = (t: number) => spring({ frame: f - t, fps, config: { damping: 11, stiffness: 160 } });
  const draw = (t: number, d = 14) => interpolate(f, [t, t + d], [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic) });
  // diagram geometry (px, inside a 1640 x 470 box)
  const H = 470, cy = H / 2;
  const cuda = { x: 0, w: 330, h: 150 }, sycl = { x: 540, w: 430, h: 180 };
  const chip = { x: 1250, w: 390, h: 104, gap: 34 };
  const chipY = (i: number) => cy + (i - 1) * (chip.h + chip.gap);
  const arrow = draw(tSycl - 6, 16);
  const syclIn = pop(tSycl);
  return (
    <Stage pad="80px 140px 170px">
      <Eyebrow>Meta's Muse · CUDA → SYCL (oneAPI)</Eyebrow>
      <Rise at={4}><div style={{ fontFamily: F.display, fontSize: 84, fontWeight: 700,
        lineHeight: 1.05, marginTop: 28 }}>Karpathy's GPT-2 trainer</div></Rise>
      <div style={{ position: 'relative', width: 1640, height: H, marginTop: 50 }}>
        <svg width={1640} height={H} style={{ position: 'absolute', inset: 0, overflow: 'visible' }}>
          {/* CUDA -> SYCL */}
          {arrow > 0 && <line x1={cuda.x + cuda.w + 14} y1={cy} x2={cuda.x + cuda.w + 14 + (sycl.x - cuda.x - cuda.w - 40) * arrow}
            y2={cy} stroke={C.accent} strokeWidth={5} strokeLinecap="round" />}
          {arrow > 0.95 && <path d={`M${sycl.x - 30},${cy - 14} L${sycl.x - 14},${cy} L${sycl.x - 30},${cy + 14}`}
            fill="none" stroke={C.accent} strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />}
          {/* SYCL -> each target, drawn as it is named */}
          {chips.map((c, i) => {
            const x0 = sycl.x + sycl.w + 10, y0 = cy, x1 = chip.x - 12, y1 = chipY(i);
            const len = 420;
            return <path key={c} d={`M${x0},${y0} C${x0 + 140},${y0} ${x1 - 140},${y1} ${x1},${y1}`}
              fill="none" stroke={C.ok} strokeWidth={4} strokeLinecap="round"
              strokeDasharray={len} strokeDashoffset={len * (1 - draw(at[i] - 10, 12))}
              opacity={f >= at[i] - 10 ? 0.85 : 0} />;
          })}
        </svg>
        {/* CUDA box */}
        <div style={{ position: 'absolute', left: cuda.x, top: cy - cuda.h / 2, width: cuda.w, height: cuda.h,
          borderRadius: 20, border: `2px solid ${C.line}`, background: C.panel, display: 'flex',
          flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
          opacity: pop(tCuda), transform: `scale(${0.85 + 0.15 * pop(tCuda)})` }}>
          <div style={{ fontFamily: F.mono, fontSize: 54, fontWeight: 700, color: C.fg }}>CUDA</div>
          <div style={{ fontSize: 26, color: C.muted, marginTop: 6 }}>NVIDIA only</div>
        </div>
        {/* SYCL box */}
        <div style={{ position: 'absolute', left: sycl.x, top: cy - sycl.h / 2, width: sycl.w, height: sycl.h,
          borderRadius: 22, border: `3px solid ${C.accent}`, background: 'rgba(91,140,255,.10)',
          display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
          opacity: Math.min(1, syclIn * 1.4), transform: `scale(${0.7 + 0.3 * syclIn})`,
          boxShadow: `0 0 ${60 * syclIn}px rgba(91,140,255,${0.25 * syclIn})` }}>
          <div style={{ fontFamily: F.mono, fontSize: 64, fontWeight: 700, color: C.fg }}>SYCL</div>
          <div style={{ fontSize: 26, color: C.fg, marginTop: 8, opacity: draw(tApi, 12),
            transform: `translateY(${(1 - draw(tApi, 12)) * 10}px)` }}>open standard · oneAPI</div>
        </div>
        {/* targets */}
        {chips.map((c, i) => {
          const p = pop(at[i]);                // 0 -> 1 with a little overshoot
          const on = f >= at[i];
          const tick = draw(at[i] + 2, 10);   // the check mark draws itself
          return (
            <div key={c} style={{ position: 'absolute', left: chip.x, top: chipY(i) - chip.h / 2,
              width: chip.w, height: chip.h, borderRadius: 18, display: 'flex', alignItems: 'center',
              gap: 20, padding: '0 30px', boxSizing: 'border-box',
              border: `2px solid ${on ? C.ok : C.line}`,
              background: on ? `rgba(60,207,142,${0.12 * Math.min(1, p)})` : C.panel,
              transform: `scale(${on ? 0.9 + 0.1 * p : 1})`,
              boxShadow: on ? `0 0 ${40 * Math.max(0, 1 - (f - at[i]) / 20)}px rgba(60,207,142,.5)` : 'none' }}>
              <svg width={44} height={44} viewBox="0 0 44 44">
                <circle cx={22} cy={22} r={20} fill="none" stroke={on ? C.ok : C.line} strokeWidth={3} />
                <path d="M12,23 L19,30 L32,15" fill="none" stroke={C.ok} strokeWidth={4.5}
                  strokeLinecap="round" strokeLinejoin="round" strokeDasharray={32}
                  strokeDashoffset={32 * (1 - tick)} />
              </svg>
              <span style={{ fontSize: 46, fontWeight: 700, color: on ? C.ok : C.muted }}>{c}</span>
            </div>
          );
        })}
      </div>
    </Stage>
  );
};

export const Speed: React.FC<{ d: Data }> = ({ d }) => {
  const f = useCurrentFrame();
  const sp = d.film.portable_speeds;
  const max = 3.2;
  const amd = d.film.amd;
  return (
    <Stage>
      <Source>evidence/l1-llmc-bf16/ptune</Source>
      <Eyebrow>Portable SYCL + oneMath · step time vs CUDA</Eyebrow>
      <div style={{ display: 'flex', gap: 90, marginTop: 40 }}>
        <div style={{ flex: 1.4 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 22, marginBottom: 36 }}>
            <span style={{ fontFamily: F.mono, fontSize: 72, color: C.muted,
              textDecoration: 'line-through' }}>{Math.round(d.film.run5.ratio)}x</span>
            <span style={{ fontSize: 30, color: C.muted }}>before tuning ({short(d.film.run5.device)})</span>
          </div>
          {sp.map((s, i) => {
            const p = interpolate(f, [30 + i * 18, 70 + i * 18], [0, 1],
              { extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
                easing: Easing.out(Easing.cubic) });
            const hot = s.ratio > 2;
            return (
              <div key={s.device} style={{ marginBottom: 46 }}>
                <div style={{ fontSize: 34, marginBottom: 12 }}>{short(s.device)}</div>
                <div style={{ position: 'relative', height: 64, background: C.panel,
                  borderRadius: 10 }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, borderRadius: 10,
                    width: `${(p * s.ratio / max) * 100}%`, background: hot ? C.warn : C.accent }} />
                  <div style={{ position: 'absolute', left: `${100 / max}%`, top: -10, bottom: -10,
                    width: 3, background: C.fg, opacity: .8 }} />
                  <div style={{ position: 'absolute', left: `calc(${(s.ratio / max) * 100}% + 20px)`,
                    top: 8, fontSize: 40, fontWeight: 700, fontFamily: F.mono, opacity: p }}>
                    {s.ratio.toFixed(2)}x</div>
                </div>
              </div>
            );
          })}
          <div style={{ fontSize: 26, color: C.muted }}>white line: llm.c's CUDA on the same GPU</div>
        </div>
        <div style={{ flex: 1, borderLeft: `2px solid ${C.line}`, paddingLeft: 70 }}>
          <div style={{ fontSize: 34 }}>AMD Instinct MI350X</div>
          <div style={{ fontSize: 26, color: C.muted, marginTop: 8 }}>ms per training step</div>
          <Rise at={110}><div style={{ fontFamily: F.mono, fontSize: 96, color: C.muted,
            textDecoration: 'line-through', marginTop: 40 }}>{amd.before_ms.toFixed(0)}</div></Rise>
          <Rise at={135}><div style={{ fontFamily: F.mono, fontSize: 150, fontWeight: 700,
            color: C.ok }}>{Math.floor(amd.after_ms)}</div></Rise>
          <div style={{ fontSize: 26, color: C.muted }}>same portable source</div>
        </div>
      </div>
    </Stage>
  );
};

export const Kernels: React.FC<{ d: Data }> = ({ d }) => {
  const k = d.film.devcuda;
  const rows: [number, string, string][] = [
    [k.attempted, `of ${k.files} CUDA kernel files ported`, C.fg],
    [k.correct, 'pass their hidden check', C.ok],
    [k.nvidia, "also pass on NVIDIA against llm.c's own CUDA", C.accent],
  ];
  return (
    <Stage>
      <Source>llm.c/dev/cuda · muse-spark-1.3</Source>
      <Eyebrow>llm.c's kernel experiments</Eyebrow>
      <div style={{ marginTop: 70 }}>
        {rows.map(([n, label, color], i) => (
          <Rise key={label} at={8 + i * 18} style={{ display: 'flex', alignItems: 'baseline',
            gap: 40, marginBottom: 40 }}>
            <div style={{ fontFamily: F.mono, fontSize: 150, fontWeight: 700, color, width: 260,
              textAlign: 'right' }}><Count to={n} at={8 + i * 18} /></div>
            <div style={{ fontSize: 52 }}>{label}</div>
          </Rise>
        ))}
      </div>
    </Stage>
  );
};

// A replay of a committed terminal log, time-compressed and labeled as such.
export const Terminal: React.FC<{ cmd: string; lines: string[]; source: string; frames: number;
  wrap?: boolean }> = ({ cmd, lines, source, frames, wrap }) => {
  const f = useCurrentFrame();
  const typed = Math.min(cmd.length, Math.floor(f * 1.6));
  const start = Math.ceil(cmd.length / 1.6) + 8;
  const per = Math.max(2, (frames * 0.7 - start) / Math.max(1, lines.length));
  const shown = Math.max(0, Math.min(lines.length, Math.floor((f - start) / per) + 1));
  // keep the newest lines in view, as a terminal scrolls
  const room = wrap ? 9 : 13;
  const first = Math.max(0, shown - room);
  return (
    <Stage>
      <Source>replay of {source} · real output, time compressed</Source>
      <div style={{ marginTop: 50, background: C.panel, border: `1px solid ${C.line}`,
        borderRadius: 18, padding: '40px 50px', fontFamily: F.mono, fontSize: 30,
        lineHeight: 1.55, height: 700, overflow: 'hidden', whiteSpace: wrap ? 'pre-wrap' : 'pre' }}>
        {first === 0 && <div><span style={{ color: C.ok }}>$ </span>{cmd.slice(0, typed)}</div>}
        {f >= start && lines.slice(first, shown).map((ln, i) => {
          const good = /overall okay: 1|OK|^Once upon/.test(ln);
          return <div key={i} style={{ color: good ? C.ok : ln.startsWith('$') ? C.fg : C.muted,
            fontWeight: /overall okay/.test(ln) ? 700 : 400,
            fontSize: /overall okay/.test(ln) ? 44 : undefined }}>{ln}</div>;
        })}
      </div>
    </Stage>
  );
};

export const OneMath: React.FC<{ d: Data }> = ({ d }) => {
  const o = d.film.onemath;
  return (
    <Stage>
      <Source>upstream/onemath</Source>
      <Eyebrow>Upstream · UXL Foundation</Eyebrow>
      <Rise at={4}><div style={{ fontFamily: F.mono, fontSize: 64, marginTop: 40 }}>{o.pr}</div></Rise>
      <Rise at={20}><div style={{ fontSize: 46, marginTop: 50, lineHeight: 1.4 }}>
        bf16 <span style={{ fontFamily: F.mono }}>gemm</span> on NVIDIA threw{' '}
        <span style={{ fontFamily: F.mono, color: C.bad }}>unimplemented</span>.<br />
        Muse's port hit it; the 2-line fix is submitted.</div></Rise>
      <Rise at={50} style={{ display: 'flex', alignItems: 'baseline', gap: 30, marginTop: 70 }}>
        <span style={{ fontSize: 34, color: C.muted }}>oneMath's own BLAS tests, passed:</span>
        <span style={{ fontFamily: F.mono, fontSize: 90, color: C.muted }}>{o.passed_before}</span>
        <span style={{ fontSize: 60 }}>→</span>
        <span style={{ fontFamily: F.mono, fontSize: 90, fontWeight: 700, color: C.ok }}>
          {o.passed_after}</span>
      </Rise>
    </Stage>
  );
};

// The published report: a slow pan down a full-page capture (scripts/capture-report.mjs).
export const Report: React.FC<{ have: boolean }> = ({ have }) => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  if (!have) return <Stage><Eyebrow>missing</Eyebrow>
    <div style={{ fontSize: 50, marginTop: 30 }}>public/report/overview.png: run
      scripts/prepare-assets.sh</div></Stage>;
  const y = interpolate(f, [0, durationInFrames], [0, -2200],
    { easing: Easing.inOut(Easing.sin), extrapolateRight: 'clamp' });
  const s = interpolate(f, [0, durationInFrames], [1.0, 1.06]);
  return (
    <AbsoluteFill style={{ background: C.bg, overflow: 'hidden' }}>
      <Img src={staticFile('report/overview.png')} style={{ width: 1920,
        transform: `translateY(${y}px) scale(${s})`, transformOrigin: 'top center' }} />
    </AbsoluteFill>
  );
};

export const Limits: React.FC = () => {
  const items = [
    'The one-run trainer port worked 1 time in 2.',
    'Kernel ports are fp32, at reduced sizes, one run each.',
    'On an H100, the portable trainer is 2.9x CUDA\'s time.',
    'AMD was not run for the two attention kernels.',
  ];
  return (
    <Stage>
      <Eyebrow>Limits, measured and published</Eyebrow>
      <div style={{ marginTop: 60 }}>
        {items.map((t, i) => <Rise key={t} at={6 + i * 14}><div style={{ fontSize: 50,
          marginBottom: 34, display: 'flex', gap: 26 }}><span style={{ color: C.warn }}>–</span>
          {t}</div></Rise>)}
      </div>
    </Stage>
  );
};

export const End: React.FC<{ credit?: string }> = ({ credit }) => (
  <Stage pad="0">
    <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
      <Rise at={4}><div style={{ fontFamily: F.display, fontSize: 84, fontWeight: 700,
        lineHeight: 1.15 }}>Moving AI off one vendor's GPUs,<br />
        <span style={{ color: C.ok }}>with proof.</span></div></Rise>
      <Rise at={30}><div style={{ fontFamily: F.mono, fontSize: 46, marginTop: 70, color: C.accent }}>
        github.com/checocadena/museport</div></Rise>
      <Rise at={44}><div style={{ fontSize: 28, marginTop: 30, color: C.muted }}>
        Meta Global AI Developer Hackathon 2026 · muse-spark-1.3</div></Rise>
      {credit && <Rise at={60}><div style={{ fontSize: 22, marginTop: 50, color: C.muted }}>
        {credit}</div></Rise>}
    </AbsoluteFill>
  </Stage>
);

const Chip: React.FC<{ on: boolean; label: string; note?: string; tone?: string }> = ({ on, label, note, tone }) => (
  <div style={{ padding: '22px 40px', borderRadius: 16, border: `2px solid ${on ? (tone ?? C.ok) : C.line}`,
    background: on ? 'rgba(60,207,142,.08)' : C.panel, minWidth: 220 }}>
    <div style={{ fontSize: 44, fontWeight: 700, color: on ? (tone ?? C.ok) : C.muted }}>{label}</div>
    {note && <div style={{ fontSize: 24, color: C.muted, marginTop: 8 }}>{note}</div>}
  </div>
);

export const Why: React.FC<{ cue: Cue }> = ({ cue }) => (
  <Stage>
    <Eyebrow>Why Muse Port</Eyebrow>
    <Rise at={6}><div style={{ fontFamily: F.display, fontSize: 88, fontWeight: 700, lineHeight: 1.1,
      marginTop: 40 }}>Code that only runs on NVIDIA,<br />
      <span style={{ color: C.ok }}>running on other GPUs and CPUs.</span></div></Rise>
    <div style={{ display: 'flex', gap: 22, marginTop: 80, flexWrap: 'wrap' }}>
      {([['leaner', /leaner/i], ['cheaper', /cheaper/i], ['runs on anybody\'s machine', /anybody/i]] as
        [string, RegExp][]).map(([w, re], i) => (
        <Rise key={w} at={cue(re, 120 + i * 40)}><div style={{ fontSize: 46, padding: '14px 30px',
          border: `1px solid ${C.line}`, borderRadius: 999, color: C.fg }}>{w}</div></Rise>))}
    </div>
    <Rise at={cue(/^AI$/i, 300, true)}><div style={{ fontSize: 36, color: C.muted, marginTop: 60 }}>
      …and AI does the porting.</div></Rise>
  </Stage>
);

export const Intel: React.FC = () => (
  <Stage>
    <Eyebrow>Where the GPT-2 trainer runs</Eyebrow>
    <div style={{ display: 'flex', gap: 26, marginTop: 70, flexWrap: 'wrap' }}>
      <Rise at={4}><Chip on label="✓ NVIDIA" note="tested" /></Rise>
      <Rise at={14}><Chip on label="✓ AMD" note="tested" /></Rise>
      <Rise at={24}><Chip on label="✓ CPU" note="tested" /></Rise>
      <Rise at={50}><Chip on={false} label="Intel GPU" note="should run · not tested: no cloud access" /></Rise>
    </div>
    <Rise at={80}><div style={{ fontSize: 34, color: C.muted, marginTop: 70, maxWidth: 1300 }}>
      SYCL compilers target Intel GPUs natively. Intel's GPU cloud had no access for this project,
      so it is not claimed.</div></Rise>
  </Stage>
);

export const Rent: React.FC<{ d: Data; cue: Cue }> = ({ d, cue }) => {
  const at = [cue(/GPUs/i, 10), cue(/cap/i, 55), cue(/deleted/i, 100)];
  const steps: [string, string][] = [
    ['Rent', 'a GPU on Runpod or DigitalOcean, with a spending cap'],
    ['Test', "build Muse's code, run llm.c's test, time it next to CUDA"],
    ['Delete', 'the machine, and confirm it is gone'],
  ];
  const g = d.film.gpu_calls;
  return (
    <Stage>
      <Source>oneapi-agent-toolkit · gpu_test</Source>
      <Eyebrow>The agent rents its own GPUs</Eyebrow>
      <div style={{ display: 'flex', gap: 30, marginTop: 70, alignItems: 'stretch', flexWrap: 'wrap' }}>
        {steps.map(([h, t], i) => (
          <Rise key={h} at={at[i]} style={{ flex: 1, minWidth: 420 }}>
            <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 18,
              padding: '34px 36px', height: '100%' }}>
              <div style={{ fontFamily: F.mono, fontSize: 24, color: C.accent }}>{String(i + 1).padStart(2, '0')}</div>
              <div style={{ fontSize: 60, fontWeight: 700, marginTop: 10 }}>{h}</div>
              <div style={{ fontSize: 32, color: C.muted, marginTop: 14, lineHeight: 1.35 }}>{t}</div>
            </div>
          </Rise>))}
      </div>
      <Rise at={at[2] + 30}><div style={{ fontSize: 34, marginTop: 50, color: C.fg }}>
        In the GPU tuning run, Muse made <b>{g.calls}</b> GPU calls, about{' '}
        <b>${g.usd.toFixed(2)}</b> of GPU time in all.</div></Rise>
    </Stage>
  );
};

export const Future: React.FC<{ cue: Cue }> = ({ cue }) => (
  <Stage>
    <Eyebrow>Next</Eyebrow>
    <div style={{ marginTop: 50, display: 'grid', gap: 30 }}>
      <Rise at={6}><div style={{ fontSize: 64 }}>Intel GPUs, and more repos.</div></Rise>
      <Rise at={cue(/agent/i, 90)}><div style={{ fontFamily: F.display, fontSize: 92, fontWeight: 700, lineHeight: 1.1,
        marginTop: 30 }}>An agent that <span style={{ color: C.accent }}>ports</span> the code,<br />
        then <span style={{ color: C.ok }}>tunes</span> it on real hardware.</div></Rise>
    </div>
  </Stage>
);

// Muse tuning its own port: every timed NVIDIA gpu_test call of the GPU tuning run, in order,
// after the one-run port's starting point. Log scale; 1x is llm.c's CUDA on the same GPU.
export const Tuning: React.FC<{ d: Data }> = ({ d }) => {
  const f = useCurrentFrame();
  const W = 1640, H = 560, L = 110, R = 40, T = 30, B = 60;
  const pts = [{ ratio: d.film.run5.ratio, pass: true, cublas: false, start: true },
    ...d.film.tuning.map(t => ({ ...t, start: false }))];
  const lo = Math.log(1), hi = Math.log(40);
  const x = (i: number) => L + (i / (pts.length - 1)) * (W - L - R);
  const y = (r: number) => T + ((hi - Math.log(r)) / (hi - lo)) * (H - T - B);
  const at = (i: number) => 24 + i * 11; // frame each point appears
  const shown = pts.filter((_, i) => f >= at(i));
  const path = shown.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.ratio).toFixed(1)}`).join(' ');
  // the headline: the best portable result so far (cuBLAS calls are vendor-specific)
  const portable = shown.filter(p => !p.cublas && p.pass);
  const best = portable.length ? Math.min(...portable.map(p => p.ratio)) : undefined;
  const sp = d.film.portable_speeds;
  return (
    <Stage pad="80px 140px 170px">
      <Source>evidence/l1-llmc-bf16/gtune · every gpu_test call</Source>
      <Eyebrow>Muse tuning its own port on rented NVIDIA GPUs</Eyebrow>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 24, marginTop: 18 }}>
        <span style={{ fontFamily: F.mono, fontSize: 84, fontWeight: 700,
          color: best !== undefined && best < 2 ? C.ok : C.fg }}>
          {best !== undefined ? `${best < 10 ? best.toFixed(2) : Math.round(best)}x` : ''}</span>
        <span style={{ fontSize: 30, color: C.muted }}>CUDA's time per training step, best portable so far</span>
      </div>
      <svg width={W} height={H} style={{ marginTop: 6, overflow: 'visible' }}>
        {[1, 2, 5, 10, 40].map(r => (
          <g key={r}>
            <line x1={L} x2={W - R} y1={y(r)} y2={y(r)} stroke={r === 1 ? C.fg : C.line}
              strokeWidth={r === 1 ? 2 : 1} strokeDasharray={r === 1 ? '0' : '4 6'} />
            <text x={L - 18} y={y(r) + 9} textAnchor="end" fill={C.muted} fontSize={24}
              fontFamily={F.mono}>{r}x</text>
          </g>))}
        <text x={W - R} y={y(1) + 34} textAnchor="end" fill={C.fg} fontSize={22}
          fontFamily={F.body}>llm.c's CUDA on the same GPU</text>
        <path d={path} fill="none" stroke={C.accent} strokeWidth={3} opacity={0.7} />
        {shown.map((p, i) => (
          <circle key={i} cx={x(i)} cy={y(p.ratio)} r={p.start ? 12 : 10}
            fill={p.cublas ? C.bg : p.pass ? C.ok : C.bad}
            stroke={p.cublas ? C.warn : p.pass ? C.ok : C.bad} strokeWidth={p.cublas ? 3 : 0} />))}
        {f >= at(0) && <text x={x(0) + 22} y={y(pts[0].ratio) + 8} fill={C.muted} fontSize={24}
          fontFamily={F.body}>the one-run port</text>}
      </svg>
      <div style={{ display: 'flex', gap: 34, fontSize: 24, color: C.muted, marginTop: 4, flexWrap: 'wrap' }}>
        <span><span style={{ color: C.ok }}>●</span> test passed</span>
        <span><span style={{ color: C.bad }}>●</span> test failed</span>
        <span><span style={{ color: C.warn }}>○</span> calls NVIDIA's cuBLAS (vendor-specific)</span>
        <span style={{ color: C.fg }}>Portable, no vendor code: {sp.map(s =>
          `${s.ratio.toFixed(2)}x ${short(s.device)}`).join(' · ')}</span>
      </div>
    </Stage>
  );
};
