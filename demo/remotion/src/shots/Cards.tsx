// The shots made in Remotion. Every number comes from public/film-data.json, which
// make_film_data.py reads from the committed evidence.
import React from 'react';
import { AbsoluteFill, Easing, Img, interpolate, staticFile, useCurrentFrame,
  useVideoConfig } from 'remotion';
import { C, F } from '../theme';
import { Count, Eyebrow, Rise, Source, Stage } from './common';

type Film = {
  devcuda: { files: number; attempted: number; correct: number; nvidia: number };
  portable_speeds: { device: string; ratio: number; ms: number; cuda_ms: number }[];
  amd: { device: string; before_ms: number; after_ms: number };
  run5: { device: string; ratio: number };
  gpu_calls: { calls: number; usd: number };
  onemath: { pr: string; passed_before: number; passed_after: number };
  j3_test: string[];
  j4_session: string[];
};
export type Data = { film: Film };

const short = (d: string) => d.replace(/^NVIDIA (GeForce )?/, '').replace(/ 80GB HBM3$/, '');

export const Hook: React.FC = () => {
  const f = useCurrentFrame();
  const chips = ['NVIDIA', 'AMD', 'CPU'];
  return (
    <Stage>
      <Eyebrow>Meta's Muse · CUDA → SYCL (oneAPI)</Eyebrow>
      <Rise at={4}><div style={{ fontFamily: F.display, fontSize: 104, fontWeight: 700,
        lineHeight: 1.05, marginTop: 40 }}>Karpathy's GPT-2 trainer,<br />
        <span style={{ color: C.muted }}>off NVIDIA-only CUDA.</span></div></Rise>
      <div style={{ display: 'flex', gap: 28, marginTop: 70 }}>
        {chips.map((c, i) => {
          const on = f > 70 + i * 20;
          return <div key={c} style={{ padding: '22px 44px', borderRadius: 16, fontSize: 44,
            fontWeight: 700, border: `2px solid ${on ? C.ok : C.line}`,
            color: on ? C.ok : C.muted, background: on ? 'rgba(60,207,142,.08)' : C.panel,
            transition: 'none' }}>{on ? '✓ ' : ''}{c}</div>;
        })}
      </div>
      <Rise at={150} style={{ marginTop: 60 }}>
        <span style={{ fontFamily: F.mono, fontSize: 40, color: C.ok }}>overall okay: 1</span>
        <span style={{ fontSize: 28, color: C.muted, marginLeft: 24 }}>
          llm.c's own test, on the ported code</span>
      </Rise>
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

export const End: React.FC = () => (
  <Stage pad="0">
    <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
      <Rise at={4}><div style={{ fontFamily: F.display, fontSize: 84, fontWeight: 700,
        lineHeight: 1.15 }}>Moving AI off one vendor's GPUs,<br />
        <span style={{ color: C.ok }}>with proof.</span></div></Rise>
      <Rise at={30}><div style={{ fontFamily: F.mono, fontSize: 46, marginTop: 70, color: C.accent }}>
        github.com/checocadena/museport</div></Rise>
      <Rise at={44}><div style={{ fontSize: 28, marginTop: 30, color: C.muted }}>
        Meta Global AI Developer Hackathon 2026 · muse-spark-1.3</div></Rise>
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

export const Why: React.FC = () => (
  <Stage>
    <Eyebrow>Why Muse Port</Eyebrow>
    <Rise at={6}><div style={{ fontFamily: F.display, fontSize: 88, fontWeight: 700, lineHeight: 1.1,
      marginTop: 40 }}>Code that only runs on NVIDIA,<br />
      <span style={{ color: C.ok }}>running on other GPUs and CPUs.</span></div></Rise>
    <div style={{ display: 'flex', gap: 22, marginTop: 80, flexWrap: 'wrap' }}>
      {['leaner', 'cheaper', 'runs on anybody\'s machine'].map((w, i) => (
        <Rise key={w} at={120 + i * 40}><div style={{ fontSize: 46, padding: '14px 30px',
          border: `1px solid ${C.line}`, borderRadius: 999, color: C.fg }}>{w}</div></Rise>))}
    </div>
    <Rise at={300}><div style={{ fontSize: 36, color: C.muted, marginTop: 60 }}>
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

export const Rent: React.FC<{ d: Data }> = ({ d }) => {
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
          <Rise key={h} at={10 + i * 45} style={{ flex: 1, minWidth: 420 }}>
            <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 18,
              padding: '34px 36px', height: '100%' }}>
              <div style={{ fontFamily: F.mono, fontSize: 24, color: C.accent }}>{String(i + 1).padStart(2, '0')}</div>
              <div style={{ fontSize: 60, fontWeight: 700, marginTop: 10 }}>{h}</div>
              <div style={{ fontSize: 32, color: C.muted, marginTop: 14, lineHeight: 1.35 }}>{t}</div>
            </div>
          </Rise>))}
      </div>
      <Rise at={170}><div style={{ fontSize: 34, marginTop: 50, color: C.fg }}>
        In the GPU tuning run, Muse made <b>{g.calls}</b> GPU calls, about{' '}
        <b>${g.usd.toFixed(2)}</b> of GPU time in all.</div></Rise>
    </Stage>
  );
};

export const Future: React.FC = () => (
  <Stage>
    <Eyebrow>Next</Eyebrow>
    <div style={{ marginTop: 50, display: 'grid', gap: 30 }}>
      <Rise at={6}><div style={{ fontSize: 64 }}>Intel GPUs, and more repos.</div></Rise>
      <Rise at={90}><div style={{ fontFamily: F.display, fontSize: 92, fontWeight: 700, lineHeight: 1.1,
        marginTop: 30 }}>An agent that <span style={{ color: C.accent }}>ports</span> the code,<br />
        then <span style={{ color: C.ok }}>tunes</span> it on real hardware.</div></Rise>
    </div>
  </Stage>
);
