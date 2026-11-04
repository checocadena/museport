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
      <Eyebrow>Meta Muse · CUDA → SYCL</Eyebrow>
      <Rise at={4}><div style={{ fontFamily: F.display, fontSize: 104, fontWeight: 700,
        lineHeight: 1.05, marginTop: 40 }}>Karpathy's GPT-2 trainer,<br />
        <span style={{ color: C.muted }}>off NVIDIA-only CUDA.</span></div></Rise>
      <div style={{ display: 'flex', gap: 28, marginTop: 70 }}>
        {chips.map((c, i) => {
          const on = f > 40 + i * 12;
          return <div key={c} style={{ padding: '22px 44px', borderRadius: 16, fontSize: 44,
            fontWeight: 700, border: `2px solid ${on ? C.ok : C.line}`,
            color: on ? C.ok : C.muted, background: on ? 'rgba(60,207,142,.08)' : C.panel,
            transition: 'none' }}>{on ? '✓ ' : ''}{c}</div>;
        })}
      </div>
      <Rise at={90} style={{ marginTop: 60 }}>
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
      <div style={{ display: 'flex', gap: 90, marginTop: 50 }}>
        <div style={{ flex: 1.4 }}>
          {sp.map((s, i) => {
            const p = interpolate(f, [12 + i * 10, 40 + i * 10], [0, 1],
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
          <Rise at={60}><div style={{ fontFamily: F.mono, fontSize: 96, color: C.muted,
            textDecoration: 'line-through', marginTop: 40 }}>{amd.before_ms.toFixed(0)}</div></Rise>
          <Rise at={80}><div style={{ fontFamily: F.mono, fontSize: 150, fontWeight: 700,
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
