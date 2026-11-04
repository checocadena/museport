// The Muse Port film: one composition, built from the voiceover (public/lines.json).
// Each line has its own shot; captions follow Whisper's word timings when they exist; the
// music bed dips about 4 dB under the voice with raised-cosine ramps.
import React from 'react';
import { AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion';
import { C, F, FPS } from './theme';
import { Assets, chunks, Line, layout, Seg } from './timeline';
import { Clip } from './shots/Clip';
import { Data, End, Future, Hook, Intel, Kernels, Limits, OneMath, Rent, Report, Speed, Terminal,
  Why } from './shots/Cards';

export type FilmProps = { lines: Line[]; assets: Assets; data: Data };

const CLIP_SHOTS = new Set(['disclosure', 'term_scan', 'pipeline', 'term_bug', 'replay', 'l1']);

const Shot: React.FC<{ seg: Seg; p: FilmProps }> = ({ seg, p }) => {
  const s = seg.line.shot;
  const d = p.data;
  // l1: only its first 6 s, the port ("ported by Muse in one run"); its tuning half is
  // replaced by the speed card two lines later
  if (s === 'l1') return <AbsoluteFill>
    <Clip name={s} secs={p.assets.clips[s]} frames={seg.frames} maxRate={1} upTo={6} />
    <Note at={60}>This one-run port worked in 1 of 2 tries.</Note>
  </AbsoluteFill>;
  if (CLIP_SHOTS.has(s)) return <Clip name={s} secs={p.assets.clips[s]} frames={seg.frames} />;
  switch (s) {
    case 'why': return <Why />;
    case 'hook': return <Hook />;
    case 'intel': return <Intel />;
    case 'rent': return <Rent d={d} />;
    case 'future': return <Future />;
    case 'speed': return <Speed d={d} />;
    case 'kernels': return <Kernels d={d} />;
    case 'j3': return <Terminal cmd="make && ./test_gpt2fp32sycl" lines={d.film.j3_test}
      source="evidence/j3-fresh-clone/test.log" frames={seg.frames} />;
    case 'j4': return <Terminal cmd="cd llama2-sycl" lines={d.film.j4_session}
      source="evidence/j4-fresh-clone/session.log" frames={seg.frames} wrap />;
    case 'onemath': return <OneMath d={d} />;
    case 'report': return <Report have={p.assets.report} />;
    case 'limits': return <Limits />;
    case 'end': return <End />;
    default: return <AbsoluteFill style={{ background: C.bad }}>unknown shot {s}</AbsoluteFill>;
  }
};

// An on-screen fact the voiceover leaves out (kept honest without re-recording).
const Note: React.FC<{ at: number; children: React.ReactNode }> = ({ at, children }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [at, at + 15], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'flex-end', paddingBottom: 190 }}>
    <div style={{ opacity: o, fontFamily: F.body, fontSize: 36, color: C.warn, border: `1px solid ${C.warn}`,
      borderRadius: 12, padding: '12px 26px', background: 'rgba(10,12,17,.85)' }}>{children}</div>
  </AbsoluteFill>;
};

// Short cross-fade into each shot.
const FadeIn: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{ opacity: interpolate(f, [0, 14], [0, 1],
    { extrapolateRight: 'clamp' }) }}>{children}</AbsoluteFill>;
};

const Captions: React.FC<{ seg: Seg; p: FilmProps }> = ({ seg, p }) => {
  const f = useCurrentFrame();
  const t = (f - seg.voFrom) / FPS;
  const secs = seg.voFrames / FPS;
  const ch = chunks(seg.line, secs, p.assets.words[String(seg.line.n)]);
  const cur = ch.find((c, i) => t >= c.t0 && (t < c.t1 || i === ch.length - 1));
  if (!cur || t < 0 || t > secs + 0.4) return null;
  const fade = interpolate(t, [cur.t0, cur.t0 + 0.12], [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 54 }}>
      <div style={{ maxWidth: 1500, textAlign: 'center', fontFamily: F.body, fontSize: 44,
        fontWeight: 600, lineHeight: 1.3, color: '#fff', background: 'rgba(0,0,0,.72)',
        padding: '14px 30px', borderRadius: 12, opacity: fade }}>{cur.text}</div>
    </AbsoluteFill>
  );
};

// Music volume: a bed that dips 4 dB under each spoken line, with 0.5 s raised-cosine ramps,
// a 1.5 s fade-in and a 3 s fade-out.
const BED = 0.32;
const DIP = Math.pow(10, -4 / 20);
const ramp = (x: number) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, x)));
function musicVolume(frame: number, segs: Seg[], total: number): number {
  const r = 0.5 * FPS;
  let duck = 0;
  for (const s of segs) {
    if (!s.hasVo) continue;
    const a = s.from + s.voFrom;
    const b = a + s.voFrames;
    const v = frame < a ? ramp((frame - (a - r)) / r) : frame > b ? ramp(((b + r) - frame) / r) : 1;
    duck = Math.max(duck, v);
  }
  const level = BED * (1 - duck + duck * DIP);
  const fadeIn = ramp(frame / (1.5 * FPS));
  const fadeOut = ramp((total - frame) / (3 * FPS));
  return level * fadeIn * fadeOut;
}

export const Film: React.FC<FilmProps> = (p) => {
  const { segs, total } = layout(p.lines, p.assets);
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      {segs.map(seg => (
        <Sequence key={seg.line.n} from={seg.from} durationInFrames={seg.frames}
          name={`${seg.line.n} ${seg.line.shot}`}>
          <FadeIn><Shot seg={seg} p={p} /></FadeIn>
          <Captions seg={seg} p={p} />
          {seg.hasVo && (
            <Sequence from={seg.voFrom}>
              <Audio src={staticFile(`vo/vo-${String(seg.line.n).padStart(2, '0')}.wav`)} />
            </Sequence>
          )}
        </Sequence>
      ))}
      {p.assets.music && (
        <Audio src={staticFile(p.assets.music)} loop
          volume={f => musicVolume(f, segs, total)} />
      )}
    </AbsoluteFill>
  );
};
