// YouTube thumbnail, 1280x720: readable at phone size, so few words and big shapes.
import React from 'react';
import { AbsoluteFill } from 'remotion';
import { C, F } from './theme';

export const Thumbnail: React.FC = () => (
  <AbsoluteFill style={{ background: `radial-gradient(1200px 700px at 78% 30%, #16301f 0%, ${C.bg} 60%)`,
    color: C.fg, fontFamily: F.display, padding: '70px 80px', boxSizing: 'border-box' }}>
    <div style={{ fontFamily: F.mono, fontSize: 30, letterSpacing: 4, color: C.accent }}>MUSE PORT</div>
    <div style={{ fontSize: 132, fontWeight: 800, lineHeight: 1.0, marginTop: 26, letterSpacing: -2 }}>
      CUDA <span style={{ color: C.ok }}>→</span><br />any GPU</div>
    <div style={{ display: 'flex', gap: 18, marginTop: 44 }}>
      {['NVIDIA', 'AMD', 'CPU'].map(c => (
        <div key={c} style={{ fontSize: 44, fontWeight: 700, color: C.ok, border: `3px solid ${C.ok}`,
          borderRadius: 16, padding: '12px 28px', background: 'rgba(60,207,142,.10)' }}>✓ {c}</div>))}
    </div>
    <div style={{ position: 'absolute', right: 80, bottom: 70, textAlign: 'right', fontFamily: F.body,
      fontSize: 34, color: C.muted, lineHeight: 1.3 }}>
      Meta's Muse ported<br /><span style={{ color: C.fg, fontWeight: 700 }}>Karpathy's GPT-2 trainer</span></div>
  </AbsoluteFill>
);
