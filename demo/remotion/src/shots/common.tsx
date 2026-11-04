import React from 'react';
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from 'remotion';
import { C, F } from '../theme';

export const Stage: React.FC<{ children: React.ReactNode; pad?: string }> = ({ children, pad }) => (
  <AbsoluteFill style={{ background: C.bg, color: C.fg, fontFamily: F.body,
    padding: pad ?? '90px 140px 170px', boxSizing: 'border-box' }}>{children}</AbsoluteFill>
);

export const Eyebrow: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ fontSize: 28, letterSpacing: 4, textTransform: 'uppercase', color: C.accent,
    fontWeight: 600 }}>{children}</div>
);

// Fade and rise in, starting at `at` frames.
export const Rise: React.FC<{ at?: number; children: React.ReactNode; style?: React.CSSProperties }>
  = ({ at = 0, children, style }) => {
    const f = useCurrentFrame();
    const p = interpolate(f, [at, at + 14], [0, 1],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic) });
    return <div style={{ opacity: p, transform: `translateY(${(1 - p) * 24}px)`, ...style }}>
      {children}</div>;
  };

export const Source: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ position: 'absolute', right: 140, top: 96, fontFamily: F.mono, fontSize: 22,
    color: C.muted }}>{children}</div>
);

// A number that counts up from 0 over `dur` frames.
export const Count: React.FC<{ to: number; at?: number; dur?: number; digits?: number }>
  = ({ to, at = 0, dur = 24, digits = 0 }) => {
    const f = useCurrentFrame();
    const v = interpolate(f, [at, at + dur], [0, to],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic) });
    return <>{v.toLocaleString('en-US', { minimumFractionDigits: digits,
      maximumFractionDigits: digits })}</>;
  };
