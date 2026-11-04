// A clip rendered by demo/render.mjs. It plays at its own speed; if the line is longer it holds
// its last frame, and if the line is shorter it plays faster (at most maxRate, 1.6x by default) and is then cut.
import React from 'react';
import { AbsoluteFill, Freeze, OffthreadVideo, Sequence, staticFile } from 'remotion';
import { FPS } from '../theme';
import { Stage, Eyebrow } from './common';

// upTo: use only the clip's first upTo seconds (then hold that frame).
export const Clip: React.FC<{ name: string; secs?: number; frames: number; maxRate?: number;
  upTo?: number }> = ({ name, secs, frames, maxRate = 1.6, upTo }) => {
  if (secs === undefined) {
    return <Stage><Eyebrow>missing clip</Eyebrow>
      <div style={{ fontSize: 60, marginTop: 30 }}>demo/out/{name}.mp4</div>
      <div style={{ fontSize: 30, marginTop: 20, opacity: .7 }}>
        run scripts/prepare-assets.sh</div></Stage>;
  }
  const clipFrames = Math.floor(Math.min(secs, upTo ?? secs) * FPS) - 1;
  const rate = Math.min(maxRate, Math.max(1, clipFrames / frames));
  const playFrames = Math.floor(clipFrames / rate);
  const src = staticFile(`clips/${name}.mp4`);
  return (
    <AbsoluteFill>
      <Sequence durationInFrames={Math.min(frames, playFrames)}>
        <OffthreadVideo src={src} playbackRate={rate} muted />
      </Sequence>
      {frames > playFrames && (
        <Sequence from={playFrames}>
          <Freeze frame={clipFrames}><OffthreadVideo src={src} muted /></Freeze>
        </Sequence>
      )}
    </AbsoluteFill>
  );
};
