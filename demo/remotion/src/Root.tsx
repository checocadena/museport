import React from 'react';
import { Composition, staticFile } from 'remotion';
import { Film, FilmProps } from './Film';
import { FPS } from './theme';
import { Assets, layout } from './timeline';

// Optional files (takes, word timings, clips, music) are read when they exist, so the film
// previews before any voiceover is recorded: lines then use an estimated length.
async function json<T>(name: string, fallback: T): Promise<T> {
  try {
    const r = await fetch(staticFile(name));
    return r.ok ? ((await r.json()) as T) : fallback;
  } catch {
    return fallback;
  }
}

async function load(): Promise<FilmProps> {
  const manifest = await json<{ music?: string | null; report?: boolean }>('assets.json', {});
  const assets: Assets = {
    voLengths: await json('vo-lengths.json', {}),
    words: await json('vo-words.json', {}),
    clips: await json('clips.json', {}),
    music: manifest.music ?? null,
    report: manifest.report ?? false,
  };
  return { lines: await json('lines.json', []), assets, data: await json('film-data.json', {} as never) };
}

export const Root: React.FC = () => (
  <Composition
    id="Film"
    component={Film as unknown as React.FC<Record<string, unknown>>}
    width={1920}
    height={1080}
    fps={FPS}
    durationInFrames={FPS}
    defaultProps={{} as Record<string, unknown>}
    calculateMetadata={async () => {
      const props = await load();
      return { durationInFrames: Math.max(FPS, layout(props.lines, props.assets).total),
        props: props as unknown as Record<string, unknown> };
    }}
  />
);
