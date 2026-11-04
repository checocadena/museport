// The voiceover drives the timing, as in the BALI film: each line gets its own shot, and the
// shot lasts lead + take + tail. Replace a take and every shot after it moves.
import { FPS } from './theme';

export type Line = {
  n: number; shot: string; text: string; lead?: number; tail?: number;
};
export type Word = { w: string; t0: number; t1: number }; // seconds from the take's start
export type Assets = {
  voLengths: Record<string, number>; // "1" -> seconds, for takes that exist in public/vo/
  words: Record<string, Word[]>; // Whisper word timings per take
  clips: Record<string, number>; // rendered clip name -> seconds, for clips in public/clips/
  music: string | null; // public/ file name of the music bed, if any
  report: boolean; // public/report/overview.png exists
};
export type Seg = {
  line: Line; from: number; frames: number; voFrom: number; voFrames: number; hasVo: boolean;
};

const DEFAULT_LEAD = 0.3;
const DEFAULT_TAIL = 0.6; // a breath between lines, so the film doesn't feel rushed
// Without a recorded take, estimate its length at a relaxed read of 2.4 words a second.
export const estimate = (text: string) => text.split(/\s+/).length / 2.4 + 0.3;

export function layout(lines: Line[], a: Assets): { segs: Seg[]; total: number } {
  let t = 0;
  const segs: Seg[] = [];
  for (const line of lines) {
    const take = a.voLengths[String(line.n)];
    const secs = take ?? estimate(line.text);
    const lead = line.lead ?? DEFAULT_LEAD;
    const tail = line.tail ?? DEFAULT_TAIL;
    const frames = Math.round((lead + secs + tail) * FPS);
    const from = Math.round(t * FPS);
    segs.push({ line, from, frames, voFrom: Math.round(lead * FPS),
      voFrames: Math.round(secs * FPS), hasVo: take !== undefined });
    t += lead + secs + tail;
  }
  return { segs, total: Math.round(t * FPS) };
}

// Caption chunks: sentences, and long sentences split at a comma, as in BALI's chunks().
export type Chunk = { text: string; t0: number; t1: number; words: { w: string; t0: number }[] };
// (seconds from the take's start)
export function chunks(line: Line, secs: number, words?: Word[]): Chunk[] {
  const parts: string[] = [];
  // sentence ends only where punctuation is followed by a space or the end (not "llm.c", "2.9")
  for (const s of line.text.match(/.+?(?:[.!?:](?=\s|$)|$)/g) ?? [line.text]) {
    const sent = s.trim();
    if (!sent) continue;
    if (sent.split(/\s+/).length > 12 && sent.includes(',')) {
      const i = sent.indexOf(',', Math.floor(sent.length / 3));
      if (i > 0) { parts.push(sent.slice(0, i + 1).trim(), sent.slice(i + 1).trim()); continue; }
    }
    parts.push(sent);
  }
  const counts = parts.map(p => p.split(/\s+/).length);
  const total = counts.reduce((x, y) => x + y, 0);
  const out: Chunk[] = [];
  let k = 0;
  let acc = 0;
  parts.forEach((p, i) => {
    let t0: number; let t1: number;
    if (words && words.length) { // each chunk starts on its first word
      // Whisper can split or merge words ("1,274"), so map by position, not by index
      const at = (j: number) => words[Math.min(words.length - 1,
        Math.round((j * words.length) / total))];
      t0 = at(k).t0;
      t1 = at(k + counts[i] - 1).t1;
    } else { // no timings yet: share the line by word count
      t0 = (acc / total) * secs;
      t1 = ((acc + counts[i]) / total) * secs;
    }
    // each word's start, for the word-by-word highlight
    const ws = p.split(/\s+/).map((w, j) => {
      const g = k + j;
      const wt = words && words.length
        ? words[Math.min(words.length - 1, Math.round((g * words.length) / total))].t0
        : (g / total) * secs;
      return { w, t0: wt };
    });
    out.push({ text: p, t0, t1, words: ws });
    k += counts[i];
    acc += counts[i];
  });
  // hold each chunk until the next one starts
  for (let i = 0; i < out.length - 1; i++) out[i].t1 = out[i + 1].t0;
  return out;
}
