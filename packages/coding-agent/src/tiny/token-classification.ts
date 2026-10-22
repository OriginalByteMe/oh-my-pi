/**
 * Model-independent half of token classification for BERT-style wordpiece models: window a token
 * sequence under the model's position limit, lay the tokens back over their source text, and merge
 * per-token BIO labels into spans. Pure, so it is testable without a model; the ONNX worker supplies
 * the tokens and per-token predictions.
 *
 * transformers.js's own `token-classification` pipeline cannot be used for this: it does not truncate
 * (the tokenizer reports no `model_max_length`), returns no character offsets, and decodes `word` to
 * lower-cased, space-separated text (`daniels @ meridiancap. com`).
 */
import type { TinyEntity } from "./title-protocol";

/** Tokens per window: BERT's 512 positions minus `[CLS]` and `[SEP]`. */
export const TOKEN_WINDOW_SIZE = 510;
// lean: fixed overlap; a span longer than half of it that straddles a window cut can split in two.
// Upgrade: size the overlap from the longest entity worth keeping whole.
/** Tokens shared by neighbouring windows; each window keeps its half of the overlap. */
export const TOKEN_WINDOW_OVERLAP = 64;
/** Characters a token may sit past the cursor (a dropped control character, an `[UNK]` word) before it counts as unmatched. */
const RESYNC_CHARS = 8;
const WORDPIECE_CONTINUATION = "##";

/** Token slice `[start, end)` the model sees, of which `[keepStart, keepEnd)` is this window's share of the predictions. */
export interface TokenWindow {
	start: number;
	end: number;
	keepStart: number;
	keepEnd: number;
}

/** Split `tokenCount` tokens into overlapping windows whose kept ranges tile `[0, tokenCount)` exactly. */
export function planTokenWindows(
	tokenCount: number,
	size = TOKEN_WINDOW_SIZE,
	overlap = TOKEN_WINDOW_OVERLAP,
): TokenWindow[] {
	const windows: TokenWindow[] = [];
	const half = overlap / 2;
	for (let start = 0; start < tokenCount; start += size - overlap) {
		const end = Math.min(start + size, tokenCount);
		windows.push({
			start,
			end,
			keepStart: start === 0 ? 0 : start + half,
			keepEnd: end === tokenCount ? end : end - half,
		});
		if (end === tokenCount) break;
	}
	return windows;
}

/** Characters the wordpiece normalizer drops or treats as separators; tokens never contain them. */
function isBlank(code: number): boolean {
	return code <= 0x20 || (code >= 0x7f && code <= 0xa0) || code === 0xfffd;
}

/** BERT normalization (lower-case, strip accents) of one string. */
function normalizeWordpiece(text: string): string {
	return text
		.toLowerCase()
		.normalize("NFD")
		.replace(/\p{Mn}/gu, "");
}

/**
 * Normalized `text` plus, for each normalized UTF-16 unit, the `[start, end)` of the source character
 * it came from. `null` map means the lengths agree and offsets are identical (the common case).
 */
function normalizeWithOrigins(text: string): { norm: string; origins: Array<[number, number]> | null } {
	const norm = normalizeWordpiece(text);
	if (norm.length === text.length) return { norm, origins: null };
	const origins: Array<[number, number]> = [];
	let rebuilt = "";
	for (let i = 0; i < text.length;) {
		const char = String.fromCodePoint(text.codePointAt(i)!);
		const piece = normalizeWordpiece(char);
		for (let k = 0; k < piece.length; k += 1) origins.push([i, i + char.length]);
		rebuilt += piece;
		i += char.length;
	}
	return { norm: rebuilt, origins };
}

/**
 * Character range of `text` each wordpiece covers, or `null` where the token could not be located
 * (an `[UNK]` word, a character the normalizer rewrote unexpectedly).
 *
 * Tokens come out of the normalizer in order, separated only by whitespace, so each is matched at the
 * cursor after skipping blanks; a short forward resync absorbs dropped characters.
 */
export function alignTokens(text: string, tokens: readonly string[]): Array<readonly [number, number] | null> {
	const { norm, origins } = normalizeWithOrigins(text);
	const ranges: Array<readonly [number, number] | null> = [];
	let cursor = 0;
	for (const token of tokens) {
		const piece = token.startsWith(WORDPIECE_CONTINUATION) ? token.slice(WORDPIECE_CONTINUATION.length) : token;
		let at = cursor;
		while (at < norm.length && isBlank(norm.charCodeAt(at))) at += 1;
		let found = piece.length > 0 && norm.startsWith(piece, at) ? at : -1;
		if (found < 0 && piece.length > 0) {
			const hit = norm.slice(at, at + RESYNC_CHARS + piece.length).indexOf(piece);
			if (hit >= 0) found = at + hit;
		}
		if (found < 0) {
			ranges.push(null);
			continue;
		}
		cursor = found + piece.length;
		ranges.push(origins ? [origins[found]![0], origins[cursor - 1]![1]] : [found, cursor]);
	}
	return ranges;
}

/** Per-token model verdict: the `id2label` entry (BIO-prefixed, or `O`) and its softmax probability. */
export interface TokenPrediction {
	label: string;
	score: number;
}

/** Join wordpieces back into text when the source offsets are unknown: `##` glues, everything else is spaced. */
function joinWordpieces(tokens: readonly string[]): string {
	let out = "";
	for (const token of tokens) {
		if (token.startsWith(WORDPIECE_CONTINUATION)) out += token.slice(WORDPIECE_CONTINUATION.length);
		else out = out ? `${out} ${token}` : token;
	}
	return out;
}

interface OpenSpan {
	tag: string;
	from: number;
	scoreSum: number;
}

/**
 * Merge per-token BIO predictions into entity spans. `B-X` opens a span, `I-X` extends the open span
 * of the same tag (and opens one otherwise), `O` closes. A span is widened to whole words (the model
 * sometimes tags only the `##2` of `hunter2`), then reads as the source slice from its first to its
 * last token, so `daniels@meridiancap.com` stays intact. Its score is the mean probability of the
 * tokens the model tagged.
 */
export function mergeTokenSpans(
	text: string,
	tokens: readonly string[],
	ranges: ReadonlyArray<readonly [number, number] | null>,
	predictions: readonly TokenPrediction[],
): TinyEntity[] {
	const entities: TinyEntity[] = [];
	let open: OpenSpan | null = null;
	const close = (to: number): void => {
		if (!open) return;
		let from = open.from;
		while (from > 0 && tokens[from]!.startsWith(WORDPIECE_CONTINUATION)) from -= 1;
		let last = to;
		while (last + 1 < tokens.length && tokens[last + 1]!.startsWith(WORDPIECE_CONTINUATION)) last += 1;
		const first = ranges[from];
		const end = ranges[last];
		entities.push({
			label: open.tag,
			text: first && end ? text.slice(first[0], end[1]) : joinWordpieces(tokens.slice(from, last + 1)),
			score: open.scoreSum / (to - open.from + 1),
		});
		open = null;
	};
	for (const [index, { label, score }] of predictions.entries()) {
		if (label === "O") {
			close(index - 1);
			continue;
		}
		const prefixed = /^([BI])-(.+)$/.exec(label);
		const tag = prefixed ? prefixed[2]! : label;
		if (open && open.tag === tag && prefixed?.[1] !== "B") {
			open.scoreSum += score;
			continue;
		}
		close(index - 1);
		open = { tag, from: index, scoreSum: score };
	}
	close(predictions.length - 1);
	return entities;
}
