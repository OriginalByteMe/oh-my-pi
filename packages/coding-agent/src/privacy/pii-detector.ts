/**
 * Local personal-data detection for the privacy gate. Runs a small ONNX token-classification model
 * (`TINY_CLASSIFIER_LOCAL_MODELS`) in the shared tiny worker, so text is classified on this machine
 * and never reaches a provider. Policy lives here (which labels matter, how confident); the worker
 * only labels spans.
 */
import * as net from "node:net";
import { tinyModelClient } from "../tiny/title-client";
import type { TinyEntity } from "../tiny/title-protocol";

export interface PiiFinding {
	/** Entity label without BIO prefix, e.g. "EMAIL_ADDRESS", "PERSON". */
	label: string;
	/** Detected surface text, as written in the input. */
	text: string;
	/** Mean token score for the span, 0..1. */
	score: number;
}

const PII_MODEL_KEY = "bert-small-pii";

// lean: fixed list of labels that are ambient noise in a coding session (dates and URLs are everywhere,
// organizations/titles/nationalities appear in every README); `O` is the model's "nothing" label.
// Upgrade: make it a setting if users need URLs or organizations flagged.
const IGNORED_LABELS: Record<string, true> = {
	O: true,
	DATE_TIME: true,
	URL: true,
	ORGANIZATION: true,
	TITLE: true,
	NRP: true,
};

// lean: one global confidence floor. Drops the weak spans the model emits on ordinary prose ("brown fox"
// as PERSON, 0.50-0.58 on repeated filler) and one-token stragglers (a stray `ss` scoring ~0.44 as a
// phone number) while keeping real spans (0.75-0.99 in the smoke run). Upgrade: per-label thresholds
// once labelled coding-session data exists to tune them.
const MIN_SCORE = 0.6;

/** Drop a trailing `:port` and IPv6 brackets: `10.0.0.1:8080`, `[::1]:3000` -> the bare address. */
function stripPort(text: string): string {
	const bracketed = /^\[([^\]]+)\](?::\d+)?$/.exec(text);
	if (bracketed) return bracketed[1]!;
	return /^(\d{1,3}(?:\.\d{1,3}){3}):\d+$/.exec(text)?.[1] ?? text;
}

/**
 * Loopback, private, link-local and documentation addresses (RFC 1122/1918/3927/5737/4193/4291). They
 * name no one, and dev-server logs and test fixtures are full of them.
 */
function isNonPersonalIp(text: string): boolean {
	const address = stripPort(text.trim().replace(/[.,;]+$/, ""));
	const family = net.isIP(address);
	if (family === 4) {
		const [a, b, c] = address.split(".").map(Number) as [number, number, number, number];
		return (
			a === 127 ||
			a === 10 ||
			(a === 172 && b >= 16 && b <= 31) ||
			(a === 192 && b === 168) ||
			(a === 169 && b === 254) ||
			(a === 192 && b === 0 && c === 2) ||
			(a === 198 && b === 51 && c === 100) ||
			(a === 203 && b === 0 && c === 113)
		);
	}
	if (family === 6) {
		const first = Number.parseInt(address.split(":")[0] || "0", 16);
		return (
			address === "::1" ||
			/^(0{1,4}:){7}0{0,3}1$/.test(address) ||
			(first & 0xfe00) === 0xfc00 || // fc00::/7 unique local
			(first & 0xffc0) === 0xfe80 // fe80::/10 link-local
		);
	}
	return false;
}

/** Keep the spans worth interrupting a coding session for. */
export function toPiiFindings(entities: readonly TinyEntity[]): PiiFinding[] {
	return entities
		.filter(
			entity =>
				IGNORED_LABELS[entity.label] !== true &&
				entity.score >= MIN_SCORE &&
				!(entity.label === "IP_ADDRESS" && isNonPersonalIp(entity.text)),
		)
		.map(({ label, text, score }) => ({ label, text, score }));
}

/**
 * Scan `texts` for personal data with the local PII model. Returns one finding array per input, same
 * order; an empty array means nothing found. Downloads the model (~29 MB) on first use.
 * Throws when the worker cannot load or run the model, or when `signal` aborts: callers must treat a
 * throw as "not scanned", never as "clean".
 */
export async function detectPii(texts: readonly string[], signal?: AbortSignal): Promise<PiiFinding[][]> {
	if (texts.length === 0) return [];
	const entities = await tinyModelClient.classify(PII_MODEL_KEY, texts, signal);
	return entities.map(toPiiFindings);
}
