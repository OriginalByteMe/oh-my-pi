import { describe, expect, it } from "bun:test";
import { toPiiFindings } from "@oh-my-pi/pi-coding-agent/privacy/pii-detector";
import {
	alignTokens,
	mergeTokenSpans,
	planTokenWindows,
	type TokenPrediction,
} from "@oh-my-pi/pi-coding-agent/tiny/token-classification";

/** Run the worker's merge path over `[wordpiece, label, score?]` rows laid over `text`. */
function merge(text: string, rows: ReadonlyArray<readonly [string, string, number?]>) {
	const tokens = rows.map(([token]) => token);
	const predictions: TokenPrediction[] = rows.map(([, label, score]) => ({ label, score: score ?? 0.9 }));
	return mergeTokenSpans(text, tokens, alignTokens(text, tokens), predictions);
}

describe("token-classification span merging", () => {
	it("merges BIO sub-word tokens into spans of the original text", () => {
		// Tokens as the bert-small-pii wordpiece tokenizer emits them (lower-cased, `##` continuations).
		const text = "Hi, I'm Daniel Whitfield, reach me at daniels@meridiancap.com or 415-555-0123.";
		const spans = merge(text, [
			["hi", "O"],
			[",", "O"],
			["i", "O"],
			["'", "O"],
			["m", "O"],
			["daniel", "B-PERSON"],
			["w", "I-PERSON", 0.5],
			["##hit", "I-PERSON"],
			["##field", "I-PERSON"],
			[",", "O"],
			["reach", "O"],
			["me", "O"],
			["at", "O"],
			["daniels", "B-EMAIL_ADDRESS"],
			["@", "I-EMAIL_ADDRESS"],
			["meridian", "I-EMAIL_ADDRESS"],
			["##cap", "I-EMAIL_ADDRESS"],
			[".", "I-EMAIL_ADDRESS"],
			["com", "I-EMAIL_ADDRESS"],
			["or", "O"],
			["415", "B-PHONE_NUMBER"],
			["-", "I-PHONE_NUMBER"],
			["555", "I-PHONE_NUMBER"],
			["-", "I-PHONE_NUMBER"],
			["01", "I-PHONE_NUMBER"],
			["##23", "I-PHONE_NUMBER"],
			[".", "O"],
		]);

		// Surface text keeps the input's casing and spacing (the decoded tokens would read `daniels @ meridiancap. com`).
		expect(spans.map(({ label, text }) => [label, text])).toEqual([
			["PERSON", "Daniel Whitfield"],
			["EMAIL_ADDRESS", "daniels@meridiancap.com"],
			["PHONE_NUMBER", "415-555-0123"],
		]);
		expect(spans[0]!.score).toBeCloseTo((0.9 + 0.5 + 0.9 + 0.9) / 4);
	});

	it("splits spans on B- and on tag change, and starts one at an orphan I-", () => {
		expect(
			merge("ann bo", [
				["ann", "B-PERSON"],
				["bo", "B-PERSON"],
			]).map(span => span.text),
		).toEqual(["ann", "bo"]);
		expect(
			merge("ann paris", [
				["ann", "I-PERSON"],
				["paris", "I-LOCATION"],
			]).map(span => [span.label, span.text]),
		).toEqual([
			["PERSON", "ann"],
			["LOCATION", "paris"],
		]);
	});

	it("widens a span to the whole word and falls back to the tokens when they cannot be located", () => {
		// The model tagged only the `##2` piece of `hunter2`.
		expect(
			merge("hunter2!", [
				["hunter", "O"],
				["##2", "I-PASSWORD", 0.7],
				["!", "O"],
			]),
		).toEqual([{ label: "PASSWORD", text: "hunter2", score: 0.7 }]);

		const unlocated = mergeTokenSpans(
			"?",
			["meridian", "##cap"],
			[null, null],
			[
				{ label: "B-ORGANIZATION", score: 0.9 },
				{ label: "I-ORGANIZATION", score: 0.9 },
			],
		);
		expect(unlocated.map(span => span.text)).toEqual(["meridiancap"]);
	});
});

describe("alignTokens", () => {
	it("maps accent-stripped tokens and skips unknown words", () => {
		expect(alignTokens("José García", ["jose", "garcia"])).toEqual([
			[0, 4],
			[5, 11],
		]);
		expect(alignTokens("hi \u{1F600} bob", ["hi", "[UNK]", "bob"])).toEqual([[0, 2], null, [6, 9]]);
	});
});

describe("planTokenWindows", () => {
	it.each([0, 1, 510, 511, 2000])("tiles %d tokens exactly once, within the 510-token limit", count => {
		const windows = planTokenWindows(count);
		let next = 0;
		for (const window of windows) {
			expect(window.end - window.start).toBeLessThanOrEqual(510);
			expect(window.keepStart).toBe(next);
			expect(window.keepStart).toBeGreaterThanOrEqual(window.start);
			expect(window.keepEnd).toBeLessThanOrEqual(window.end);
			next = window.keepEnd;
		}
		expect(next).toBe(count);
	});
});

describe("toPiiFindings", () => {
	it("drops noise labels and low-confidence spans", () => {
		expect(
			toPiiFindings([
				{ label: "PERSON", text: "Daniel Whitfield", score: 0.99 },
				{ label: "DATE_TIME", text: "tomorrow", score: 0.99 },
				{ label: "URL", text: "https://example.com", score: 0.99 },
				{ label: "ORGANIZATION", text: "Meridian", score: 0.99 },
				{ label: "PHONE_NUMBER", text: "ss", score: 0.44 },
				{ label: "US_SSN", text: "123-45-6789", score: 0.85 },
			]),
		).toEqual([
			{ label: "PERSON", text: "Daniel Whitfield", score: 0.99 },
			{ label: "US_SSN", text: "123-45-6789", score: 0.85 },
		]);
	});

	it.each([
		"127.0.0.1",
		"127.0.0.1:8080",
		"127.255.255.254",
		"::1",
		"[::1]:3000",
		"10.0.0.5",
		"172.16.0.1",
		"172.31.255.255:22",
		"192.168.4.17",
		"169.254.169.254",
		"fc00::1",
		"fd12:3456:789a::1",
		"fe80::1",
		"febf::1",
		"192.0.2.10",
		"198.51.100.7:443",
		"203.0.113.9",
	])("drops the non-personal IP_ADDRESS %s", text => {
		expect(toPiiFindings([{ label: "IP_ADDRESS", text, score: 0.95 }])).toEqual([]);
	});

	it.each(["8.8.8.8", "8.8.8.8:53", "172.15.0.1", "172.32.0.1", "192.169.0.1", "2001:4860:4860::8888", "fec0::1"])(
		"keeps the public IP_ADDRESS %s",
		text => {
			expect(toPiiFindings([{ label: "IP_ADDRESS", text, score: 0.95 }])).toEqual([
				{ label: "IP_ADDRESS", text, score: 0.95 },
			]);
		},
	);
});
