/**
 * PII privacy gate. Before a request goes to a model that is not on this machine, scan the
 * text of the messages it carries that has not been cleared yet (user prompts, tool output,
 * attached files) for personal data with a local detector. When something is found, ask the
 * user whether to send it anyway, switch to a local model, or cancel.
 *
 * The gate runs inside the session's `transformContext` (see `sdk.ts`), so every provider
 * request built from session messages passes it. A blocked request throws
 * {@link PiiGateBlockedError}; the caller turns that into a clean stop.
 */
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { Model } from "@oh-my-pi/pi-ai";
import { hasLocalLoopbackBaseUrl } from "@oh-my-pi/pi-catalog/compat/resolve";
import { logger } from "@oh-my-pi/pi-utils";
import type { ConfiguredThinkingLevel } from "@oh-my-pi/pi-tui/thinking";
import type { ExtensionUISelectItem } from "../extensibility/extensions/types";
import { textFromContent } from "../session/messages";
import { type HardwareProfile, type LocalModelAdvice, probeHardware, recommendLocalModel } from "./local-model-advice";
import { detectPii, type PiiFinding } from "./pii-detector";

/** The gate stopped a request: the user cancelled, nobody could be asked, or the scan failed. The message is user-facing. */
export class PiiGateBlockedError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "PiiGateBlockedError";
	}
}

/** A configured local model the gate may switch the session to. */
export interface PiiGateLocalModel {
	model: Model;
	thinkingLevel?: ConfiguredThinkingLevel;
}

export type PiiGateSelect = (
	title: string,
	options: ExtensionUISelectItem[],
	dialogOptions?: { signal?: AbortSignal },
) => Promise<string | undefined>;

/** Where the gate reads session state and asks the user; supplied by `sdk.ts`. */
export interface PiiGateHost {
	/** `privacy.piiGate`, read per request. */
	enabled(): boolean;
	/** The model the next request goes to. */
	model(): Model | undefined;
	/** `privacy.localModel` as configured; empty when unset. */
	localModelSelector(): string;
	/** The configured local model when it names an available model that stays on this machine. */
	resolveLocalModel(): PiiGateLocalModel | undefined;
	/** The dialog a human can answer, or undefined when there is none (print, RPC/ACP without prompts, subagents). */
	select(): PiiGateSelect | undefined;
	/** Switch the session to `local` (session-only). */
	switchModel(local: PiiGateLocalModel): Promise<void>;
}

/** Replaceable pieces for tests and embedders; the defaults are the local ONNX detector and the real hardware probe. */
export interface PiiGateDeps {
	detect?: (texts: readonly string[], signal?: AbortSignal) => Promise<PiiFinding[][]>;
	probeHardware?: () => Promise<HardwareProfile>;
	recommendLocalModel?: (hw: HardwareProfile) => LocalModelAdvice;
}

export interface PiiGateCheckOptions {
	signal?: AbortSignal;
	/**
	 * Whether the request follows a model switch made here. True for the main loop, which resolves its
	 * model after the context transform; false for side requests that name their model up front, where
	 * switching the session could not redirect the request, so only send/cancel are offered.
	 */
	redirectable: boolean;
}

interface Pending {
	hash: number | bigint;
	text: string;
	sources: Set<string>;
	findings: PiiFinding[];
}

/** Whether requests to `model` stay on this machine or LAN: an in-process model or a loopback/private-network base URL. */
export function isLocalModel(model: Pick<Model, "api" | "baseUrl">): boolean {
	return model.api === "local-inference" || hasLocalLoopbackBaseUrl(model.baseUrl);
}

// lean: user prompts, tool output, shell/python output, attached files and compaction/branch summaries only.
// Custom and hook messages are skipped (mostly static reminders that would trip the detector every session);
// add roles here if one is found carrying user data.
function scannableTexts(message: AgentMessage): { source: string; text: string }[] {
	let source: string;
	let text: string;
	switch (message.role) {
		case "user":
			source = "your prompt";
			text = textFromContent(message.content);
			break;
		case "toolResult":
			source = `${message.toolName} output`;
			text = textFromContent(message.content);
			break;
		case "bashExecution":
			if (message.excludeFromContext) return [];
			source = "command output";
			text = `${message.command}\n${message.output}`.trim();
			break;
		case "pythonExecution":
			if (message.excludeFromContext) return [];
			source = "python output";
			text = `${message.code}\n${message.output}`.trim();
			break;
		case "compactionSummary":
		case "branchSummary":
			source = "session summary";
			text = message.summary.trim();
			break;
		case "fileMention":
			return message.files
				.filter(file => file.content)
				.map(file => ({ source: "attached files", text: file.content }));
		default:
			return [];
	}
	return text ? [{ source, text }] : [];
}

/** First character and the tail, so the user recognizes the value without it landing in the transcript. */
function mask(text: string): string {
	const at = text.indexOf("@");
	if (at > 0) return `${text[0]}•••${text.slice(at)}`;
	return text.length > 4 ? `${text[0]}•••${text.slice(-2)}` : "•••";
}

/** `EMAIL_ADDRESS ×2 (d•••@meridiancap.com), PHONE_NUMBER ×1 (+•••45)` */
function summarizeFindings(pending: readonly Pending[]): string {
	const byLabel = new Map<string, { count: number; sample: string }>();
	for (const entry of pending) {
		for (const finding of entry.findings) {
			const seen = byLabel.get(finding.label);
			if (seen) seen.count++;
			else byLabel.set(finding.label, { count: 1, sample: mask(finding.text) });
		}
	}
	return [...byLabel].map(([label, { count, sample }]) => `${label} ×${count} (${sample})`).join(", ");
}

export class PiiGate {
	readonly #host: PiiGateHost;
	readonly #detect: NonNullable<PiiGateDeps["detect"]>;
	readonly #probeHardware: NonNullable<PiiGateDeps["probeHardware"]>;
	readonly #recommend: NonNullable<PiiGateDeps["recommendLocalModel"]>;
	/** Hashes of texts that were scanned clean or that the user approved; keyed by content, since messages are re-created. */
	// lean: in memory per session; a resumed session rescans once, persist the set if that proves annoying.
	readonly #cleared = new Set<number | bigint>();
	/** Serializes checks so concurrent requests never ask twice about the same text. */
	#tail: Promise<void> = Promise.resolve();
	#advice: Promise<string | undefined> | undefined;

	constructor(host: PiiGateHost, deps: PiiGateDeps = {}) {
		this.#host = host;
		this.#detect = deps.detect ?? detectPii;
		this.#probeHardware = deps.probeHardware ?? probeHardware;
		this.#recommend = deps.recommendLocalModel ?? recommendLocalModel;
	}

	/**
	 * Resolve when `messages` may go to the active model: the gate is off, the model is local, nothing new
	 * has personal data, or the user chose to send or switch. Throws {@link PiiGateBlockedError} when the
	 * request must not be sent.
	 */
	check(messages: readonly AgentMessage[], options: PiiGateCheckOptions): Promise<void> {
		const run = this.#tail.then(() => this.#check(messages, options));
		this.#tail = run.catch(() => {});
		return run;
	}

	/**
	 * Whether `text` may go to a model the gate does not see (session titles): already cleared, or scanned
	 * clean now. Never asks; text with personal data that the user has not approved stays local, and so does
	 * text that could not be scanned.
	 */
	async clears(text: string, signal?: AbortSignal): Promise<boolean> {
		if (!this.#host.enabled()) return true;
		const hash = Bun.hash(text);
		if (this.#cleared.has(hash)) return true;
		try {
			const [findings] = await this.#detect([text], signal);
			if (findings === undefined || findings.length > 0) return false;
		} catch (error) {
			logger.warn("privacy gate: scan failed; text stays local", { error: String(error) });
			return false;
		}
		this.#cleared.add(hash);
		return true;
	}

	async #check(messages: readonly AgentMessage[], options: PiiGateCheckOptions): Promise<void> {
		if (!this.#host.enabled()) return;
		const model = this.#host.model();
		if (!model || isLocalModel(model)) return;

		const pending = this.#uncleared(messages);
		if (pending.length === 0) return;
		const target = `${model.provider}/${model.id}`;
		await this.#scan(pending, target, options.signal);
		const found = pending.filter(entry => entry.findings.length > 0);
		if (found.length === 0) {
			this.#clear(pending);
			return;
		}

		const where = [...new Set(found.flatMap(entry => [...entry.sources]))].join(", ");
		const what = `${summarizeFindings(found)} in ${where}`;
		const select = this.#host.select();
		if (!select) {
			throw new PiiGateBlockedError(
				`Privacy gate: found ${what}. The request to ${target} was not sent because no interactive prompt is available. Turn off privacy.piiGate, or run with a local model.`,
			);
		}

		const local = options.redirectable ? this.#host.resolveLocalModel() : undefined;
		const selector = this.#host.localModelSelector();
		const sendLabel = `Send to ${target}`;
		const switchLabel = `Switch to local model ${selector}`;
		// Safe choices first: the highlighted default (Enter) must never send.
		const choices: ExtensionUISelectItem[] = [];
		if (local) {
			choices.push({
				label: switchLabel,
				description: "Keeps it on this machine; this request goes to the local model.",
			});
		}
		choices.push(
			{ label: "Cancel", description: "Nothing is sent." },
			{ label: sendLabel, description: "Sends it; this text is not asked about again this session." },
		);

		const hint = local ? undefined : await this.#localModelHint(selector, options.redirectable);
		const question = [
			`Personal data found: ${what}`,
			`Sending this request to ${target} would share it with that provider.`,
			hint,
		]
			.filter(Boolean)
			.join("\n");
		const choice = await select(question, choices, { signal: options.signal });

		if (choice === sendLabel) {
			this.#clear(pending);
			return;
		}
		if (local && choice === switchLabel) {
			try {
				await this.#host.switchModel(local);
			} catch (error) {
				throw new PiiGateBlockedError(
					`Privacy gate: could not switch to ${selector} (${error instanceof Error ? error.message : String(error)}). The request to ${target} was not sent.`,
				);
			}
			return;
		}
		throw new PiiGateBlockedError(`Privacy gate: request cancelled. ${what} was not sent to ${target}.`);
	}

	/** Texts in `messages` that are not cleared yet, one entry per distinct text. */
	#uncleared(messages: readonly AgentMessage[]): Pending[] {
		const pending = new Map<number | bigint, Pending>();
		for (const message of messages) {
			for (const { source, text } of scannableTexts(message)) {
				const hash = Bun.hash(text);
				if (this.#cleared.has(hash)) continue;
				const entry = pending.get(hash);
				if (entry) entry.sources.add(source);
				else pending.set(hash, { hash, text, sources: new Set([source]), findings: [] });
			}
		}
		return [...pending.values()];
	}

	async #scan(pending: Pending[], target: string, signal: AbortSignal | undefined): Promise<void> {
		let results: PiiFinding[][];
		try {
			results = await this.#detect(
				pending.map(entry => entry.text),
				signal,
			);
			if (results.length !== pending.length) {
				throw new Error("detector returned a result count that does not match its input");
			}
		} catch (error) {
			if (signal?.aborted) throw error;
			logger.warn("privacy gate: scan failed; request not sent", { error: String(error) });
			throw new PiiGateBlockedError(
				`Privacy gate: could not scan for personal data (${error instanceof Error ? error.message : String(error)}). The request to ${target} was not sent. Turn off privacy.piiGate to send without scanning.`,
			);
		}
		for (const [index, entry] of pending.entries()) entry.findings = results[index]!;
	}

	#clear(pending: readonly Pending[]): void {
		for (const entry of pending) this.#cleared.add(entry.hash);
	}

	/** Why no local model is offered; with none configured, a suggestion that fits this machine. */
	async #localModelHint(selector: string, redirectable: boolean): Promise<string | undefined> {
		if (selector) {
			return redirectable
				? `privacy.localModel is "${selector}", but that is not an available local model.`
				: undefined;
		}
		this.#advice ??= this.#adviseLocalModel();
		return await this.#advice;
	}

	async #adviseLocalModel(): Promise<string | undefined> {
		try {
			const advice = this.#recommend(await this.#probeHardware());
			return `This machine: ${advice.summary}. Suggested local model: ${advice.selector} — run \`${advice.pullCommand}\`, then set privacy.localModel.`;
		} catch (error) {
			logger.debug("privacy gate: hardware probe failed", { error: String(error) });
			return undefined;
		}
	}
}
