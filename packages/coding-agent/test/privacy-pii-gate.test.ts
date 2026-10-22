import { afterEach, describe, expect, it, vi } from "bun:test";
import * as ai from "@oh-my-pi/pi-ai";
import { type Api, clearCustomApis, type Model, registerCustomApi } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import {
	type ExtensionUISelectItem,
	getExtensionUISelectOptionLabel,
} from "@oh-my-pi/pi-coding-agent/extensibility/extensions/types";
import { initializeExtensions } from "@oh-my-pi/pi-coding-agent/modes/runtime-init";
import { type PiiGateDeps, PiiGateBlockedError } from "@oh-my-pi/pi-coding-agent/privacy/pii-gate";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import type { AgentSession, AgentSessionEvent } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage } from "./helpers/agent-session-setup";

const API = "pii-gate-test-api";
const CLOUD = "acme-cloud/cloud-model";
const LOCAL = "acme-local/local-model";
const EMAIL = "dana@meridiancap.com";

interface Question {
	title: string;
	labels: string[];
}

interface Harness {
	session: AgentSession;
	/** Requests that reached a provider: the model id and the JSON of the messages it carried. */
	requests: { model: string; messages: string }[];
	/** Every text batch handed to the detector. */
	scans: string[][];
	questions: Question[];
	events: AgentSessionEvent[];
}

interface HarnessOptions {
	settings?: Record<string, unknown>;
	startOn?: typeof CLOUD | typeof LOCAL;
	/** Answers the gate's select; absent means the session has no prompt UI. */
	answer?: (question: Question) => string | undefined;
	interactivePrompts?: boolean;
	detect?: PiiGateDeps["detect"];
}

/** Detector stand-in: an e-mail address is the only personal data it knows. */
const detectEmails: NonNullable<PiiGateDeps["detect"]> = async texts =>
	texts.map(text => {
		const match = text.match(/[\w.]+@[\w.]+/);
		return match ? [{ label: "EMAIL_ADDRESS", text: match[0], score: 0.99 }] : [];
	});

const cleanups: Array<() => Promise<void> | void> = [];

afterEach(async () => {
	vi.restoreAllMocks();
	clearCustomApis();
	for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

async function startSession(options: HarnessOptions = {}): Promise<Harness> {
	const tempDir = TempDir.createSync("@pi-pii-gate-");
	cleanups.push(() => tempDir.removeSync());
	const requests: Harness["requests"] = [];
	registerCustomApi(API, (model, context) => {
		requests.push({ model: model.id, messages: JSON.stringify(context.messages) });
		const stream = new AssistantMessageEventStream();
		queueMicrotask(() => {
			stream.push({ type: "done", reason: "stop", message: createAssistantMessage("ok") });
		});
		return stream;
	});
	const authStorage = await AuthStorage.create(tempDir.join("auth.db"));
	cleanups.push(() => authStorage.close());
	const modelRegistry = new ModelRegistry(authStorage, tempDir.join("models.yml"));
	const providerModel = {
		api: API,
		reasoning: false,
		input: ["text" as const],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 1024,
	};
	modelRegistry.registerProvider("acme-cloud", {
		api: API,
		apiKey: "test-key",
		baseUrl: "https://api.acme.example/v1",
		models: [{ id: "cloud-model", name: "Cloud", ...providerModel }],
	});
	modelRegistry.registerProvider("acme-local", {
		api: API,
		apiKey: "test-key",
		baseUrl: "http://127.0.0.1:11434/v1",
		models: [{ id: "local-model", name: "Local", ...providerModel }],
	});
	const [provider, id] = (options.startOn ?? CLOUD).split("/");
	const model = modelRegistry.find(provider!, id!) as Model<Api>;

	const scans: string[][] = [];
	const detect = options.detect ?? detectEmails;
	const { session } = await createAgentSession({
		cwd: tempDir.path(),
		agentDir: tempDir.path(),
		sessionManager: SessionManager.inMemory(tempDir.path()),
		authStorage,
		modelRegistry,
		settings: Settings.isolated({
			"compaction.enabled": false,
			"retry.baseDelayMs": 1,
			// A fallback to the local model would deliver a request there; the gate must not trigger one.
			"retry.fallbackChains": { default: [LOCAL] },
			...options.settings,
		}),
		model,
		interactivePrompts: options.interactivePrompts,
		piiGateDeps: {
			detect: (texts, signal) => {
				scans.push([...texts]);
				return detect(texts, signal);
			},
			probeHardware: async () => ({
				platform: "linux",
				arch: "x64",
				totalMemBytes: 16 * 1024 ** 3,
				appleSilicon: false,
			}),
			recommendLocalModel: () => ({
				selector: "ollama/suggested:1b",
				pullCommand: "ollama pull suggested:1b",
				summary: "16 GB RAM",
			}),
		},
		disableExtensionDiscovery: true,
		skills: [],
		contextFiles: [],
		promptTemplates: [],
		slashCommands: [],
		enableMCP: false,
		enableLsp: false,
		skipPythonPreflight: true,
		taskDepth: 1,
	});
	cleanups.push(() => session.dispose());

	const questions: Question[] = [];
	const events: AgentSessionEvent[] = [];
	session.subscribe(event => events.push(event));
	const { answer } = options;
	if (answer) {
		await initializeExtensions(session, {
			reportSendError: () => {},
			reportRuntimeError: () => {},
			uiContext: Object.create(session.extensionRunner!.getUIContext(), {
				select: {
					value: async (title: string, items: ExtensionUISelectItem[]) => {
						const question = { title, labels: items.map(getExtensionUISelectOptionLabel) };
						questions.push(question);
						return answer(question);
					},
					// The runner wraps the context in a Proxy, which rejects a non-configurable member.
					configurable: true,
					writable: true,
				},
			}),
		});
	}
	return { session, requests, scans, questions, events };
}

const pick = (prefix: string) => (question: Question) => question.labels.find(label => label.startsWith(prefix));

function lastAssistant(session: AgentSession) {
	const message = session.agent.state.messages.findLast(candidate => candidate.role === "assistant");
	if (message?.role !== "assistant") throw new Error("Expected an assistant message");
	return message;
}

describe("PII privacy gate", () => {
	it("does not scan or ask while privacy.piiGate is off", async () => {
		const { session, requests, scans, questions } = await startSession({ answer: pick("Cancel") });

		await session.sendUserMessage(`Mail ${EMAIL} the report`);

		expect(scans).toEqual([]);
		expect(questions).toEqual([]);
		expect(requests.map(request => request.model)).toEqual(["cloud-model"]);
	});

	it("does not scan a model that stays on this machine", async () => {
		const { session, requests, scans, questions } = await startSession({
			startOn: LOCAL,
			settings: { "privacy.piiGate": true },
			answer: pick("Cancel"),
		});

		await session.sendUserMessage(`Mail ${EMAIL} the report`);

		expect(scans).toEqual([]);
		expect(questions).toEqual([]);
		expect(requests.map(request => request.model)).toEqual(["local-model"]);
	});

	it("sends text without personal data silently and scans each text once", async () => {
		const { session, requests, scans, questions } = await startSession({
			settings: { "privacy.piiGate": true },
			answer: pick("Cancel"),
		});

		await session.sendUserMessage("Summarize the repository");
		await session.sendUserMessage("Now the tests");

		expect(questions).toEqual([]);
		expect(requests).toHaveLength(2);
		expect(scans).toEqual([["Summarize the repository"], ["Now the tests"]]);
	});

	it("asks what was found and where, and does not ask again about text the user chose to send", async () => {
		const { session, requests, scans, questions } = await startSession({
			settings: { "privacy.piiGate": true },
			answer: pick("Send to"),
		});

		await session.sendUserMessage(`Mail ${EMAIL} the report`);

		expect(questions).toHaveLength(1);
		const { title, labels } = questions[0]!;
		expect(title).toContain("EMAIL_ADDRESS ×1 (d•••@meridiancap.com) in your prompt");
		expect(title).toContain(CLOUD);
		// No privacy.localModel yet: the question carries the hardware suggestion and offers no switch.
		expect(title).toContain("ollama pull suggested:1b");
		expect(labels).toEqual(["Cancel", `Send to ${CLOUD}`]);
		expect(requests.map(request => request.model)).toEqual(["cloud-model"]);
		expect(requests[0]!.messages).toContain(EMAIL);

		await session.sendUserMessage("Thanks");
		expect(questions).toHaveLength(1);
		expect(requests).toHaveLength(2);
		expect(scans.flat()).toEqual([`Mail ${EMAIL} the report`, "Thanks"]);

		// New personal data is a new question.
		await session.sendUserMessage("Also cc priya@meridiancap.com");
		expect(questions).toHaveLength(2);
	});

	it("scans tool output as well as the prompt", async () => {
		const { session, requests, questions } = await startSession({
			settings: { "privacy.piiGate": true },
			answer: pick("Cancel"),
		});
		session.agent.appendMessage({
			role: "toolResult",
			toolCallId: "read-1",
			toolName: "read",
			content: [{ type: "text", text: `owner: ${EMAIL}` }],
			isError: false,
			timestamp: Date.now(),
		});

		await session.sendUserMessage("continue");

		expect(questions).toHaveLength(1);
		expect(questions[0]!.title).toContain("EMAIL_ADDRESS ×1 (d•••@meridiancap.com) in read output");
		expect(requests).toEqual([]);
	});

	it("switches to the configured local model and sends the same request there", async () => {
		const { session, requests, scans, questions } = await startSession({
			settings: { "privacy.piiGate": true, "privacy.localModel": LOCAL },
			answer: pick("Switch to local model"),
		});

		await session.sendUserMessage(`Mail ${EMAIL} the report`);

		expect(questions[0]!.labels).toEqual([`Switch to local model ${LOCAL}`, "Cancel", `Send to ${CLOUD}`]);
		// The suggestion is only for users who have not configured a local model.
		expect(questions[0]!.title).not.toContain("suggested");
		expect(requests.map(request => request.model)).toEqual(["local-model"]);
		expect(requests[0]!.messages).toContain(EMAIL);
		expect(session.model?.id).toBe("local-model");

		// Now on a local model: nothing is scanned.
		await session.sendUserMessage("Also cc priya@meridiancap.com");
		expect(questions).toHaveLength(1);
		expect(scans).toHaveLength(1);
		expect(requests.map(request => request.model)).toEqual(["local-model", "local-model"]);
	});

	it("never offers a configured model that does not stay on this machine", async () => {
		const { session, questions } = await startSession({
			settings: { "privacy.piiGate": true, "privacy.localModel": CLOUD },
			answer: pick("Cancel"),
		});

		await session.sendUserMessage(`Mail ${EMAIL} the report`);

		expect(questions[0]!.labels).toEqual(["Cancel", `Send to ${CLOUD}`]);
		expect(questions[0]!.title).toContain(
			`privacy.localModel is "${CLOUD}", but that is not an available local model`,
		);
	});

	it("cancel sends nothing and ends the turn without retry, fallback or a provider error", async () => {
		const { session, requests, questions, events } = await startSession({
			settings: { "privacy.piiGate": true, "privacy.localModel": LOCAL },
			answer: pick("Cancel"),
		});

		await session.sendUserMessage(`Mail ${EMAIL} the report`);
		await session.waitForIdle();

		expect(questions).toHaveLength(1);
		expect(requests).toEqual([]);
		expect(
			events.filter(event => event.type.startsWith("auto_retry") || event.type.startsWith("retry_fallback")),
		).toEqual([]);
		expect(session.model?.id).toBe("cloud-model");
		const reply = lastAssistant(session);
		expect(reply.stopReason).toBe("aborted");
		expect(reply.errorMessage).toBe(
			`Privacy gate: request cancelled. EMAIL_ADDRESS ×1 (d•••@meridiancap.com) in your prompt was not sent to ${CLOUD}.`,
		);
	});

	it.each([
		["no prompt UI", {}],
		["a host whose prompts cannot reach a human", { interactivePrompts: false, answer: pick("Send to") }],
	] as const)("fails closed with a message when there is %s", async (_name, extra) => {
		const { session, requests, questions, events } = await startSession({
			settings: { "privacy.piiGate": true },
			...extra,
		});

		await session.sendUserMessage(`Mail ${EMAIL} the report`);
		await session.waitForIdle();

		expect(questions).toEqual([]);
		expect(requests).toEqual([]);
		expect(
			events.filter(event => event.type.startsWith("auto_retry") || event.type.startsWith("retry_fallback")),
		).toEqual([]);
		const reply = lastAssistant(session);
		expect(reply.stopReason).toBe("aborted");
		expect(reply.errorMessage).toContain("EMAIL_ADDRESS ×1 (d•••@meridiancap.com) in your prompt");
		expect(reply.errorMessage).toContain("no interactive prompt is available");
		expect(reply.errorMessage).toContain("privacy.piiGate");
	});

	it("does not send when the detector fails", async () => {
		const { session, requests } = await startSession({
			settings: { "privacy.piiGate": true },
			answer: pick("Send to"),
			detect: async () => {
				throw new Error("model not downloaded");
			},
		});

		await session.sendUserMessage("Summarize the repository");
		await session.waitForIdle();

		expect(requests).toEqual([]);
		expect(lastAssistant(session).errorMessage).toContain("could not scan for personal data (model not downloaded)");
	});

	it("gates side requests built from session messages but only offers send or cancel", async () => {
		const { session, questions, events } = await startSession({
			settings: { "privacy.piiGate": true, "privacy.localModel": LOCAL },
			answer: pick("Cancel"),
		});

		const side = session.convertMessagesToLlm([
			{ role: "user", content: `Mail ${EMAIL} the report`, attribution: "user", timestamp: Date.now() },
		]);

		await expect(side).rejects.toBeInstanceOf(PiiGateBlockedError);
		// A switch could not redirect a request that names its model up front.
		expect(questions[0]!.labels).toEqual(["Cancel", `Send to ${CLOUD}`]);
		// A refused side request fails alone; it does not abort the session's own turn.
		expect(events.filter(event => event.type === "agent_end")).toEqual([]);
	});

	it("titles a session from approved or clean text only", async () => {
		const completeSimple = vi.spyOn(ai, "completeSimple").mockResolvedValue({
			stopReason: "stop",
			content: [{ type: "text", text: "<title>Meridian Report</title>" }],
		} as never);
		const { session } = await startSession({
			settings: { "privacy.piiGate": true, modelRoles: { tiny: CLOUD } },
			answer: pick("Send to"),
		});
		const prompt = `Mail ${EMAIL} the report`;

		// Not approved yet: the title request would carry the address to the title model.
		expect(await session.generateTitle(prompt)).toBeNull();
		expect(completeSimple).not.toHaveBeenCalled();

		await session.sendUserMessage(prompt);
		expect(await session.generateTitle(prompt)).toBe("Meridian Report");
		expect(completeSimple).toHaveBeenCalledTimes(1);
	});
});
