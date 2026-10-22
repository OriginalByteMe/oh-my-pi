/**
 * Settings declared by this domain (see `config/registry.ts`). Declaration order is the
 * settings-panel order; `config/all-settings.ts` registers every domain.
 */
import { register } from "../config/registry";

// ────────────────────────────────────────────────────────────────────────
// Providers
// ────────────────────────────────────────────────────────────────────────

// PII privacy gate
export const cfgPrivacyPiiGate = register({
	id: "privacy.piiGate",
	type: "boolean",
	default: false,
	ui: {
		tab: "providers",
		group: "Privacy",
		label: "PII Gate",
		description:
			"Scan prompts and tool output for personal data with a small local model before sending to a cloud model, and ask whether to send, switch to a local model, or cancel",
	},
});

export const cfgPrivacyLocalModel = register({
	id: "privacy.localModel",
	type: "string",
	default: "",
	ui: {
		tab: "providers",
		group: "Privacy",
		label: "PII Gate Local Model",
		description:
			"Model (provider/id, like the model roles) the PII gate offers to switch to when personal data is found, e.g. ollama/gemma4:e4b; empty suggests one that fits this machine",
	},
});
