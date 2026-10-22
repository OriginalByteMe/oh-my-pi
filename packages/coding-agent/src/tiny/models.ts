/** Local model the `tiny-models` CLI downloads when none is named. */
export const DEFAULT_TINY_TITLE_LOCAL_MODEL_KEY = "lfm2.5-230m";

export interface TinyTitleLocalModelSpec {
	key: string;
	/** ONNX export loaded by transformers.js on every platform. */
	repo: string;
	dtype: "q4" | "q8";
	/**
	 * Pipeline the ONNX worker loads; omitted means `text-generation` (chat/title models).
	 * A `token-classification` model only answers `classify` requests, never `chat`.
	 */
	task?: "token-classification";
	/** Pre-quantized MLX export loaded by mlx-lm when `PI_TINY_DEVICE=mlx`; absent for ONNX-only models. */
	mlxRepo?: string;
	label: string;
	description: string;
	contextNote: string;
	/** Model family emits hidden reasoning unless the chat template disables it. */
	reasoning?: boolean;
	/** Reason the ONNX backend refuses this model before loading the runtime; the MLX backend ignores it. */
	onnxUnsupportedReason?: string;
}

export const TINY_TITLE_LOCAL_MODELS = [
	{
		key: "lfm2.5-230m",
		repo: "LiquidAI/LFM2.5-230M-ONNX",
		dtype: "q4",
		mlxRepo: "LiquidAI/LFM2.5-230M-MLX-4bit",
		label: "LFM2.5 230M",
		description: "Recommended local model; fastest LFM2.5 option, about 214 MB cached.",
		contextNote: "Best balance among the current compact title models.",
	},
	{
		key: "lfm2.5-350m",
		repo: "onnx-community/LFM2.5-350M-ONNX",
		dtype: "q4",
		mlxRepo: "LiquidAI/LFM2.5-350M-MLX-4bit",
		label: "LFM2.5 350M",
		description: "Larger LFM2.5 option, about 292 MB cached; tends toward terse titles.",
		contextNote: "Use when compact labels are preferred over descriptive titles.",
	},
	{
		key: "falcon-h1-90m",
		repo: "onnx-community/Falcon-H1-Tiny-90M-Instruct-ONNX",
		dtype: "q4",
		mlxRepo: "mlx-community/Falcon-H1-Tiny-90M-Instruct-4bit",
		label: "Falcon H1 Tiny 90M",
		description: "Smallest option, about 147 MB cached; lower fidelity on complex prompts.",
		contextNote: "Use on constrained machines where download size matters most.",
	},
] as const satisfies readonly TinyTitleLocalModelSpec[];

export type TinyTitleLocalModelKey = (typeof TINY_TITLE_LOCAL_MODELS)[number]["key"];

export function isTinyTitleLocalModelKey(value: string): value is TinyTitleLocalModelKey {
	return TINY_TITLE_LOCAL_MODELS.some(model => model.key === value);
}

export function getTinyTitleModelSpec(key: TinyTitleLocalModelKey): (typeof TINY_TITLE_LOCAL_MODELS)[number] {
	const spec = TINY_TITLE_LOCAL_MODELS.find(model => model.key === key);
	if (!spec) throw new Error(`Unknown tiny title model: ${key}`);
	return spec;
}

/** Recommended local model for memory tasks when none is named. */
export const DEFAULT_MEMORY_LOCAL_MODEL_KEY = "lfm2-1.2b";

/**
 * Local models for Mnemopi memory tasks (fact extraction + consolidation).
 * These are larger (1B-1.7B) than the title models: structured extraction and
 * faithful summarization need more capacity than 3-6 word titles. All q4.
 * Ranking/recipe rationale lives in docs/local-models.md.
 */
export const TINY_MEMORY_LOCAL_MODELS = [
	{
		key: "qwen3-1.7b",
		repo: "onnx-community/Qwen3-1.7B-ONNX",
		dtype: "q4",
		mlxRepo: "mlx-community/Qwen3-1.7B-4bit",
		label: "Qwen3 1.7B",
		description:
			"MLX only (providers.tinyModelDevice=mlx): onnxruntime-node cannot run this ONNX export's RotaryEmbedding cache updates.",
		contextNote: "Blocked on the ONNX backend to avoid the unsupported RotaryEmbedding runtime path.",
		reasoning: true,
		onnxUnsupportedReason:
			"onnxruntime-node does not support Qwen3 RotaryEmbedding cache updates in onnx-community/Qwen3-1.7B-ONNX",
	},
	{
		key: "llama3.2:3b",
		repo: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
		dtype: "q4",
		mlxRepo: "mlx-community/Llama-3.2-3B-Instruct-4bit",
		label: "Llama 3.2 3B",
		description:
			"Larger Llama 3.2 option for local memory/classifier tasks; higher quality potential at higher disk/RAM/latency cost.",
		contextNote: "Use when larger model capacity is preferred over faster load times.",
	},
	{
		key: "gemma-3-1b",
		repo: "onnx-community/gemma-3-1b-it-ONNX",
		dtype: "q4",
		mlxRepo: "mlx-community/gemma-3-1b-it-4bit",
		label: "Gemma 3 1B",
		description: "Best consolidation/dedup; lighter footprint, but leaks small talk during extraction.",
		contextNote: "Use when consolidation quality and size matter most.",
	},
	{
		key: "qwen2.5-1.5b",
		repo: "onnx-community/Qwen2.5-1.5B-Instruct",
		dtype: "q4",
		mlxRepo: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
		label: "Qwen2.5 1.5B",
		description: "Best extraction granularity (atomic facts); weaker consolidation.",
		contextNote: "Use when fine-grained, deduplicatable facts matter more than summaries.",
	},
	{
		key: "lfm2-1.2b",
		repo: "onnx-community/LFM2-1.2B-ONNX",
		dtype: "q4",
		mlxRepo: "mlx-community/LFM2-1.2B-4bit",
		label: "LFM2 1.2B",
		description: "Fastest load; solid all-rounder, slightly noisier extraction labels.",
		contextNote: "Use when local startup cost is the priority.",
	},
] as const satisfies readonly TinyTitleLocalModelSpec[];

export type TinyMemoryLocalModelKey = (typeof TINY_MEMORY_LOCAL_MODELS)[number]["key"];

export function isTinyMemoryLocalModelKey(value: string): value is TinyMemoryLocalModelKey {
	return TINY_MEMORY_LOCAL_MODELS.some(model => model.key === value);
}

export function getTinyMemoryModelSpec(key: TinyMemoryLocalModelKey): (typeof TINY_MEMORY_LOCAL_MODELS)[number] {
	const spec = TINY_MEMORY_LOCAL_MODELS.find(model => model.key === key);
	if (!spec) throw new Error(`Unknown tiny memory model: ${key}`);
	return spec;
}

/** Return whether a memory local model may emit reasoning tokens before answers. */
export function isTinyMemoryReasoningModelKey(key: TinyMemoryLocalModelKey): boolean {
	const spec = getTinyMemoryModelSpec(key);
	return "reasoning" in spec && spec.reasoning === true;
}

/**
 * Token-classification (NER) models served by the same ONNX worker. ONNX-only: no `mlxRepo`.
 * The shipped dtype is fixed; `PI_TINY_DTYPE` does not apply because quantizing a PII detector
 * to the LLM-oriented q4/q2 levels would silently cost recall.
 */
export const TINY_CLASSIFIER_LOCAL_MODELS = [
	{
		key: "bert-small-pii",
		repo: "onnx-community/bert-small-pii-detection-ONNX",
		dtype: "q8",
		task: "token-classification",
		label: "BERT small PII detector",
		description: "Local personal-data detector for the privacy gate (privacy.piiGate); about 29 MB cached.",
		contextNote: "Not a chat model: labels names, emails, phone numbers, SSNs and similar spans.",
	},
] as const satisfies readonly TinyTitleLocalModelSpec[];

export type TinyClassifierLocalModelKey = (typeof TINY_CLASSIFIER_LOCAL_MODELS)[number]["key"];

/** Any local model key (title, memory, or classifier), used by the shared inference worker. */
export type TinyLocalModelKey = TinyTitleLocalModelKey | TinyMemoryLocalModelKey | TinyClassifierLocalModelKey;

/** Resolve a local model spec by key across the title, memory and classifier registries. */
export function getTinyLocalModelSpec(key: string): TinyTitleLocalModelSpec | undefined {
	return (
		TINY_TITLE_LOCAL_MODELS.find(model => model.key === key) ??
		TINY_MEMORY_LOCAL_MODELS.find(model => model.key === key) ??
		TINY_CLASSIFIER_LOCAL_MODELS.find(model => model.key === key)
	);
}

export function isTinyLocalModelKey(value: string): value is TinyLocalModelKey {
	return getTinyLocalModelSpec(value) !== undefined;
}

/** Combined local model registry (title + memory + classifier) for the shared tiny-models CLI. */
export const TINY_LOCAL_MODELS = [
	...TINY_TITLE_LOCAL_MODELS,
	...TINY_MEMORY_LOCAL_MODELS,
	...TINY_CLASSIFIER_LOCAL_MODELS,
] as const satisfies readonly TinyTitleLocalModelSpec[];
