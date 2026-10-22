/**
 * ONNX tiny-model worker: one process per local model, owning the model's
 * socket (see `title-protocol.ts`), serving every omp process on the machine,
 * and exiting on its own once idle. Entered from `cli.ts` via
 * {@link TINY_WORKER_ARG} with the socket/model/tag env set by
 * `title-client.ts`. Runs `onnxruntime-node` outside every omp process so its
 * NAPI finalizer never runs in a shared address space.
 */
import * as path from "node:path";
import type {
	PretrainedConfig,
	ProgressInfo,
	Tensor,
	TextGenerationPipeline,
	TextGenerationStringOutput,
	TokenClassificationPipeline,
	StoppingCriteria as TransformersStoppingCriteria,
} from "@huggingface/transformers";
import { getTinyModelsCacheDir, logger, setProcessName } from "@oh-my-pi/pi-utils";
import {
	errorMessage,
	errorText,
	formatOnnxRuntimeCudaDiagnostics,
	getTransformersVersionSpec,
	loadTransformersRuntime,
	MemoizedRuntime,
	sendProgress,
	type TransformersRuntimeMetadata,
} from "../subprocess/worker-runtime";
import { renderTextChatTemplate } from "./completion-prompt";
import {
	resolveTinyModelDevicePreference,
	type TinyModelDevicePreference,
	type TinyOnnxDevice,
	tinyModelDeviceLoadOrder,
} from "./device";
import { resolveTinyModelDtypeOverride, type TinyModelDtype } from "./dtype";
import {
	getTinyLocalModelSpec,
	isTinyLocalModelKey,
	type TinyLocalModelKey,
	type TinyTitleLocalModelSpec,
} from "./models";
import { alignTokens, mergeTokenSpans, planTokenWindows, type TokenPrediction } from "./token-classification";
import {
	TINY_WORKER_IDLE_MS,
	TINY_WORKER_IDLE_MS_ENV,
	TINY_WORKER_MODEL_ENV,
	TINY_WORKER_SOCKET_ENV,
	TINY_WORKER_TAG_ENV,
	type TinyEntity,
	type TinyWorkerRequest,
	type TinyWorkerResponse,
} from "./title-protocol";
import { TinyWorkerServer } from "./worker-server";

const STOP_DECODE_WINDOW_TOKENS = 32;
/** Windows run per forward pass; bounds the `[batch, 512, labels]` logits tensor. */
const CLASSIFY_BATCH_SIZE = 16;

export interface TransformersRuntime extends TransformersRuntimeMetadata {
	env: {
		cacheDir?: string;
		allowLocalModels?: boolean;
		logLevel?: unknown;
	};
	LogLevel: {
		ERROR: unknown;
	};
	StoppingCriteria: new () => TransformersStoppingCriteria;
	Tensor: typeof Tensor;
	pipeline: {
		(task: "text-generation", model: string, options: PipelineOptions): Promise<TextGenerationPipeline>;
		(task: "token-classification", model: string, options: PipelineOptions): Promise<TokenClassificationPipeline>;
	};
}

interface PipelineOptions {
	device: TinyOnnxDevice;
	dtype: TinyModelDtype;
	progress_callback: (info: ProgressInfo) => void;
}

/** The model's resident pipeline; `task` mirrors the spec so callers narrow without casts. */
type LoadedPipeline =
	| { task: "text-generation"; pipeline: TextGenerationPipeline }
	| { task: "token-classification"; pipeline: TokenClassificationPipeline };

/** Minimal outbound surface the shared progress/runtime helpers need for one request. */
interface ReplyTransport {
	send(message: TinyWorkerResponse): void;
}

function getTinyTitleRuntimeDir(): string {
	return path.join(
		path.dirname(getTinyModelsCacheDir()),
		"tiny-title-runtime",
		`transformers-${getTransformersVersionSpec().replace(/[^A-Za-z0-9._-]/g, "_")}`,
	);
}

/** Stops generation at the first occurrence of `text` in the *generated* tokens.
 *
 *  The window must be anchored to the generation boundary, not to the end of the
 *  whole sequence: a prompt that itself contains the stop string (chat-level
 *  few-shot examples ending in `</title>`, for instance) would otherwise match on
 *  prompt tokens and stop before the model emits anything. */
export function createStopOnTextCriteria(
	transformers: TransformersRuntime,
	tokenizer: TextGenerationPipeline["tokenizer"],
	text: string,
): TransformersStoppingCriteria {
	class StopOnTextCriteria extends transformers.StoppingCriteria {
		#tokenizer: TextGenerationPipeline["tokenizer"];
		#text: string;
		/** First generated index per batch entry, captured on the first call. */
		#generatedStarts: number[] = [];

		constructor() {
			super();
			this.#tokenizer = tokenizer;
			this.#text = text;
		}

		override _call(inputIds: number[][]): boolean[] {
			return inputIds.map((ids, index) => {
				const generatedStart = this.#generatedStarts[index] ?? Math.max(0, ids.length - 1);
				this.#generatedStarts[index] = generatedStart;
				const tail = ids.slice(Math.max(generatedStart, ids.length - STOP_DECODE_WINDOW_TOKENS));
				const decoded = this.#tokenizer.decode(tail, {
					skip_special_tokens: false,
					clean_up_tokenization_spaces: false,
				});
				return decoded.includes(this.#text);
			});
		}
	}
	return new StopOnTextCriteria();
}

/** Softmax probability of the largest of `length` logits starting at `offset`, with that logit's index. */
function argmaxProbability(
	logits: ArrayLike<number>,
	offset: number,
	length: number,
): { index: number; probability: number } {
	let index = 0;
	for (let k = 1; k < length; k += 1) {
		if (logits[offset + k]! > logits[offset + index]!) index = k;
	}
	const top = logits[offset + index]!;
	let sum = 0;
	for (let k = 0; k < length; k += 1) sum += Math.exp(logits[offset + k]! - top);
	return { index, probability: 1 / sum };
}

/**
 * Label every text with `classifier`: tokenize each text whole, run overlapping 512-position windows
 * in batches, stitch the per-token predictions back together, and merge them into spans.
 */
async function classifyTexts(
	transformers: TransformersRuntime,
	classifier: TokenClassificationPipeline,
	texts: readonly string[],
): Promise<TinyEntity[][]> {
	const { tokenizer, model } = classifier;
	const { id2label } = model.config as PretrainedConfig & { id2label?: Record<number, string> };
	if (!id2label) throw new Error("token-classification model config has no id2label");
	const labelCount = Object.keys(id2label).length;
	// An empty text encodes to just the template's special tokens.
	const [clsId, sepId] = tokenizer.encode("") as [number, number];
	const docs = texts.map(text => {
		const tokens = tokenizer.tokenize(text);
		return {
			text,
			tokens,
			ids: tokenizer.convert_tokens_to_ids(tokens),
			predictions: [] as TokenPrediction[],
		};
	});
	const jobs = docs.flatMap(doc => planTokenWindows(doc.tokens.length).map(window => ({ doc, window })));
	// Similar lengths share a batch so short texts are not padded out to a full window.
	jobs.sort((a, b) => a.window.end - a.window.start - (b.window.end - b.window.start));
	for (let first = 0; first < jobs.length; first += CLASSIFY_BATCH_SIZE) {
		const batch = jobs.slice(first, first + CLASSIFY_BATCH_SIZE);
		const width = batch.reduce((widest, { window }) => Math.max(widest, window.end - window.start), 0) + 2;
		const inputIds = new BigInt64Array(batch.length * width).fill(BigInt(tokenizer.pad_token_id));
		const attention = new BigInt64Array(batch.length * width);
		for (const [row, { doc, window }] of batch.entries()) {
			const base = row * width;
			const length = window.end - window.start;
			inputIds[base] = BigInt(clsId);
			for (let t = 0; t < length; t += 1) inputIds[base + 1 + t] = BigInt(doc.ids[window.start + t]!);
			inputIds[base + 1 + length] = BigInt(sepId);
			attention.fill(1n, base, base + length + 2);
		}
		const dims = [batch.length, width];
		const { logits } = await model({
			input_ids: new transformers.Tensor("int64", inputIds, dims),
			attention_mask: new transformers.Tensor("int64", attention, dims),
			token_type_ids: new transformers.Tensor("int64", new BigInt64Array(batch.length * width), dims),
		});
		const scores = logits.data as Float32Array;
		for (const [row, { doc, window }] of batch.entries()) {
			for (let t = window.keepStart; t < window.keepEnd; t += 1) {
				const at = (row * width + 1 + t - window.start) * labelCount;
				const { index, probability } = argmaxProbability(scores, at, labelCount);
				doc.predictions[t] = { label: id2label[index] ?? "O", score: probability };
			}
		}
	}
	return docs.map(doc => mergeTokenSpans(doc.text, doc.tokens, alignTokens(doc.text, doc.tokens), doc.predictions));
}

/** The worker's single ONNX model: transformers.js pipeline with the device fallback chain. */
class OnnxModel {
	#spec: TinyTitleLocalModelSpec;
	#modelKey: TinyLocalModelKey;
	#devicePreference: TinyModelDevicePreference;
	#dtypeOverride: TinyModelDtype | undefined;
	#runtime = new MemoizedRuntime<TransformersRuntime>();
	#loaded: Promise<LoadedPipeline> | null = null;

	constructor(
		modelKey: TinyLocalModelKey,
		spec: TinyTitleLocalModelSpec,
		devicePreference: TinyModelDevicePreference,
		dtypeOverride: TinyModelDtype | undefined,
	) {
		this.#modelKey = modelKey;
		this.#spec = spec;
		this.#devicePreference = devicePreference;
		this.#dtypeOverride = dtypeOverride;
	}

	/** `PI_TINY_DTYPE` retargets the LLM models' precision; a token classifier always loads its vetted dtype. */
	#dtype(): TinyModelDtype {
		return this.#spec.task === "token-classification" ? this.#spec.dtype : (this.#dtypeOverride ?? this.#spec.dtype);
	}

	#loadRuntime(reply: ReplyTransport, requestId: string): Promise<TransformersRuntime> {
		return loadTransformersRuntime(this.#runtime, reply, requestId, this.#modelKey, getTinyTitleRuntimeDir);
	}

	async #loadPipelineWithDeviceFallback(
		transformers: TransformersRuntime,
		reply: ReplyTransport,
		requestId: string,
	): Promise<{ loaded: LoadedPipeline; device: TinyOnnxDevice }> {
		const devices = tinyModelDeviceLoadOrder(this.#devicePreference);
		if (devices[0] !== this.#devicePreference.device) {
			logger.warn("tiny-model: requested device is not an ONNX provider usable in the worker; using CPU", {
				modelKey: this.#modelKey,
				requestedDevice: this.#devicePreference.device,
				device: devices[0],
			});
		}
		let cudaDiagnostics: string | null = null;
		for (let i = 0; i < devices.length; i += 1) {
			const device = devices[i]!;
			try {
				const options: PipelineOptions = {
					device,
					dtype: this.#dtype(),
					progress_callback: info => sendProgress(reply, requestId, this.#modelKey, info),
				};
				const loaded: LoadedPipeline =
					this.#spec.task === "token-classification"
						? {
								task: "token-classification",
								pipeline: await transformers.pipeline("token-classification", this.#spec.repo, options),
							}
						: {
								task: "text-generation",
								pipeline: await transformers.pipeline("text-generation", this.#spec.repo, options),
							};
				return { loaded, device };
			} catch (error) {
				const deviceDiagnostics = await formatOnnxRuntimeCudaDiagnostics(transformers, device, error);
				if (deviceDiagnostics) cudaDiagnostics = deviceDiagnostics;
				if (i === devices.length - 1) {
					if (cudaDiagnostics) throw new Error(`${errorText(error)}\n${cudaDiagnostics}`);
					throw error;
				}
				const meta: Record<string, unknown> = {
					modelKey: this.#modelKey,
					device,
					fallbackDevice: devices[i + 1],
					error: errorMessage(error),
				};
				if (deviceDiagnostics) meta.cudaDiagnostics = deviceDiagnostics;
				logger.warn("tiny-model: accelerated device failed; falling back", meta);
			}
		}
		throw new Error("No tiny model devices configured");
	}

	/** Resident pipeline, loading (with progress for `requestId`) on first use. */
	load(reply: ReplyTransport, requestId: string): Promise<LoadedPipeline> {
		if (this.#loaded) return this.#loaded;
		if (this.#spec.onnxUnsupportedReason) {
			return Promise.reject(new Error(`${this.#modelKey} is unavailable: ${this.#spec.onnxUnsupportedReason}`));
		}
		const startedAt = performance.now();
		const loading = this.#loadRuntime(reply, requestId)
			.then(transformers => this.#loadPipelineWithDeviceFallback(transformers, reply, requestId))
			.then(
				({ loaded, device }) => {
					logger.debug("tiny-model: local model loaded", {
						modelKey: this.#modelKey,
						repo: this.#spec.repo,
						device,
						requestedDevice: this.#devicePreference.device,
						dtype: this.#dtype(),
						elapsedMs: Math.round(performance.now() - startedAt),
					});
					return loaded;
				},
				error => {
					this.#loaded = null;
					throw error;
				},
			);
		this.#loaded = loading;
		return loading;
	}

	/** Send the `ready` marker the client's download UI waits for. */
	sendReady(reply: ReplyTransport, requestId: string): void {
		reply.send({
			type: "progress",
			id: requestId,
			event: {
				modelKey: this.#modelKey,
				status: "ready",
				task: this.#spec.task ?? "text-generation",
				model: this.#spec.repo,
			},
		});
	}

	async chat(request: Extract<TinyWorkerRequest, { type: "chat" }>, reply: ReplyTransport): Promise<string> {
		const loaded = await this.load(reply, request.id);
		if (loaded.task !== "text-generation") {
			throw new Error(`${this.#modelKey} is a ${loaded.task} model and cannot chat`);
		}
		const generator = loaded.pipeline;
		const rendered = renderTextChatTemplate(generator.tokenizer, request.messages, {
			addGenerationPrompt: true,
			enableThinking: false,
		});
		const promptText = request.prefill ? `${rendered}${request.prefill}` : rendered;
		const stoppingCriteria = request.stop
			? createStopOnTextCriteria(await this.#loadRuntime(reply, request.id), generator.tokenizer, request.stop)
			: undefined;
		const output = (await generator(promptText, {
			max_new_tokens: request.maxNewTokens,
			do_sample: false,
			return_full_text: false,
			...(stoppingCriteria ? { stopping_criteria: stoppingCriteria } : {}),
		})) as TextGenerationStringOutput;
		return output[0]?.generated_text ?? "";
	}

	async classify(
		request: Extract<TinyWorkerRequest, { type: "classify" }>,
		reply: ReplyTransport,
	): Promise<TinyEntity[][]> {
		const loaded = await this.load(reply, request.id);
		if (loaded.task !== "token-classification") {
			throw new Error(`${this.#modelKey} is a ${loaded.task} model and cannot classify`);
		}
		return classifyTexts(await this.#loadRuntime(reply, request.id), loaded.pipeline, request.texts);
	}
}

/** Run the ONNX worker for the model/endpoint selected by the CLI worker host environment. */
export async function startTinyWorkerFromEnvironment(): Promise<void> {
	const endpoint = process.env[TINY_WORKER_SOCKET_ENV];
	const modelKey = process.env[TINY_WORKER_MODEL_ENV];
	const tag = process.env[TINY_WORKER_TAG_ENV];
	if (!endpoint || !modelKey || !tag) throw new Error("tiny worker environment is incomplete");
	if (!isTinyLocalModelKey(modelKey)) throw new Error(`Unknown tiny local model: ${modelKey}`);
	const spec = getTinyLocalModelSpec(modelKey);
	if (!spec) throw new Error(`Unknown tiny local model: ${modelKey}`);
	setProcessName(`omp tiny ${modelKey}`);
	const model = new OnnxModel(modelKey, spec, resolveTinyModelDevicePreference(), resolveTinyModelDtypeOverride());
	const server = new TinyWorkerServer({
		tag,
		idleMs: Number(process.env[TINY_WORKER_IDLE_MS_ENV]) || TINY_WORKER_IDLE_MS,
		async handle(request, reply) {
			if (request.type === "load") {
				await model.load(reply, request.id);
				model.sendReady(reply, request.id);
				reply.send({ type: "loaded", id: request.id });
				return;
			}
			if (request.type === "classify") {
				reply.send({ type: "entities", id: request.id, entities: await model.classify(request, reply) });
				return;
			}
			const text = await model.chat(request, reply);
			reply.send({ type: "text", id: request.id, text });
		},
	});
	await server.serve(endpoint);
}
