/**
 * Hardware-aware local model suggestion for the PII privacy gate.
 *
 * When personal data is about to go to a cloud model, the gate offers to switch
 * to a local one. If the user has not configured `privacy.localModel`, this
 * module tells them which ollama model fits this machine.
 */
import * as os from "node:os";
import { $ } from "bun";
import { tinyMlxSupported } from "../tiny/device";

export interface HardwareProfile {
	platform: NodeJS.Platform;
	arch: string;
	totalMemBytes: number;
	/** darwin + arm64: CPU and GPU share unified memory. */
	appleSilicon: boolean;
	/** NVIDIA GPU with the most VRAM, via nvidia-smi, when present. */
	gpu?: { name: string; vramBytes: number };
}

export interface LocalModelAdvice {
	/** Model selector in `provider/id` form, e.g. `ollama/gemma4:e4b`. */
	selector: string;
	pullCommand: string;
	/** Human description of the memory budget, e.g. `16 GB unified memory (Apple Silicon)`. */
	summary: string;
}

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

/**
 * Machines report less than their nominal size (a 32 GB host shows 31.3 GiB, a 16 GB
 * GPU 15.99 GiB, firmware/kernel/driver reserve the rest), so a budget meets a tier
 * at this fraction of its floor.
 */
// lean: one flat fraction; it is a calibration knob, tune it if real machines land in the wrong tier.
const NOMINAL_SIZE_TOLERANCE = 0.9;

/**
 * Ollama tags (verified on ollama.com 2026-10-22), largest tier first; the first
 * tier whose `minGb` the memory budget reaches (within NOMINAL_SIZE_TOLERANCE) wins. Sizes are the default q4
 * download: `gemma4:26b-a4b` is 16 GB, which leaves room for KV cache and the OS
 * at a 32 GB budget.
 */
const MODEL_TIERS: readonly { minGb: number; tag: string }[] = [
	{ minGb: 32, tag: "gemma4:26b-a4b" },
	{ minGb: 16, tag: "gemma4:e4b" }, // 6.6 GB
	{ minGb: 8, tag: "gemma4:e2b" }, // 4.6 GB
	{ minGb: 0, tag: "qwen3:1.7b" },
];

/** Parse `name, memory.total` CSV lines from nvidia-smi (MiB) and keep the largest GPU. */
function parseNvidiaSmi(output: string): HardwareProfile["gpu"] {
	let best: HardwareProfile["gpu"];
	for (const line of output.split("\n")) {
		const comma = line.lastIndexOf(",");
		if (comma < 0) continue;
		const mib = Number.parseInt(line.slice(comma + 1).trim(), 10);
		if (!Number.isFinite(mib) || mib <= 0) continue;
		if (!best || mib * MIB > best.vramBytes) {
			best = { name: line.slice(0, comma).trim(), vramBytes: mib * MIB };
		}
	}
	return best;
}

// lean: NVIDIA only; AMD/Intel GPUs fall back to system RAM, add rocm-smi/xpu-smi probing if needed.
async function probeNvidiaGpu(): Promise<HardwareProfile["gpu"]> {
	try {
		const result = await $`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`.quiet().nothrow();
		return result.exitCode === 0 ? parseNvidiaSmi(result.stdout.toString()) : undefined;
	} catch {
		// nvidia-smi absent: Bun Shell throws on a missing binary even with nothrow().
		return undefined;
	}
}

export async function probeHardware(): Promise<HardwareProfile> {
	const platform = process.platform;
	const arch = process.arch;
	const appleSilicon = tinyMlxSupported(platform, arch);
	const gpu = appleSilicon ? undefined : await probeNvidiaGpu();
	return { platform, arch, totalMemBytes: os.totalmem(), appleSilicon, gpu };
}

function budgetOf(hw: HardwareProfile): { bytes: number; describe: (gb: number) => string } {
	if (hw.appleSilicon) return { bytes: hw.totalMemBytes, describe: gb => `${gb} GB unified memory (Apple Silicon)` };
	if (hw.gpu) {
		const name = hw.gpu.name;
		return { bytes: hw.gpu.vramBytes, describe: gb => `${gb} GB VRAM (${name})` };
	}
	return { bytes: hw.totalMemBytes, describe: gb => `${gb} GB RAM (no GPU found)` };
}

export function recommendLocalModel(hw: HardwareProfile): LocalModelAdvice {
	const { bytes, describe } = budgetOf(hw);
	const gb = bytes / GIB;
	const tier = MODEL_TIERS.find(t => gb >= t.minGb * NOMINAL_SIZE_TOLERANCE) ?? MODEL_TIERS[MODEL_TIERS.length - 1];
	// Report the nominal size the tier decision was made on (31.3 GiB shows as 32 GB), not the raw reading.
	const summary = describe(Math.max(Math.round(gb), tier.minGb));
	return { selector: `ollama/${tier.tag}`, pullCommand: `ollama pull ${tier.tag}`, summary };
}
