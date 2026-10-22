import { describe, expect, it } from "bun:test";
import { type HardwareProfile, recommendLocalModel } from "@oh-my-pi/pi-coding-agent/privacy/local-model-advice";

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

const apple = (gb: number): HardwareProfile => ({
	platform: "darwin",
	arch: "arm64",
	totalMemBytes: gb * GIB,
	appleSilicon: true,
});
const nvidia = (vramBytes: number, ramGb = 128): HardwareProfile => ({
	platform: "linux",
	arch: "x64",
	totalMemBytes: ramGb * GIB,
	appleSilicon: false,
	gpu: { name: "NVIDIA RTX 4090", vramBytes },
});
const ram = (bytes: number): HardwareProfile => ({
	platform: "linux",
	arch: "x64",
	totalMemBytes: bytes,
	appleSilicon: false,
});

describe("recommendLocalModel", () => {
	it.each([
		// Realistic reported values: nominal-size machines report slightly less than the nominal size.
		["Linux 32 GB host", ram(33572700160), "gemma4:26b-a4b"],
		["16 GB PC", ram(16.5 * GIB), "gemma4:e4b"],
		["8 GB PC", ram(7.5 * GIB), "gemma4:e2b"],
		["7 GiB RAM", ram(7 * GIB), "qwen3:1.7b"],
		["Apple 16 GiB exact", apple(16), "gemma4:e4b"],
		["NVIDIA 16376 MiB (16 GB card)", nvidia(16376 * MIB), "gemma4:e4b"],
		["NVIDIA 24564 MiB (24 GB card)", nvidia(24564 * MIB), "gemma4:e4b"],
		["NVIDIA 8188 MiB (8 GB card)", nvidia(8188 * MIB), "gemma4:e2b"],
		["NVIDIA 6 GB card on a 64 GB host", nvidia(6 * GIB, 64), "qwen3:1.7b"],
		// Tolerance edges: 0.9 x tier floor meets the tier, just under does not.
		["Apple 7.1 GiB", apple(7.1), "qwen3:1.7b"],
		["Apple 7.2 GiB", apple(7.2), "gemma4:e2b"],
		["Apple 14.3 GiB", apple(14.3), "gemma4:e2b"],
		["Apple 14.4 GiB", apple(14.4), "gemma4:e4b"],
		["Apple 28.7 GiB", apple(28.7), "gemma4:e4b"],
		["Apple 28.8 GiB", apple(28.8), "gemma4:26b-a4b"],
		["Apple 64 GiB", apple(64), "gemma4:26b-a4b"],
	])("%s", (_name, hw, tag) => {
		const advice = recommendLocalModel(hw);
		expect(advice.selector).toBe(`ollama/${tag}`);
		expect(advice.pullCommand).toBe(`ollama pull ${tag}`);
	});

	it("describes the memory budget at its nominal size", () => {
		expect(recommendLocalModel(apple(16)).summary).toBe("16 GB unified memory (Apple Silicon)");
		expect(recommendLocalModel(nvidia(24564 * MIB)).summary).toBe("24 GB VRAM (NVIDIA RTX 4090)");
		expect(recommendLocalModel(nvidia(16376 * MIB)).summary).toBe("16 GB VRAM (NVIDIA RTX 4090)");
		expect(recommendLocalModel(ram(33572700160)).summary).toBe("32 GB RAM (no GPU found)");
		expect(recommendLocalModel(ram(64 * GIB)).summary).toBe("64 GB RAM (no GPU found)");
	});
});
