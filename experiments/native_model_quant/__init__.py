"""Native CPU execution backends for packed Qwen2.5-VL checkpoints."""

BACKENDS = ("streaming_dequant_fp32", "streaming_a16_sim", "avx512_w4a8")
KERNEL_FORMAT_VERSION = "starvla_native_w4a8_v1"
