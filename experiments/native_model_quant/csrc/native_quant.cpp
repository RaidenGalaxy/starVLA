#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <tuple>

namespace {

void check_cpu_contiguous(const torch::Tensor& tensor, const char* label) {
  TORCH_CHECK(tensor.device().is_cpu(), label, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
}

bool supports_avx512_vnni_impl() {
#if defined(__x86_64__) || defined(_M_X64)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> quantize_activation_groupwise(
    const torch::Tensor& input,
    int64_t group_size) {
  check_cpu_contiguous(input, "input");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be FP32");
  TORCH_CHECK(input.dim() == 2, "input must be two-dimensional");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  const int64_t rows = input.size(0);
  const int64_t width = input.size(1);
  const int64_t groups = (width + group_size - 1) / group_size;
  auto codes = torch::empty({rows, width}, input.options().dtype(torch::kInt8));
  auto scales = torch::empty({rows, groups}, input.options().dtype(torch::kFloat32));
  const float* source = input.data_ptr<float>();
  int8_t* target = codes.data_ptr<int8_t>();
  float* scale_values = scales.data_ptr<float>();
  at::parallel_for(0, rows * groups, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t row = item / groups;
      const int64_t group = item % groups;
      const int64_t start = group * group_size;
      const int64_t stop = std::min(start + group_size, width);
      float maximum = 0.0f;
      for (int64_t column = start; column < stop; ++column) {
        const float value = source[row * width + column];
        TORCH_CHECK(std::isfinite(value), "activation contains NaN or Inf");
        maximum = std::max(maximum, std::abs(value));
      }
      const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
      scale_values[row * groups + group] = scale;
      for (int64_t column = start; column < stop; ++column) {
        const float normalized = source[row * width + column] / scale;
        const long rounded = std::lrint(normalized);
        target[row * width + column] = static_cast<int8_t>(std::max(-127L, std::min(127L, rounded)));
      }
    }
  });
  return {codes, scales};
}

inline int32_t dot_scalar(const uint8_t* shifted_weight, const int8_t* activation, int64_t length) {
  int32_t result = 0;
  for (int64_t index = 0; index < length; ++index) {
    result += (static_cast<int32_t>(shifted_weight[index]) - 8) * static_cast<int32_t>(activation[index]);
  }
  return result;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
int32_t dot_vnni(const uint8_t* shifted_weight, const int8_t* activation, int64_t length) {
  __m512i dot_accumulator = _mm512_setzero_si512();
  __m512i sum_accumulator = _mm512_setzero_si512();
  const __m512i ones = _mm512_set1_epi8(1);
  int64_t index = 0;
  for (; index + 64 <= length; index += 64) {
    const __m512i weight = _mm512_loadu_si512(reinterpret_cast<const void*>(shifted_weight + index));
    const __m512i activation_values =
        _mm512_loadu_si512(reinterpret_cast<const void*>(activation + index));
    dot_accumulator = _mm512_dpbusd_epi32(dot_accumulator, weight, activation_values);
    sum_accumulator = _mm512_dpbusd_epi32(sum_accumulator, ones, activation_values);
  }
  int32_t result = _mm512_reduce_add_epi32(dot_accumulator) - 8 * _mm512_reduce_add_epi32(sum_accumulator);
  for (; index < length; ++index) {
    result += (static_cast<int32_t>(shifted_weight[index]) - 8) * static_cast<int32_t>(activation[index]);
  }
  return result;
}
#endif

template <bool UseVnni>
torch::Tensor packed_w4a8_linear(
    const torch::Tensor& qactivation,
    const torch::Tensor& activation_scales,
    const torch::Tensor& qweight,
    const torch::Tensor& weight_scales,
    const c10::optional<torch::Tensor>& bias,
    int64_t in_features) {
  check_cpu_contiguous(qactivation, "qactivation");
  check_cpu_contiguous(activation_scales, "activation_scales");
  check_cpu_contiguous(qweight, "qweight");
  check_cpu_contiguous(weight_scales, "weight_scales");
  TORCH_CHECK(qactivation.scalar_type() == torch::kInt8, "qactivation must be INT8");
  TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32, "activation_scales must be FP32");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, "qweight must be packed UINT8");
  TORCH_CHECK(
      weight_scales.scalar_type() == torch::kFloat16 || weight_scales.scalar_type() == torch::kFloat32,
      "weight_scales must be FP16 or FP32");
  TORCH_CHECK(qactivation.dim() == 2 && qweight.dim() == 2, "qactivation and qweight must be 2D");
  TORCH_CHECK(in_features > 0 && qactivation.size(1) == in_features, "invalid in_features");
  TORCH_CHECK(qweight.size(1) == (in_features + 1) / 2, "packed qweight width mismatch");
  constexpr int64_t group_size = 128;
  const int64_t rows = qactivation.size(0);
  const int64_t outputs = qweight.size(0);
  const int64_t groups = (in_features + group_size - 1) / group_size;
  TORCH_CHECK(
      activation_scales.sizes() == torch::IntArrayRef({rows, groups}),
      "activation scale shape mismatch");
  TORCH_CHECK(weight_scales.sizes() == torch::IntArrayRef({outputs, groups}), "weight scale shape mismatch");
  if constexpr (UseVnni) {
    TORCH_CHECK(supports_avx512_vnni_impl(), "AVX-512 VNNI is unavailable; refusing floating fallback");
  }
  const int8_t* activation = qactivation.data_ptr<int8_t>();
  const float* activation_scale = activation_scales.data_ptr<float>();
  const uint8_t* packed_weight = qweight.data_ptr<uint8_t>();
  const float* weight_scale_float =
      weight_scales.scalar_type() == torch::kFloat32 ? weight_scales.data_ptr<float>() : nullptr;
  const c10::Half* weight_scale_half =
      weight_scales.scalar_type() == torch::kFloat16 ? weight_scales.data_ptr<c10::Half>() : nullptr;
  const float* bias_values = nullptr;
  if (bias.has_value()) {
    check_cpu_contiguous(*bias, "bias");
    TORCH_CHECK(bias->scalar_type() == torch::kFloat32 && bias->numel() == outputs, "bias must be FP32 [N]");
    bias_values = bias->data_ptr<float>();
  }
  auto output = torch::empty({rows, outputs}, activation_scales.options());
  float* output_values = output.data_ptr<float>();
  at::parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
    alignas(64) uint8_t shifted_weight[128];
    for (int64_t out = begin; out < end; ++out) {
      const float bias_value = bias_values == nullptr ? 0.0f : bias_values[out];
      for (int64_t row = 0; row < rows; ++row) {
        output_values[row * outputs + out] = bias_value;
      }
      const uint8_t* packed_row = packed_weight + out * qweight.size(1);
      for (int64_t group = 0; group < groups; ++group) {
        const int64_t start = group * group_size;
        const int64_t stop = std::min(start + group_size, in_features);
        const int64_t length = stop - start;
        for (int64_t index = 0; index < length; ++index) {
          const int64_t column = start + index;
          const uint8_t byte = packed_row[column / 2];
          const uint8_t nibble = (byte >> ((column & 1) * 4)) & 0x0f;
          shifted_weight[index] = nibble ^ 0x08;
        }
        const float weight_scale = weight_scale_float != nullptr
            ? weight_scale_float[out * groups + group]
            : static_cast<float>(weight_scale_half[out * groups + group]);
        for (int64_t row = 0; row < rows; ++row) {
          const int8_t* activation_group = activation + row * in_features + start;
          int32_t dot;
          if constexpr (UseVnni) {
#if defined(__x86_64__) || defined(_M_X64)
            dot = dot_vnni(shifted_weight, activation_group, length);
#else
            TORCH_CHECK(false, "AVX-512 VNNI is only supported on x86-64");
#endif
          } else {
            dot = dot_scalar(shifted_weight, activation_group, length);
          }
          output_values[row * outputs + out] +=
              static_cast<float>(dot) * activation_scale[row * groups + group] * weight_scale;
        }
      }
    }
  });
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("supports_avx512_vnni", &supports_avx512_vnni_impl);
  module.def("quantize_activation_groupwise", &quantize_activation_groupwise);
  module.def("packed_w4a8_linear_scalar", &packed_w4a8_linear<false>);
  module.def("packed_w4a8_linear_vnni", &packed_w4a8_linear<true>);
}
