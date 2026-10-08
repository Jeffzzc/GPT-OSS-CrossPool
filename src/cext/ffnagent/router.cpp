#include <xpool/ffnagent/router.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>

#include <xpool/arena.hpp>

namespace xpool::ffnagent {

namespace {

auto matmul_descriptor() {
  auto raw = cublasLtMatmulDesc_t{};
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&raw, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  return std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>, decltype(&cublasLtMatmulDescDestroy)>(
      raw, &cublasLtMatmulDescDestroy);
}

auto matrix_layout(std::int64_t rows, std::int64_t columns, std::int64_t leading_dimension) {
  auto raw = cublasLtMatrixLayout_t{};
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&raw, CUDA_R_16BF, rows, columns, leading_dimension));
  return std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)>(
      raw, &cublasLtMatrixLayoutDestroy);
}

auto matmul_preference() {
  auto raw = cublasLtMatmulPreference_t{};
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw));
  return std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>,
                         decltype(&cublasLtMatmulPreferenceDestroy)>(raw, &cublasLtMatmulPreferenceDestroy);
}

std::uint32_t pointer_alignment(const void *pointer) {
  auto alignment = std::uint32_t{256};
  while (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
    alignment /= 2;
  }
  return alignment;
}

bool overlaps(const at::Tensor &left, const at::Tensor &right) {
  const auto a = reinterpret_cast<std::uintptr_t>(left.data_ptr());
  const auto b = reinterpret_cast<std::uintptr_t>(right.data_ptr());
  return a < b + right.nbytes() && b < a + left.nbytes();
}

} // namespace

void biased_router_gemm(const at::Tensor &hidden_states, const at::Tensor &weight, const at::Tensor &bias,
                        const at::Tensor &logits, const at::Tensor &workspace) {
  const auto tensors = std::array{hidden_states, weight, bias, logits, workspace};
  for (const auto &tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == hidden_states.device() && tensor.is_contiguous(),
                "xpool biased Router GEMM requires contiguous tensors on one CUDA device");
  }
  TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
                  bias.scalar_type() == at::kBFloat16 && logits.scalar_type() == at::kBFloat16,
              "xpool biased Router GEMM requires BF16 input, weight, bias and logits");
  TORCH_CHECK(hidden_states.dim() == 2 && weight.dim() == 2 && bias.dim() == 1 && logits.dim() == 2,
              "xpool biased Router GEMM has invalid tensor ranks");
  const auto rows = hidden_states.size(0);
  const auto hidden = hidden_states.size(1);
  const auto experts = weight.size(0);
  for (auto extent : {rows, hidden, experts}) {
    TORCH_CHECK(extent > 0 && extent <= std::numeric_limits<int>::max(),
                "xpool biased Router GEMM has an invalid dimension");
  }
  TORCH_CHECK(weight.size(1) == hidden && bias.size(0) == experts && logits.size(0) == rows &&
                  logits.size(1) == experts,
              "xpool biased Router GEMM tensor geometry disagrees");
  TORCH_CHECK(workspace.scalar_type() == at::kByte && workspace.dim() == 1 &&
                  workspace.nbytes() >= kBiasedRouterGemmWorkspaceBytes &&
                  pointer_alignment(workspace.data_ptr()) == kBiasedRouterGemmAlignmentBytes &&
                  pointer_alignment(logits.data_ptr()) == kBiasedRouterGemmAlignmentBytes,
              "xpool biased Router GEMM requires aligned logits and at least 1 MiB of aligned uint8 scratch");
  for (const auto &input : {hidden_states, weight, bias}) {
    TORCH_CHECK(!overlaps(input, logits) && !overlaps(input, workspace),
                "xpool biased Router GEMM destinations overlap input storage");
  }
  TORCH_CHECK(!overlaps(logits, workspace), "xpool biased Router GEMM logits overlap scratch");
  TORCH_CHECK(pointer_alignment(hidden_states.data_ptr()) >= xpool::arena::kPayloadAlignment,
              "xpool biased Router GEMM input violates the Fabric payload alignment");

  const auto guard = c10::cuda::CUDAGuard{hidden_states.device()};
  const auto *properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major >= 8, "xpool biased BF16 Router GEMM requires SM80 or newer");

  // Match pinned Torch 2.13 gemm_and_bias: column-major views of the row-major
  // tensors, FP32 accumulation/scalars, bias epilogue and the first heuristic.
  // Only the scratch owner differs. No Torch workspace getter/setter is used.
  const auto descriptor = matmul_descriptor();
  const auto preference = matmul_preference();
  const auto transa = CUBLAS_OP_T;
  const auto transb = CUBLAS_OP_N;
  const auto epilogue = CUBLASLT_EPILOGUE_BIAS;
  const auto *bias_pointer = bias.data_ptr();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                                    sizeof(transa)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                                    sizeof(transb)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                    sizeof(epilogue)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias_pointer, sizeof(bias_pointer)));
  const auto reduction = at::globalContext().allowBF16ReductionCuBLAS();
  if (reduction != at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
    const auto mask = reduction == at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
                          ? static_cast<std::uint32_t>(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) |
                                static_cast<std::uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE)
                          : static_cast<std::uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE);
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference.get(), CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));
  }
  if (const auto carveout = at::globalContext()._SMCarveout_EXPERIMENTAL(); carveout.has_value()) {
    TORCH_CHECK(*carveout >= 0 && *carveout < properties->multiProcessorCount, "xpool invalid cuBLAS SM carveout");
    const auto count = static_cast<std::int32_t>(properties->multiProcessorCount - *carveout);
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                      &count, sizeof(count)));
  }
  const auto a = matrix_layout(hidden, experts, hidden);
  const auto b = matrix_layout(hidden, rows, hidden);
  const auto c = matrix_layout(experts, rows, experts);
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference.get(),
                                                          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &kBiasedRouterGemmWorkspaceBytes, sizeof(std::size_t)));
  const auto alignments = std::array{
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, pointer_alignment(weight.data_ptr())},
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, pointer_alignment(hidden_states.data_ptr())},
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, pointer_alignment(logits.data_ptr())},
      // Torch uses bias alignment for D even though C and D share the logits.
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, pointer_alignment(bias_pointer)}};
  for (const auto &[attribute, alignment] : alignments) {
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulPreferenceSetAttribute(preference.get(), attribute, &alignment, sizeof(alignment)));
  }
  auto heuristic = cublasLtMatmulHeuristicResult_t{};
  auto count = int{0};
  const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, descriptor.get(), a.get(), b.get(), c.get(), c.get(),
                                                   preference.get(), 1, &heuristic, &count));
  TORCH_CHECK(count == 1 && heuristic.state == CUBLAS_STATUS_SUCCESS &&
                  heuristic.workspaceSize <= kBiasedRouterGemmWorkspaceBytes,
              "xpool biased Router cuBLASLt heuristic failed for rows=", rows, ", experts=", experts,
              ", hidden=", hidden);
  // Capture inputs are Torch allocations, but Lane relocation targets the
  // Fabric's 16-byte-aligned payload. The selected algorithm must support that
  // weaker alignment even when the reference heuristic was offered 256 bytes.
  auto input_alignment = std::uint32_t{0};
  auto attribute_bytes = std::size_t{0};
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(&heuristic.algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
                                                       &input_alignment, sizeof(input_alignment), &attribute_bytes));
  TORCH_CHECK(attribute_bytes == sizeof(input_alignment) && input_alignment <= xpool::arena::kPayloadAlignment,
              "xpool biased Router cuBLASLt algorithm requires input alignment ", input_alignment,
              " incompatible with Fabric payload relocation");
  const auto alpha = 1.0F;
  const auto beta = 0.0F;
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(handle, descriptor.get(), &alpha, weight.data_ptr(), a.get(),
                                    hidden_states.data_ptr(), b.get(), &beta, logits.data_ptr(), c.get(),
                                    logits.data_ptr(), c.get(), &heuristic.algo, workspace.data_ptr(),
                                    kBiasedRouterGemmWorkspaceBytes, c10::cuda::getCurrentCUDAStream()));
}

} // namespace xpool::ffnagent
