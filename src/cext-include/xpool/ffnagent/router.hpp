#pragma once

/// \file xpool/ffnagent/router.hpp
/// \brief Biased BF16 Router GEMM with Lane-owned cuBLASLt scratch.

#include <cstddef>

#include <ATen/core/Tensor.h>

namespace xpool::ffnagent {

/// Pinned Torch 2.13 default cuBLASLt heuristic workspace budget, in bytes.
inline constexpr std::size_t kBiasedRouterGemmWorkspaceBytes = 1024 * 1024;

/// Required alignment of Router logits and cuBLASLt scratch, in bytes.
inline constexpr std::size_t kBiasedRouterGemmAlignmentBytes = 256;

/// Compute `hidden_states @ weight.T + bias`, rounding once into BF16 logits.
/// Inputs and destination are contiguous BF16 CUDA tensors on one SM80+ device;
/// shapes are [rows, hidden], [experts, hidden], [experts] and [rows, experts].
/// Workspace is disjoint contiguous uint8 storage, aligned to 256 bytes, with
/// at least kBiasedRouterGemmWorkspaceBytes bytes. The operator borrows Torch's
/// current Lt handle/stream and supplies caller scratch explicitly; it neither
/// changes the process workspace policy nor allocates device storage. Callers
/// retain all tensors until execution completes. Captured workspace interior
/// pointers must be relocated with the enclosing Lane compute-workspace span.
/// Input is at least 16-byte aligned; the selected algorithm must also support
/// the Fabric payload's 16-byte alignment after Lane relocation.
/// Heuristic/launch failures and invalid tensor geometry raise c10::Error.
void biased_router_gemm(const at::Tensor &hidden_states, const at::Tensor &weight, const at::Tensor &bias,
                        const at::Tensor &logits, const at::Tensor &workspace);

} // namespace xpool::ffnagent
