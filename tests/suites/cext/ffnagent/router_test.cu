#include <array>
#include <cstddef>
#include <cstdint>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/ffnagent/parameterization.hpp>
#include <xpool/ffnagent/router.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/graph.hpp>

namespace {

XPOOL_KERNEL_FN void rebind_router(const xpool::ffnagent::BindingSite *sites, std::size_t count,
                                   const xpool::ffnagent::LayerBindingValues *values, cudaError_t *status) {
  *status = cudaSuccess;
  for (auto index = std::size_t{0}; index < count; ++index) {
    const auto result = cudaGraphKernelNodeSetParam(
        sites[index].node, sites[index].parameter_offset_bytes,
        reinterpret_cast<const unsigned char *>(values) + sites[index].value_offset_bytes, sizeof(std::uintptr_t));
    if (result != cudaSuccess) {
      *status = result;
      return;
    }
  }
}

struct RouterLane {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  cudaStream_t stream{};
  at::Tensor input;
  at::Tensor storage;
  at::Tensor logits;
  at::Tensor sites;
  at::Tensor values;
  at::Tensor status;
};

// This test observes actual cuBLASLt kernel argument schemas, including the
// bias pointer in a separate Split-K reduction kernel. Integer-valued products
// make the independent expected BF16 result exact on every supported device.
TEST(RouterGemmTest, RelocatesScratchAndDeviceRebindsConcurrentLanes) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  const auto bf16 = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  const auto bytes = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
  for (const auto capacity : {std::int64_t{32}, std::int64_t{64}}) {
    const auto logits_bytes = capacity * 32 * 2;
    const auto scratch_offset = (logits_bytes + 255) / 256 * 256;
    const auto storage_bytes = scratch_offset + xpool::ffnagent::kBiasedRouterGemmWorkspaceBytes;
    const auto input = at::ones({capacity, 2880}, bf16);
    const auto storage = at::empty({static_cast<std::int64_t>(storage_bytes)}, bytes);
    const auto logits = storage.narrow(0, 0, logits_bytes).view(at::kBFloat16).view({capacity, 32});
    const auto scratch = storage.narrow(0, scratch_offset, xpool::ffnagent::kBiasedRouterGemmWorkspaceBytes);
    const auto weights = std::array{at::full({32, 2880}, 1.0 / 128, bf16),
                                    at::full({32, 2880}, 2.0 / 128, bf16)};
    const auto biases = std::array{at::full({32}, 0.5, bf16), at::full({32}, -0.5, bf16)};
    auto capture_stream = cudaStream_t{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), cudaSuccess);
    auto primary = cudaGraph_t{};
    auto control = cudaGraph_t{};
    {
      const auto guard = c10::cuda::CUDAStreamGuard{c10::cuda::getStreamFromExternal(capture_stream, 0)};
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (auto layer : {0, 1}) {
        xpool::ffnagent::biased_router_gemm(input, weights[layer], biases[layer], logits, scratch);
      }
      ASSERT_EQ(cudaStreamSynchronize(capture_stream), cudaSuccess);
      for (auto layer : {0, 1}) {
        ASSERT_EQ(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        xpool::ffnagent::biased_router_gemm(input, weights[layer], biases[layer], logits, scratch);
        ASSERT_EQ(cudaStreamEndCapture(capture_stream, layer == 0 ? &primary : &control), cudaSuccess);
      }
    }
    auto lanes = std::array<RouterLane, 2>{};
    for (auto &lane : lanes) {
      lane.input = at::empty({input.numel() + 8}, bf16).narrow(0, 8, input.numel()).view({capacity, 2880});
      ASSERT_EQ(reinterpret_cast<std::uintptr_t>(lane.input.data_ptr()) % 256, 16);
      lane.storage = at::empty_like(storage);
      lane.logits = lane.storage.narrow(0, 0, logits_bytes).view(at::kBFloat16).view({capacity, 32});
      ASSERT_NE(lane.storage.data_ptr(), storage.data_ptr());
      ASSERT_EQ(cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking), cudaSuccess);
      ASSERT_EQ(cudaGraphCreate(&lane.graph, 0), cudaSuccess);
      const auto child = xpool::utils::graph::embed_child_graph(lane.graph, primary);
      const auto resources = std::array{
          xpool::ffnagent::ResourceReplacement{
              .primary_address = reinterpret_cast<std::uintptr_t>(weights[0].data_ptr()),
              .control_address = reinterpret_cast<std::uintptr_t>(weights[1].data_ptr()),
              .target_address = reinterpret_cast<std::uintptr_t>(weights[0].data_ptr()),
              .value_offset_bytes = offsetof(xpool::ffnagent::LayerBindingValues, router_weight_address)},
          xpool::ffnagent::ResourceReplacement{
              .primary_address = reinterpret_cast<std::uintptr_t>(biases[0].data_ptr()),
              .control_address = reinterpret_cast<std::uintptr_t>(biases[1].data_ptr()),
              .target_address = reinterpret_cast<std::uintptr_t>(biases[0].data_ptr()),
              .value_offset_bytes = offsetof(xpool::ffnagent::LayerBindingValues, router_projection_bias_address)}};
      const auto replacements = std::array{
          xpool::ffnagent::LaneAddressReplacement{
              .capture_address = reinterpret_cast<std::uintptr_t>(input.data_ptr()),
              .bytes = input.nbytes(),
              .target_address = reinterpret_cast<std::uintptr_t>(lane.input.data_ptr())},
          xpool::ffnagent::LaneAddressReplacement{
              .capture_address = reinterpret_cast<std::uintptr_t>(storage.data_ptr()),
              .bytes = storage.nbytes(),
              .target_address = reinterpret_cast<std::uintptr_t>(lane.storage.data_ptr())}};
      const auto parameterization =
          xpool::ffnagent::parameterize_graph(child, control, resources, replacements, 0, 0, 0);
      // Inspect every relocated pointer, so scratch aliasing cannot pass merely
      // because two very short GEMMs happened to finish without overlapping.
      for (const auto node : xpool::utils::graph::nodes(child)) {
        ASSERT_EQ(xpool::utils::graph::node_type(node), cudaGraphNodeTypeKernel);
        const auto parameters = xpool::utils::graph::KernelNodeParameters::read(node);
        for (const auto &argument : parameters.arguments()) {
          for (auto offset = std::size_t{0}; offset + sizeof(std::uintptr_t) <= argument.bytes.size(); ++offset) {
            const auto address = argument.read_address(offset);
            const auto capture_begin = reinterpret_cast<std::uintptr_t>(storage.data_ptr());
            EXPECT_FALSE(address >= capture_begin && address - capture_begin < storage.nbytes());
          }
        }
      }
      auto count = parameterization.binding_sites.size();
      ASSERT_GE(count, std::size_t{2});
      lane.sites = at::empty({static_cast<std::int64_t>(count * sizeof(xpool::ffnagent::BindingSite))}, bytes);
      lane.values = at::empty({sizeof(xpool::ffnagent::LayerBindingValues)}, bytes);
      lane.status = at::empty({sizeof(cudaError_t)}, bytes);
      ASSERT_EQ(cudaMemcpy(lane.sites.data_ptr(), parameterization.binding_sites.data(), lane.sites.nbytes(),
                           cudaMemcpyHostToDevice), cudaSuccess);
      auto sites = static_cast<xpool::ffnagent::BindingSite *>(lane.sites.data_ptr());
      auto values = static_cast<xpool::ffnagent::LayerBindingValues *>(lane.values.data_ptr());
      auto status = static_cast<cudaError_t *>(lane.status.data_ptr());
      auto arguments = std::array<void *, 4>{&sites, &count, &values, &status};
      const auto child_node = xpool::utils::graph::nodes(lane.graph).front();
      const auto update = xpool::utils::graph::add_kernel_node(
          lane.graph, reinterpret_cast<const void *>(rebind_router), dim3{1}, dim3{1}, 0, arguments.data());
      xpool::utils::graph::add_dependency(lane.graph, update, child_node);
      ASSERT_EQ(cudaGraphInstantiate(&lane.executable, lane.graph, nullptr, nullptr, 0), cudaSuccess);
    }
    ASSERT_NE(lanes[0].storage.data_ptr(), lanes[1].storage.data_ptr());
    // Captured storage must be irrelevant after installation, including the
    // scratch used by both GEMM and Split-K reduction nodes.
    input.fill_(99);
    storage.fill_(255);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto bindings = std::array<xpool::ffnagent::LayerBindingValues, 2>{};
    for (auto iteration = 0; iteration < 3; ++iteration) {
      for (auto lane_index = std::size_t{0}; lane_index < lanes.size(); ++lane_index) {
        auto &lane = lanes[lane_index];
        const auto layer = (iteration + lane_index) % 2;
        const auto value = iteration + lane_index + 1;
        const auto guard = c10::cuda::CUDAStreamGuard{c10::cuda::getStreamFromExternal(lane.stream, 0)};
        lane.input.fill_(static_cast<double>(value));
        bindings[lane_index].router_weight_address = reinterpret_cast<std::uintptr_t>(weights[layer].data_ptr());
        bindings[lane_index].router_projection_bias_address =
            reinterpret_cast<std::uintptr_t>(biases[layer].data_ptr());
        ASSERT_EQ(cudaMemcpyAsync(lane.values.data_ptr(), &bindings[lane_index], sizeof(bindings[lane_index]),
                                  cudaMemcpyHostToDevice, lane.stream), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(lane.executable, lane.stream), cudaSuccess);
      }
      // Enqueue both independent Lane executions before waiting for either.
      for (auto lane_index = std::size_t{0}; lane_index < lanes.size(); ++lane_index) {
        auto &lane = lanes[lane_index];
        ASSERT_EQ(cudaStreamSynchronize(lane.stream), cudaSuccess);
        auto status = cudaErrorUnknown;
        ASSERT_EQ(cudaMemcpy(&status, lane.status.data_ptr(), sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(status, cudaSuccess);
        const auto layer = (iteration + lane_index) % 2;
        const auto value = iteration + lane_index + 1;
        const auto expected = 22.5 * (layer + 1) * value + (layer == 0 ? 0.5 : -0.5);
        EXPECT_TRUE(at::equal(lane.logits, at::full({capacity, 32}, expected, bf16)));
      }
    }
    for (auto &lane : lanes) {
      EXPECT_EQ(cudaGraphExecDestroy(lane.executable), cudaSuccess);
      EXPECT_EQ(cudaGraphDestroy(lane.graph), cudaSuccess);
      EXPECT_EQ(cudaStreamDestroy(lane.stream), cudaSuccess);
    }
    EXPECT_EQ(cudaGraphDestroy(control), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(primary), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(capture_stream), cudaSuccess);
  }
}

} // namespace
