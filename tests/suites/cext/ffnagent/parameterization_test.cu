#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/ffnagent/parameterization.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/graph.hpp>

namespace {

constexpr auto resource_count = std::size_t{8};
constexpr auto offsets = std::array<std::size_t, resource_count>{
    offsetof(xpool::ffnagent::LayerBindingValues, gate_up_weight_address),
    offsetof(xpool::ffnagent::LayerBindingValues, down_weight_address),
    offsetof(xpool::ffnagent::LayerBindingValues, router_weight_address),
    offsetof(xpool::ffnagent::LayerBindingValues, router_projection_bias_address),
    offsetof(xpool::ffnagent::LayerBindingValues, gate_up_scales_address),
    offsetof(xpool::ffnagent::LayerBindingValues, down_scales_address),
    offsetof(xpool::ffnagent::LayerBindingValues, gate_up_bias_address),
    offsetof(xpool::ffnagent::LayerBindingValues, down_bias_address)};

XPOOL_KERNEL_FN void read_resources(int *output, const int *w13, const int *w2, const int *router,
                                    const int *router_bias, const int *s13, const int *s2,
                                    const int *b13, const int *b2) {
  *output = *w13 + *w2 + *router + *router_bias + *s13 + *s2 + *b13 + *b2;
}

XPOOL_KERNEL_FN void rebind_resources(const xpool::ffnagent::BindingSite *sites,
                                      const xpool::ffnagent::LayerBindingValues *values,
                                      cudaError_t *status) {
  *status = cudaSuccess;
  for (auto i = std::size_t{0}; i < resource_count; ++i) {
    const auto result = cudaGraphKernelNodeSetParam(
        sites[i].node, sites[i].parameter_offset_bytes,
        reinterpret_cast<const unsigned char *>(values) + sites[i].value_offset_bytes, sizeof(std::uintptr_t));
    if (result != cudaSuccess) {
      *status = result;
      return;
    }
  }
}

TEST(Mxfp4ParameterizationTest, RebindsEveryBiasAndScaleAcrossLayers) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto storage = static_cast<int *>(nullptr);
  ASSERT_EQ(cudaMalloc(&storage, (resource_count * 3 + 2) * sizeof(int)), cudaSuccess);
  auto contents = std::array<int, resource_count * 3 + 2>{};
  std::iota(contents.begin(), contents.end(), 1);
  ASSERT_EQ(cudaMemcpy(storage, contents.data(), sizeof(contents), cudaMemcpyHostToDevice), cudaSuccess);
  auto capture_output = storage + resource_count * 3;
  auto lane_output = capture_output + 1;
  auto primary_pointers = std::array<int *, resource_count>{};
  auto control_pointers = std::array<int *, resource_count>{};
  auto replacements = std::array<xpool::ffnagent::ResourceReplacement, resource_count>{};
  auto values = xpool::ffnagent::LayerBindingValues{};
  for (auto i = std::size_t{0}; i < resource_count; ++i) {
    primary_pointers[i] = storage + i;
    control_pointers[i] = storage + resource_count + i;
    replacements[i] = {.primary_address = reinterpret_cast<std::uintptr_t>(primary_pointers[i]),
                       .control_address = reinterpret_cast<std::uintptr_t>(control_pointers[i]),
                       .target_address = reinterpret_cast<std::uintptr_t>(storage + 2 * resource_count + i),
                       .value_offset_bytes = offsets[i]};
  }
  auto primary = cudaGraph_t{};
  auto control = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&primary, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphCreate(&control, 0), cudaSuccess);
  for (auto graph : {primary, control}) {
    auto &pointers = graph == primary ? primary_pointers : control_pointers;
    auto args = std::array<void *, resource_count + 1>{};
    args[0] = &capture_output;
    for (auto i = std::size_t{0}; i < resource_count; ++i) {
      args[i + 1] = &pointers[i];
    }
    const auto node = xpool::utils::graph::add_kernel_node(
        graph, reinterpret_cast<const void *>(read_resources), dim3{1}, dim3{1}, 0, args.data());
    ASSERT_EQ(xpool::utils::graph::node_type(node), cudaGraphNodeTypeKernel);
  }
  auto parent = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&parent, 0), cudaSuccess);
  const auto embedded = xpool::utils::graph::embed_child_graph(parent, primary);
  const auto lane_replacements = std::array{xpool::ffnagent::LaneAddressReplacement{
      .capture_address = reinterpret_cast<std::uintptr_t>(capture_output), .bytes = sizeof(int),
      .target_address = reinterpret_cast<std::uintptr_t>(lane_output)}};
  const auto result = xpool::ffnagent::parameterize_graph(
      embedded, control, replacements, lane_replacements, 0, 0, 0);
  ASSERT_EQ(result.binding_sites.size(), resource_count);
  auto sites = static_cast<xpool::ffnagent::BindingSite *>(nullptr);
  auto device_values = static_cast<xpool::ffnagent::LayerBindingValues *>(nullptr);
  auto status = static_cast<cudaError_t *>(nullptr);
  ASSERT_EQ(cudaMalloc(&sites, sizeof(xpool::ffnagent::BindingSite) * resource_count), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_values, sizeof(values)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&status, sizeof(cudaError_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(sites, result.binding_sites.data(), sizeof(xpool::ffnagent::BindingSite) * resource_count,
                       cudaMemcpyHostToDevice), cudaSuccess);
  auto args = std::array<void *, 3>{&sites, &device_values, &status};
  const auto child = xpool::utils::graph::nodes(parent).front();
  const auto update_node = xpool::utils::graph::add_kernel_node(
      parent, reinterpret_cast<const void *>(rebind_resources), dim3{1}, dim3{1}, 0, args.data());
  xpool::utils::graph::add_dependency(parent, update_node, child);
  auto executable = cudaGraphExec_t{};
  ASSERT_EQ(cudaGraphInstantiate(&executable, parent, nullptr, nullptr, 0), cudaSuccess);
  for (auto layer : {std::size_t{0}, std::size_t{2}, std::size_t{0}}) {
    for (auto i = std::size_t{0}; i < resource_count; ++i) {
      *reinterpret_cast<std::uintptr_t *>(reinterpret_cast<unsigned char *>(&values) + offsets[i]) =
          reinterpret_cast<std::uintptr_t>(storage + layer * resource_count + i);
    }
    ASSERT_EQ(cudaMemcpy(device_values, &values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto observed_status = cudaErrorUnknown;
    auto output = 0;
    ASSERT_EQ(cudaMemcpy(&observed_status, status, sizeof(observed_status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&output, lane_output, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(observed_status, cudaSuccess);
    const auto begin = contents.begin() + static_cast<std::ptrdiff_t>(layer * resource_count);
    EXPECT_EQ(output, std::accumulate(begin, begin + resource_count, 0));
  }
  EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(parent), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(control), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(primary), cudaSuccess);
  EXPECT_EQ(cudaFree(status), cudaSuccess);
  EXPECT_EQ(cudaFree(device_values), cudaSuccess);
  EXPECT_EQ(cudaFree(sites), cudaSuccess);
  EXPECT_EQ(cudaFree(storage), cudaSuccess);
}

} // namespace
