#include <cstddef>
#include <cstdint>
#include <variant>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <xpool/ffnagent/projection.hpp>

namespace {

xpool::ffnagent::MoeBindingResourceProjection resources(std::uintptr_t base) {
  return {.expert_gate_up_weight_address = base,
          .expert_down_weight_address = base + 16,
          .router = xpool::ffnagent::MoeRouterBindingResourceProjection{
              .weight_address = base + 32,
              .correction_bias_address = std::nullopt,
              .projection_bias_address = base + 48},
          .mxfp4 = xpool::ffnagent::Mxfp4ExpertBindingResourceProjection{
              .gate_up_scales_address = base + 64,
              .down_scales_address = base + 80,
              .gate_up_bias_address = base + 96,
              .down_bias_address = base + 112}};
}

xpool::ffnagent::ExecutionProjection projection() {
  const auto signature = xpool::ffnagent::MoeExecutionSignatureProjection{
      .payload_dtype = c10::ScalarType::BFloat16,
      .payload_row_capacity = 32,
      .hidden_size = 64,
      .local_intermediate_size = 64,
      .expert_count = 2,
      .effective_topk = 2,
      .routed_expert_count = 2,
      .primary_graph_address = 1,
      .control_graph_address = 2,
      .capture_input_address = 256,
      .capture_partial_address = 512,
      .capture_workspace_address = 768,
      .compute_workspace_bytes = 4096,
      .capture_routing_metadata_address = 1024,
      .capture_payload_rows_address = 2048,
      .primary_capture_resources = resources(4096),
      .control_capture_resources = resources(8192)};
  return {.signatures = {signature},
          .layers = {{.instance_index = 0, .layer_ordinal = 0, .execution_signature_indices = {0},
                      .layer_resource_targets = resources(12288)},
                     {.instance_index = 0, .layer_ordinal = 12, .execution_signature_indices = {0},
                      .layer_resource_targets = resources(16384)}}};
}

TEST(Mxfp4ProjectionTest, AdmitsCompleteIndependentLayerResources) {
  EXPECT_NO_THROW(projection().validate());
}

TEST(Mxfp4ProjectionTest, RejectsIncompleteTargetSchema) {
  auto value = projection();
  auto &target = std::get<xpool::ffnagent::MoeBindingResourceProjection>(value.layers[1].layer_resource_targets);
  target.mxfp4.reset();
  EXPECT_THROW(value.validate(), c10::Error);
  target = resources(16384);
  target.router->projection_bias_address.reset();
  EXPECT_THROW(value.validate(), c10::Error);
  target = resources(16384);
  target.router->correction_bias_address = target.router->projection_bias_address;
  target.router->projection_bias_address.reset();
  EXPECT_THROW(value.validate(), c10::Error);
}

TEST(Mxfp4ProjectionTest, RejectsUndiscoverableBiasAndScaleCaptures) {
  for (auto field = std::size_t{0}; field < 5; ++field) {
    auto value = projection();
    auto &signature = std::get<xpool::ffnagent::MoeExecutionSignatureProjection>(value.signatures[0]);
    const auto &primary = signature.primary_capture_resources;
    auto &control = signature.control_capture_resources;
    if (field == 0) {
      control.router->projection_bias_address = primary.router->projection_bias_address;
    }
    if (field == 1) {
      control.mxfp4->gate_up_scales_address = primary.mxfp4->gate_up_scales_address;
    }
    if (field == 2) {
      control.mxfp4->down_scales_address = primary.mxfp4->down_scales_address;
    }
    if (field == 3) {
      control.mxfp4->gate_up_bias_address = primary.mxfp4->gate_up_bias_address;
    }
    if (field == 4) {
      control.mxfp4->down_bias_address = primary.mxfp4->down_bias_address;
    }
    EXPECT_THROW(value.validate(), c10::Error) << field;
  }
}

TEST(Mxfp4ProjectionTest, RejectsNonBlockGeometryAndFloatingPayload) {
  auto value = projection();
  auto &signature = std::get<xpool::ffnagent::MoeExecutionSignatureProjection>(value.signatures[0]);
  signature.local_intermediate_size = 63;
  EXPECT_THROW(value.validate(), c10::Error);
  signature.local_intermediate_size = 64;
  signature.payload_dtype = c10::ScalarType::Half;
  EXPECT_THROW(value.validate(), c10::Error);
}

} // namespace
