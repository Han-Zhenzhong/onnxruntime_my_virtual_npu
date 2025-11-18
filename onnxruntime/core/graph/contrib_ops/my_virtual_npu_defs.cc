// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA_EX(
    FastGelu,
    MyVirtualNpu,
    ::onnxruntime::kMyCustomDomain,
    1,
    true,
    OpSchema()
        .SetDoc("FastGelu activation function implementation for virtual NPU domain.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext&, const OpSchema&, FunctionProto&) {
              return true;
            }));

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {

void RegisterMyVirtualNpuSchemas() {
  static bool schemas_registered = false;
  if (schemas_registered) {
    return;  // Already registered, skip
  }

  // Register domain version range
  auto& domainToVersionRangeInstance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  domainToVersionRangeInstance.AddDomainToVersion(kMyCustomDomain, 1, 1);

  // Register FastGelu schema
  auto schema = ONNX_NAMESPACE::GetOpSchema<ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(MyVirtualNpu, 1, FastGelu)>();
  ONNX_NAMESPACE::RegisterSchema(schema);

  schemas_registered = true;
}

}  // namespace contrib
}  // namespace onnxruntime
