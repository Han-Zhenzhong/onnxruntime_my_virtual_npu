// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_SET_SCHEMA_EX(
    FastGelu,
    MyVirtualNpu,
    kMyCustomDomain,
    1,
    OpSchema()
        .SetDoc("FastGelu activation function implementation for virtual NPU domain.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
              return true;
            }));

void RegisterMyVirtualNpuSchemas() {
  // Register domain version range
  auto& domainToVersionRangeInstance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  domainToVersionRangeInstance.AddDomainToVersion(kMyCustomDomain, 1, 1);
  
  // Register FastGelu schema
  auto schema = ONNX_NAMESPACE::GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(MyVirtualNpu, 1, FastGelu)>();
  ONNX_NAMESPACE::RegisterSchema(schema);
}

}  // namespace contrib
}  // namespace onnxruntime