# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.eq.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_eq_Tensor"

input_t = Tuple[torch.Tensor]


class Equal(torch.nn.Module):
    def __init__(self, input, other):
        super().__init__()
        self.input_ = input
        self.other_ = other

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        return input_ == other_

    def get_inputs(self):
        return (self.input_, self.other_)


op_eq_rank1_ones = Equal(
    torch.ones(5),
    torch.ones(5),
)
op_eq_rank2_rand = Equal(
    torch.rand(4, 5),
    torch.rand(1, 5),
)
op_eq_rank3_randn = Equal(
    torch.randn(10, 5, 2),
    torch.randn(10, 5, 2),
)
op_eq_rank4_randn = Equal(
    torch.randn(3, 2, 2, 2),
    torch.randn(3, 2, 2, 2),
)

test_data_common = {
    "eq_rank1_ones": op_eq_rank1_ones,
    "eq_rank2_rand": op_eq_rank2_rand,
    "eq_rank3_randn": op_eq_rank3_randn,
    "eq_rank4_randn": op_eq_rank4_randn,
}


@common.parametrize("test_module", test_data_common)
def test_eq_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_common)
def test_eq_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_common)
def test_eq_u55_BI(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_module", test_data_common)
def test_eq_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module,
        test_module.get_inputs(),
        aten_op,
        exir_op,
        run_on_fvp=False,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_common)
@pytest.mark.skip(reason="The same as test_eq_u55_BI")
def test_eq_u55_BI_on_fvp(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {exir_op: 1},
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_common,
    xfails={"eq_rank4_randn": "4D fails because boolean Tensors can't be subtracted"},
)
@common.SkipIfNoCorstone320
def test_eq_u85_BI_on_fvp(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module,
        test_module.get_inputs(),
        aten_op,
        exir_op,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
