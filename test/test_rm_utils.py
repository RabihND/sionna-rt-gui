#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from sionna_rt_gui.rm_utils import radio_map_texture, radio_map_colorbar_to_image


@pytest.mark.parametrize("db_scale", [True, False])
def test_radio_map_texture(db_scale: bool):
    rng = dr.random.Philox4x32Generator(seed=1234)

    rm_values = rng.uniform(mi.TensorXf, shape=(100, 100))
    texture, alpha = radio_map_texture(
        rm_values, db_scale=db_scale, rm_cmap="magma", vmin=-100, vmax=0
    )
    assert texture.shape == (100, 100, 3)
    assert alpha.shape == (100, 100)


def test_colorbar_image():
    image = radio_map_colorbar_to_image(cmap="magma", vmin=-100, vmax=0, dpi=100)
    # The exact pixel width depends on how matplotlib lays the figure out, and
    # has drifted across versions, so only the shape that matters is asserted.
    assert image.ndim == 3 and image.shape[2] == 4
    assert image.shape[0] == pytest.approx(66, abs=4)
    assert image.shape[1] == pytest.approx(672, abs=16)
    assert np.mean(image[:, :, :3]) > 0.15
    assert np.mean(image[:, :, -1]) > 0.3
