"""Tests for domain.cube — SeismicCube and LabelCube."""

import numpy as np
import pytest

from seismic_visualizer.domain.cube import LabelCube, SeismicCube

# Small cube used across tests: (inline=4, crossline=5, time=6)
SHAPE = (4, 5, 6)


# ---------------------------------------------------------------------------
# SeismicCube
# ---------------------------------------------------------------------------


class TestSeismicCubeCreation:
    def test_float32_accepted(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        assert cube.shape == SHAPE

    def test_uint8_accepted(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.uint8))
        assert cube.shape == SHAPE

    def test_int8_accepted(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.int8))
        assert cube.shape == SHAPE

    def test_float64_accepted(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float64))
        assert cube.shape == SHAPE

    def test_2d_array_rejected(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            SeismicCube(np.zeros((5, 6)))

    def test_4d_array_rejected(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            SeismicCube(np.zeros((2, 3, 4, 5)))

    def test_complex_dtype_rejected(self) -> None:
        with pytest.raises(ValueError):
            SeismicCube(np.zeros(SHAPE, dtype=np.complex64))

    def test_string_dtype_rejected(self) -> None:
        with pytest.raises(ValueError):
            SeismicCube(np.zeros(SHAPE, dtype="U5"))


class TestSeismicCubeShape:
    def test_shape_tuple(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        assert cube.shape == SHAPE

    def test_shape_elements_are_ints(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        h, w, d = cube.shape
        assert isinstance(h, int)
        assert isinstance(w, int)
        assert isinstance(d, int)


class TestSeismicCubeReadOnly:
    def test_data_is_read_only(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        assert cube.data.flags.writeable is False

    def test_write_raises(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        with pytest.raises(ValueError):
            cube.data[0, 0, 0] = 1.0

    def test_original_array_still_writable(self) -> None:
        original = np.zeros(SHAPE, dtype=np.float32)
        SeismicCube(original)
        # SeismicCube must not mutate the caller's writeable flag
        assert original.flags.writeable is True

    def test_data_values_match_input(self) -> None:
        arr = np.arange(120, dtype=np.float32).reshape(SHAPE)
        cube = SeismicCube(arr)
        np.testing.assert_array_equal(cube.data, arr)


# ---------------------------------------------------------------------------
# LabelCube
# ---------------------------------------------------------------------------


class TestLabelCubeCreation:
    def test_uint8_accepted(self) -> None:
        cube = LabelCube(np.zeros(SHAPE, dtype=np.uint8))
        assert cube.shape == SHAPE

    def test_float32_rejected(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            LabelCube(np.zeros(SHAPE, dtype=np.float32))

    def test_int32_rejected(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            LabelCube(np.zeros(SHAPE, dtype=np.int32))

    def test_2d_rejected(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            LabelCube(np.zeros((5, 6), dtype=np.uint8))


class TestLabelCubeEmpty:
    def test_empty_creates_correct_shape(self) -> None:
        cube = LabelCube.empty(SHAPE)
        assert cube.shape == SHAPE

    def test_empty_is_all_zeros(self) -> None:
        cube = LabelCube.empty(SHAPE)
        assert np.all(cube.data == 0)

    def test_empty_dtype_is_uint8(self) -> None:
        cube = LabelCube.empty(SHAPE)
        assert cube.data.dtype == np.uint8


class TestLabelCubeMutability:
    def test_data_is_writable(self) -> None:
        cube = LabelCube.empty(SHAPE)
        assert cube.data.flags.writeable is True

    def test_mutation_persists(self) -> None:
        cube = LabelCube.empty(SHAPE)
        cube.data[1, 2, 3] = 42
        assert cube.data[1, 2, 3] == 42

    def test_data_is_not_a_copy(self) -> None:
        arr = np.zeros(SHAPE, dtype=np.uint8)
        cube = LabelCube(arr)
        # Writing through the cube's data must affect the original array
        cube.data[0, 0, 0] = 7
        assert arr[0, 0, 0] == 7

    def test_max_entity_id_constant(self) -> None:
        assert LabelCube.MAX_ENTITY_ID == 200
