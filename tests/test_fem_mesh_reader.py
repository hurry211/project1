"""Compatibility of the structured element reader with the native text format."""

import numpy as np
import pytest

from synthetic import write_level
from verifier.fem import ArtifactError, load_mesh_data


@pytest.mark.parametrize("geom", ["tri", "quad"])
def test_element_text_preserves_coordinate_bits_and_order(tmp_path, geom):
    level = tmp_path / "mesh"
    write_level(level, geom, 2, 1)
    path = level / "elements.txt"
    rows = [line.split() for line in path.read_text().splitlines()[1:]]
    rows[0][0] = "+00"
    rows[0][2] = "+" + rows[0][2]
    rows[0][3:9] = ["-0", "+1_0.5", "4.9406564584124654e-324",
                     "0.49999999999999994", "1e-01", "-1.0000000000000002"]
    coords = np.array([[float(word) for word in row[3:]] for row in rows])
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# header\r\n\r\n")
        stream.write("\r\n# between rows\r\n".join("\t".join(row) for row in rows))
    mesh = load_mesh_data(level)
    assert mesh.geom == geom
    np.testing.assert_array_equal(mesh.verts.reshape(coords.shape).view(np.uint64),
                                  coords.view(np.uint64))


@pytest.mark.parametrize(("field", "bad"), [
    (0, "1.0"), (0, "1e0"), (0, "0"), (0, "18446744073709551617"),
    (1, "qua"), (1, "quadx"), (1, "quad\x00suffix"), (1, "tri"),
    (2, "4.0"), (2, "5"), (3, "0x1p-2"), (3, "1\x00suffix"),
    (3, "NaN"), (3, "-inf"),
])
def test_bad_token_on_later_element_fails(tmp_path, field, bad):
    level = tmp_path / "mesh"
    write_level(level, "quad", 2, 1)
    path = level / "elements.txt"
    lines = path.read_text().splitlines()
    second = lines[2].split()
    second[field] = bad
    lines[2] = " ".join(second)
    path.write_text("\n".join(lines))
    with pytest.raises(ArtifactError):
        load_mesh_data(level)


@pytest.mark.parametrize("row", ["1 quad 4 0 0", "1 quad 4" + " 0" * 9,
                                 "1 quad 4" + " 0" * 8 + " # trailing", " # indented"])
def test_malformed_element_row_fails(tmp_path, row):
    level = tmp_path / "mesh"
    write_level(level, "quad", 2, 1)
    path = level / "elements.txt"
    lines = path.read_text().splitlines()
    lines[2] = row
    path.write_text("\n".join(lines))
    with pytest.raises(ArtifactError):
        load_mesh_data(level)
