"""Diagnostic SVG geometry follows the supplied signed estimates and paths."""

import copy
import math
import xml.etree.ElementTree as ET

import pytest

from easy_glm.workflow._diagnostic_svg import (
    coefficient_path_chart,
    permutation_importance_chart,
)
from easy_glm.workflow._svg import BLUE, ORANGE


def parse(svg):
    root = ET.fromstring(svg)
    assert root.attrib["role"] == "img"
    assert root[0].tag == "{http://www.w3.org/2000/svg}title"
    assert not any(node.tag.endswith(("script", "image")) for node in root.iter())
    assert all("href" not in key for node in root.iter() for key in node.attrib)
    for node in root.iter():
        for key in ("x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "width", "height"):
            if key in node.attrib:
                assert math.isfinite(float(node.attrib[key]))
    return root


def elements(root, kind):
    return [node for node in root.iter() if node.attrib.get("class") == kind]


def path_rows(values, *, index=0, feature="Age", variable="Age"):
    return [
        {
            "feature_index": index,
            "feature": feature,
            "variable": variable,
            "alpha": alpha,
            "coefficient": value,
            "stage": 1,
            "l1_ratio": 1.0,
        }
        for alpha, value in values
    ]


def test_importance_signed_bars_and_whiskers_share_a_true_zero():
    rows = [
        {"variable": "Positive", "importance": 0.2, "std": 0.1},
        {"variable": "Negative", "importance": -0.1, "std": 0.2},
        {"variable": "Zero", "importance": 0.0, "std": 0.0},
    ]
    root = parse(permutation_importance_chart(rows, title="Importance"))
    zero = float(elements(root, "importance-zero")[0].attrib["x1"])
    positive, negative, null = elements(root, "importance-bar")
    assert positive.attrib["fill"] == BLUE
    assert negative.attrib["fill"] == ORANGE
    assert float(positive.attrib["x"]) == zero
    assert float(negative.attrib["x"]) < zero
    assert float(negative.attrib["x"]) + float(
        negative.attrib["width"]
    ) == pytest.approx(zero, abs=0.002)
    assert float(positive.attrib["width"]) == pytest.approx(
        2 * float(negative.attrib["width"]), abs=0.002
    )
    assert float(null.attrib["width"]) == 0
    assert float(elements(root, "importance-mean")[2].attrib["cx"]) == zero
    whisker = elements(root, "importance-whisker")[1]
    assert (
        230 <= float(whisker.attrib["x1"]) < zero < float(whisker.attrib["x2"]) <= 772
    )


def test_importance_preserves_ranking_full_labels_and_accessible_title():
    label = '<script>alert("x")</script> & unusually long predictor name'
    rows = [
        {"variable": label, "importance": -1},
        {"variable": "Second", "importance": 2},
    ]
    original = copy.deepcopy(rows)
    root = parse(
        permutation_importance_chart(rows, title='Importance <training> & "fit"')
    )
    assert root[0].text == 'Importance <training> & "fit"'
    assert [
        node.attrib["data-variable"] for node in elements(root, "importance-row")
    ] == [label, "Second"]
    assert elements(root, "importance-variable")[0][0].text == label
    assert rows == original


def test_missing_importance_and_sd_are_not_converted_to_zero():
    rows = [
        {"variable": "Missing", "importance": None, "std": 0.2},
        {"variable": "NaN", "importance": float("nan")},
        {"variable": "Infinite", "importance": float("inf")},
        {"variable": "No SD", "importance": 0.1},
    ]
    root = parse(permutation_importance_chart(rows, title="Missing"))
    assert len(elements(root, "importance-bar")) == 1
    assert not elements(root, "importance-whisker")
    assert [node.text for node in elements(root, "importance-value")] == [
        "—",
        "—",
        "—",
        "0.1",
    ]
    assert "Undefined" in "".join(root.itertext())


@pytest.mark.parametrize("values", [[], [0.0], [0.2, 0.2], [1e-20], [1e-310]])
def test_empty_constant_zero_and_small_importance_scales_stay_finite(values):
    rows = [
        {"variable": str(index), "importance": value}
        for index, value in enumerate(values)
    ]
    root = parse(permutation_importance_chart(rows, title="Degenerate"))
    assert len(elements(root, "importance-bar")) == len(values)
    if values and values[0] > 0:
        assert float(elements(root, "importance-bar")[0].attrib["width"]) > 400


@pytest.mark.parametrize(
    "importance,std",
    [(1, -0.1), (1e308, 1e308), ("bad", None)],
)
def test_invalid_importance_estimates_fail_clearly(importance, std):
    with pytest.raises(ValueError):
        permutation_importance_chart(
            [{"variable": "X", "importance": importance, "std": std}], title="Invalid"
        )


def test_small_values_use_short_scientific_labels_without_losing_displayed_sign():
    root = parse(
        permutation_importance_chart(
            [
                {"variable": "Small", "importance": 0.00602},
                {"variable": "Negative", "importance": -0.000637},
                {"variable": "Ordinary", "importance": 0.123456},
            ],
            title="Precision",
        )
    )
    assert [node.text for node in elements(root, "importance-value")] == [
        "6.02e-3",
        "-6.37e-4",
        "0.123",
    ]
    path = parse(
        coefficient_path_chart(
            path_rows([(0.001, -0.2), (0.01, 0.3)]),
            title="Path",
            selected_alpha=0.002223,
        )
    )
    assert "Selected λ = 2.223e-3" in "".join(path.itertext())


def test_coefficient_log_spacing_signed_geometry_and_selected_alpha():
    rows = path_rows([(0.1, 2), (0.001, -2), (0.01, 0)])
    original = copy.deepcopy(rows)
    root = parse(coefficient_path_chart(rows, title="Path", selected_alpha=0.01))
    points = elements(root, "coefficient-point")
    xs = [float(point.attrib["cx"]) for point in points]
    ys = [float(point.attrib["cy"]) for point in points]
    assert xs[1] - xs[0] == pytest.approx(xs[2] - xs[1], abs=0.002)
    assert ys[0] > ys[1] > ys[2]
    assert [float(point.attrib["data-coefficient"]) for point in points] == [-2, 0, 2]
    assert float(elements(root, "selected-alpha")[0].attrib["x1"]) == xs[1]
    assert float(elements(root, "coefficient-zero")[0].attrib["y1"]) == ys[1]
    assert len(elements(root, "coefficient-curve")) == 1
    assert rows == original


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_missing_coefficient_breaks_path_instead_of_imputing_or_joining(missing):
    root = parse(
        coefficient_path_chart(
            path_rows([(0.001, -2), (0.01, missing), (0.1, 2)]), title="Gap"
        )
    )
    assert not elements(root, "coefficient-curve")
    assert len(elements(root, "coefficient-point")) == 2


def test_absent_global_alpha_is_a_gap_and_feature_index_controls_identity():
    rows = path_rows([(0.001, 1), (0.1, 2)], index=0, feature="Same label")
    rows += path_rows([(0.001, 3), (0.01, 4), (0.1, 5)], index=1, feature="Same label")
    root = parse(coefficient_path_chart(rows, title="Distinct columns"))
    groups = {
        int(node.attrib["data-feature-index"]): node
        for node in elements(root, "coefficient-trajectory")
    }
    assert len(groups) == 2
    assert not elements(groups[0], "coefficient-curve")
    assert len(elements(groups[1], "coefficient-curve")) == 1


@pytest.mark.parametrize(
    "values", [[(0.1, 0)], [(0.001, 0), (0.1, 0)], [(0.1, -3)], [(0.1, 1e-310)]]
)
def test_fixed_constant_zero_and_tiny_coefficient_values_are_preserved(values):
    root = parse(coefficient_path_chart(path_rows(values), title="Fixed"))
    assert len(elements(root, "coefficient-curve")) == (len(values) > 1)
    assert [
        float(node.attrib["data-coefficient"])
        for node in elements(root, "coefficient-point")
    ] == [value for _, value in values]


def test_marker_outside_stored_path_expands_axis_without_inventing_points():
    root = parse(
        coefficient_path_chart(
            path_rows([(0.01, 1), (0.1, 2)]), title="Selection", selected_alpha=1
        )
    )
    points = elements(root, "coefficient-point")
    marker = elements(root, "selected-alpha")[0]
    assert float(points[-1].attrib["cx"]) < float(marker.attrib["x1"]) < 878
    assert len(points) == 2


@pytest.mark.parametrize("alpha", [0, -1, None, float("nan"), float("inf")])
def test_invalid_log_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="positive finite alpha"):
        coefficient_path_chart(path_rows([(alpha, 1)]), title="Invalid")
    if alpha is not None:
        with pytest.raises(ValueError, match="positive finite alpha"):
            coefficient_path_chart(
                path_rows([(1, 1)]), title="Invalid", selected_alpha=alpha
            )


def test_invalid_path_identity_and_mixed_contexts_are_rejected():
    rows = path_rows([(0.01, 1)])
    for bad in [
        rows * 2,
        rows + [dict(rows[0], alpha=0.1, feature="Different label")],
        rows + [dict(rows[0], feature_index=1, stage=2)],
        rows + [dict(rows[0], feature_index=1, l1_ratio=0.5)],
        [dict(rows[0], feature_index=-1)],
        [dict(rows[0], feature_index=True)],
    ]:
        with pytest.raises(ValueError):
            coefficient_path_chart(bad, title="Invalid")


def test_many_paths_keep_every_supplied_curve_with_a_small_legend():
    rows = [
        row
        for index in range(12)
        for row in path_rows(
            [(0.001, -(index + 1)), (0.01, 0), (0.1, index + 1)],
            index=index,
            feature=f"Coefficient {index}",
        )
    ]
    root = parse(coefficient_path_chart(rows, title="Many paths"))
    groups = elements(root, "coefficient-trajectory")
    assert len(groups) == len(elements(root, "coefficient-curve")) == 12
    highlighted = [node for node in groups if node.attrib["data-highlighted"] == "true"]
    assert [int(node.attrib["data-feature-index"]) for node in highlighted] == [
        11,
        10,
        9,
        8,
        7,
    ]
    assert len(elements(root, "coefficient-legend-label")) == 5
    assert len(elements(root, "coefficient-point")) == 15
    assert "7 in grey" in "".join(root.itertext())


def test_path_tooltips_escape_complete_feature_and_parent_labels():
    label = '<script>"coefficient" & term</script>'
    parent = "Age <years> & region"
    root = parse(
        coefficient_path_chart(
            path_rows([(0.01, 1), (0.1, -1)], feature=label, variable=parent),
            title="A < B",
        )
    )
    assert root[0].text == "A < B"
    assert (
        elements(root, "coefficient-trajectory")[0][0].text
        == label + "\nVariable: " + parent
    )
    assert elements(root, "coefficient-legend-label")[0][0].text == label


def test_empty_or_undefined_paths_remain_accessible_without_a_fabricated_line():
    for rows in ([], path_rows([(0.1, None)])):
        root = parse(coefficient_path_chart(rows, title="Unavailable"))
        assert "No finite coefficient" in "".join(root.itertext())
        assert not elements(root, "coefficient-curve")
