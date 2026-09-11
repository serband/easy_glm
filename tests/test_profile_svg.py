"""Profile SVGs preserve signed values, paired counts and accessible labels."""

import xml.etree.ElementTree as ET

import pytest

from easy_glm.workflow._profile_svg import correlation_matrix, mini_histogram
from easy_glm.workflow._svg import BLUE, GREY, ORANGE

NS = {"s": "http://www.w3.org/2000/svg"}


def parse(svg):
    root = ET.fromstring(svg)
    assert root.attrib["role"] == "img"
    assert root[0].tag == "{http://www.w3.org/2000/svg}title"
    assert not root.findall(".//s:script", NS)
    assert not root.findall(".//s:image", NS)
    assert all("href" not in key for element in root.iter() for key in element.attrib)
    return root


def elements(root, kind):
    return [element for element in root.iter() if element.attrib.get("class") == kind]


def test_histogram_escapes_labels_and_keeps_exact_counts_in_tooltips():
    title = 'Claim <frequency> & "rows"'
    label = '<script>alert("x")</script> & North'
    root = parse(mini_histogram([label, "South"], [12345, 1], title=title))
    assert root[0].text == title
    bins = elements(root, "profile-bin")
    assert len(bins) == 2
    assert bins[0][0].text == label + ": 12,345 rows"
    assert bins[1][0].text == "South: 1 row"
    assert "12.3k" in [element.text for element in root.findall(".//s:text", NS)]


def test_histogram_bar_heights_are_proportional_and_zero_is_not_fabricated():
    root = parse(mini_histogram(["0–1", "1–2", "2–3"], [0, 10, 20], title="Age"))
    bars = elements(root, "profile-bar")
    assert [float(bar.attrib["height"]) for bar in bars] == [0, 32, 64]
    assert len(elements(root, "profile-baseline")) == 1
    assert len(elements(root, "profile-bin-label")) == 3
    empty = parse(mini_histogram([], [], title="Empty"))
    assert not elements(empty, "profile-bar")
    assert "No observed values" in "".join(empty.itertext())


def test_histogram_long_category_edges_do_not_force_overlapping_middle_label():
    labels = [f"Very long category label {index}" for index in range(16)]
    root = parse(mini_histogram(labels, list(range(16)), title="Categories"))
    ticks = elements(root, "profile-bin-label")
    assert len(ticks) == 2
    assert [tick[0].text for tick in ticks] == [labels[0], labels[-1]]
    assert root.attrib["viewBox"] == "0 0 320 110"
    assert all(int(tick.attrib["font-size"]) >= 13 for tick in ticks)
    assert all(len(tick[0].tail) <= 14 for tick in ticks)
    wide = parse(mini_histogram(["W" * 30] * 16, list(range(16)), title="Wide labels"))
    wide_ticks = elements(wide, "profile-bin-label")
    assert len(wide_ticks) == 2
    assert all(len(tick[0].tail) <= 8 for tick in wide_ticks)


@pytest.mark.parametrize(
    "labels,counts",
    [(["A"], []), (["A"] * 17, [1] * 17), (["A"], [-1]), (["A"], [1.5])],
)
def test_histogram_rejects_inconsistent_input(labels, counts):
    with pytest.raises(ValueError):
        mini_histogram(labels, counts, title="Invalid")


def test_correlation_sign_and_undefined_have_distinct_colours_and_labels():
    root = parse(
        correlation_matrix(
            ["A", "B"],
            [[-1, 0], [1, None]],
            [[50, 40], [40, 0]],
            title="Correlations",
        )
    )
    cells = elements(root, "profile-correlation-cell")
    assert [cell[1].attrib["fill"] for cell in cells] == [BLUE, "#ffffff", ORANGE, GREY]
    assert [cell[2].text for cell in cells] == ["-1.00", "0.00", "1.00", "—"]
    text = "".join(root.itertext())
    assert all(value in text for value in ("−1", "+1", "0", "Undefined"))
    assert "r = undefined\nPaired rows: 0" in cells[-1][0].text


def test_correlation_tooltips_escape_pair_names_and_show_paired_counts():
    names = ['Claims <paid> & "open"', "Age > 30"]
    root = parse(
        correlation_matrix(
            names,
            [[1, -0.8754], [-0.8754, float("nan")]],
            [[2000, 1234], [1234, 0]],
            title="A & B",
        )
    )
    cells = elements(root, "profile-correlation-cell")
    assert (
        cells[1][0].text
        == names[0] + " × " + names[1] + "\nr = -0.875\nPaired rows: 1,234"
    )
    assert cells[-1][1].attrib["fill"] == GREY
    assert cells[-1][2].text == "—"


def test_twenty_variable_matrix_keeps_complete_tooltip_key_and_printable_size():
    names = [f"Very long numeric variable description {index}" for index in range(20)]
    root = parse(
        correlation_matrix(
            names,
            [[1.0] * 20 for _ in names],
            [[120] * 20 for _ in names],
            title="Twenty",
        )
    )
    _, _, width, height = map(int, root.attrib["viewBox"].split())
    assert width <= 1000 and height < 750
    assert len(elements(root, "profile-correlation-cell")) == 400
    rows = elements(root, "profile-row-label")
    columns = elements(root, "profile-column-label")
    assert [label[0].text for label in rows] == names
    assert [label.text for label in columns] == [None] * 20
    assert [label[0].tail for label in columns] == [
        str(index + 1) for index in range(20)
    ]
    assert all(len(label[0].tail) < 36 for label in rows)


@pytest.mark.parametrize(
    "names,values,counts",
    [
        (["A"], [], [[1]]),
        (["A"], [[0, 1]], [[1]]),
        (["A"], [[1.1]], [[1]]),
        (["A"], [[1]], [[-1]]),
        (["A"] * 21, [[1] * 21 for _ in range(21)], [[1] * 21 for _ in range(21)]),
    ],
)
def test_correlation_rejects_invalid_shape_range_or_counts(names, values, counts):
    with pytest.raises(ValueError):
        correlation_matrix(names, values, counts, title="Invalid")


def test_empty_and_single_variable_matrices_are_meaningful():
    empty = parse(correlation_matrix([], [], [], title="No numeric columns"))
    assert "No numeric variables" in "".join(empty.itertext())
    single = parse(correlation_matrix(["Constant"], [[None]], [[42]], title="Constant"))
    assert len(elements(single, "profile-correlation-cell")) == 1
    assert "r = undefined\nPaired rows: 42" in "".join(single.itertext())
