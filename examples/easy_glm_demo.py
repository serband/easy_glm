"""Create a project file for the browser workbench.

The workbench is useful after the first model: it gives a visual route through
data roles, factor design, diagnostics, rate tables and export. This script
creates the project file without launching a server:

    python examples/easy_glm_demo.py
    easy-glm-workbench french_motor.easyglm-project.json
"""

from pathlib import Path

from easy_glm.workflow import Project, VariableDesign

DATA = Path(__file__).resolve().parents[1] / "tests/fixtures/french_motor_50k.parquet"

project = Project(name="french_motor")
project.data.source.path = str(DATA)
project.data.roles = {
    "ClaimNb": "target",
    "Exposure": "weight",
    "DrivAge": "predictor",
    "Region": "predictor",
    "BonusMalus": "predictor",
    "Density": "predictor",
}
project.data.split.mode = "random"
project.data.split.fraction = 0.7
project.data.split.seed = 42
project.design.variables["Density"] = VariableDesign(kind="linear")

project.new_model(
    "frequency",
    family="poisson",
    divide_target_by_weight=True,
    predictors=["DrivAge", "Region", "BonusMalus", "Density"],
)
project.models["frequency"].penalty.alpha = 0.001
project.models["frequency"].penalty.cv = None

out = Path("french_motor.easyglm-project.json")
project.to_json(out)
print(f"Wrote {out}")
print(f"Validation: {project.validate('frequency') or 'ready for the workbench'}")
print(f"Open with: easy-glm-workbench {out}")
