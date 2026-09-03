"""Create a ready-to-open French motor project for the graphical workbench.

This script saves the data and project setup. Use the workbench itself to fit
and review the model on screen.
"""

import easy_glm
from easy_glm.workflow import Project

# Download the public French motor data and save the sample used by the project.
df = easy_glm.load_external_dataframe().sample(n=50_000, seed=42)
df.write_parquet("french_motor.parquet")

project = Project(name="French motor frequency")
project.data.source.path = "french_motor.parquet"

# Tell the workbench which columns hold the claim count and exposure, and
# which columns are the rating factors.
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

# Set up the Poisson model for annual claim frequency.
project.new_model(
    "Frequency",
    family="poisson",
    divide_target_by_weight=True,
    predictors=["DrivAge", "Region", "BonusMalus", "Density"],
)

project.to_json("french_motor_project.json")

print("Saved the project setup as french_motor_project.json")
print("Open it with: easy-glm-workbench french_motor_project.json")
