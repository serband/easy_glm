"""Export the fitted rate tables for review in Excel.

Run ``advanced_pipeline.py`` first to create ``french_motor_model``.
"""

import easy_glm

model = easy_glm.EasyGLM.load("french_motor_model")
model.to_excel("french_motor_rate_tables.xlsx")

print("Saved the fitted rate tables as french_motor_rate_tables.xlsx")
