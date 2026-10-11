# Workbench walkthrough

Use the French motor example to fit a GLM, review an interaction, then fit
Vehicle and Area models on what remains. Each model gets its own variables;
each later fit uses the predictions from all earlier models as its offset.

We will also cover changing an earlier model, checking A/E and exporting rates.
For the same modelling actions in Python, use
[Building a pricing model](pricing_walkthrough.md).

## Worked exercise: French motor claim frequency

Build these three models:

| Order | Name | Type | Variables |
| --- | --- | --- | --- |
| 1 | Main effects | GLM | `DrivAge`, `VehAge`, `BonusMalus`, `Density` |
| 2 | Vehicle | CatBoost | `VehBrand`, `VehGas`, `VehPower` |
| 3 | Area | CatBoost | `Area`, `Region` |

The GLM supplies the main-effect rates. Vehicle fits a correction to the GLM
and any interaction tables you keep. Area fits a further correction to that
combined model.

## 1. Open the data

In **Project & data**, choose **Example dataset**, then **Motor insurance —
claim frequency**. Click **Load example**. If another dataset is open, save
any work you want to retain and tick **Replace this session** first.

Choose **Build in stages** for this exercise. Choose **Single model** when
you want one GLM, CatBoost or catalogue model without later models.

You can open CSV, Parquet or Excel data here as well. The workbench reads the
file and leaves the source unchanged.

## 2. Define the project-wide columns

Assign the columns shared by all three models:

| Assignment | French motor column | Use |
| --- | --- | --- |
| Target | `ClaimNb` | Claim count |
| Weight or exposure | `Exposure` | Policy-years for fitting frequency and calculating expected claims |
| ID | `IDpol` | Record identifier, excluded from factors |
| Split, if supplied | `traintest` | Existing training and holdout assignments |

For an existing split column, choose its **Training value**. The other value
becomes holdout. The column must contain exactly two values and no missing
values. Check the row counts, then click **Preview changes** and **Apply
changes**.

Training rows are used for fitting, tuning and factor searches. Holdout is
reserved for the later validation check. The workbench's random split assigns
rows individually. If you need all rows for one ID to stay together, supply a
grouped split column in the data.

Use **Ignore** for columns you want to exclude from every model and search,
such as information only known after a claim. Leave potential rating factors
unassigned until you choose them for a model. For this exercise, keep the nine
factors listed for Main effects, Vehicle and Area available.

Later models share this target, exposure and split. You will choose their
rating factors separately.

## 3. Define the model route before choosing factors

Name the first model **Main effects** and choose **GLM**. Use **Add model** to
add **Vehicle** as **CatBoost**, then **Area** as **CatBoost**.

The listed order is the fitting order. Each model shows the preceding model
whose combined prediction it will use as an offset. There is no prediction
column to create or pass between these models yourself.

Use these types:

| Type | When to use it |
| --- | --- |
| GLM | Fit main effects as rating tables. Add interaction tables after reviewing the fit. |
| CatBoost | Fit a flexible model from the selected raw variables. |
| Catalogue | Build a full vehicle, occupation or postcode file, including codes with no training experience. |

For this claim-frequency model, choose Poisson, Log and **Divide target by
weight**. Staged models use a log link so their corrections multiply. The
staged route currently supports Poisson, Tweedie and Gaussian; other supported
family/link choices are available in the single-model route.

Interactions belong to a GLM. You will add them after fitting that GLM, rather
than putting them in the model plan upfront.

## 4. Select variables for each model

Use **Choose variables** on Main effects. The Variables setup step shows
which model you are editing. Select `DrivAge`, `VehAge`, `BonusMalus` and
`Density`.

For numeric factors, choose a number of bands or enter your own cuts.
Automatic boundaries use training data. A variable's explicit cuts take
precedence over its own band count, which takes precedence over the default.
For example, driver-age cuts at 25 and 35 put ages 25–34 in the same band.

Categorical factors have a relativity per retained level. If you limit the
number of levels, EasyGLM keeps the most frequent training levels and puts
the rest in **Other**.

Use **Table** to edit factors individually. Use **JSON** to paste or amend a
large list. Both views change only the selected model; shared columns such
as target and split stay locked.

Save the Main effects variables, then choose the Vehicle inputs:
`VehBrand`, `VehGas` and `VehPower`. Leave power as raw numeric and the brand
and fuel fields as categories. CatBoost uses raw values, not the GLM bands.

Save those choices, then select `Area` and `Region` for Area. Selecting a
variable here does not add it to either earlier model.

When every model is defined, continue to **Model**.

## 5. Fit the main-effects GLM

On **Model**, check the factors and settings under **Fit Model 1 — Main
effects**, then click its fit button.

When the fit completes, review it before fitting Vehicle:

- In **Diagnostics**, select Main effects. Check overall A/E, A/E by each
  factor and lift by predicted risk.
- In **Explore**, look for one-way patterns or run the missing-factor search.
  A suggested factor is a choice to investigate; it is not added automatically.

If you want to add a factor, return to that model's Variables setup, create a
revised plan, change the selection and refit. The previous fitted version stays
available for comparison.

## 6. Search, fit and accept interaction tables

The interaction review appears directly beneath Main effects. Open it with
**Manage interactions** and use **Find interactions** to suggest pairs based
on the current model's training residuals.

Try driver age by bonus-malus:

1. Choose `DrivAge` and `BonusMalus`, then click **Add interaction**.
2. Click **Fit interaction**.
3. Inspect the fitted corrections, A/E and exposure by cell, and the training
   validation comparison with the GLM.
4. Choose **Keep interaction** to include the table, or **Discard interaction**.

Fitting a candidate leaves the model's score unchanged until you keep it.
A kept table multiplies the GLM prediction. A second interaction is fitted
against the GLM and the first kept table. The main-effect rates stay fixed.

A pair can include an unassigned factor, such as Region, without adding it
as a main effect. After keeping or discarding the pair, continue to Vehicle.

## 7. Fit a CatBoost or catalogue stage

Fit Vehicle next, then Area. Vehicle automatically uses Main effects and any
kept interaction tables as its offset. Area uses that score plus Vehicle.

CatBoost defaults to **Optuna tuning**, using training data only. Choose an
**Optuna trials** budget or a **CatBoost tuning time** budget. A trial is one
set of CatBoost settings tested across training folds. The time limit covers
that model's tuning fits, not the earlier GLM or final fit on all training rows.

To choose the settings yourself, select **Manual settings** and enter
iterations, depth, learning rate and regularisation. The result records the
settings used. “No change selected” means none of the tested corrections
improved the preceding score.

After each fit, inspect A/E for the combined prediction through that model.
Vehicle's correction alone is not a predicted claim frequency.

## Separate catalogue exercise: complete vehicle, occupation and postcode files

For full rating files, use the sample data in
[Stacking models and building rating files](stacked_models.md#load-experience-and-the-complete-catalogues).
Run its loading blocks through the four `write_parquet` calls to create:

| File | Use in the workbench |
| --- | --- |
| `motor_experience.parquet` | Open as the modelling data |
| `vehicle_catalogue.parquet` | Full Vehicle file |
| `occupation_catalogue.parquet` | Full Occupation file |
| `postcode_catalogue.parquet` | Full Postcode file |

Open `motor_experience.parquet` and start a staged plan. Assign `PolicyID` as
ID, `Claims` as target, `Exposure` as weight and `traintest` as split. Choose
**1 · 4,500 rows** as the Training value; `0` becomes the 1,500-row holdout.
Preview and apply the assignments.

Define these models in order:

| Name | Type | Variables or catalogue settings |
| --- | --- | --- |
| Main effects | GLM | `DriverAge`, `BonusMalus`, `Region` |
| Vehicle | Catalogue | Upload `vehicle_catalogue.parquet`; key `VehicleCode`; attributes `Make`, `Model`, `Power`, `BodyType` |
| Occupation | Catalogue | Upload `occupation_catalogue.parquet`; key `OccupationCode`; attributes `OccupationGroup`, `PhysicalWork`, `DrivingForWork` |
| Postcode | Catalogue | Upload `postcode_catalogue.parquet`; key `Postcode`; attributes `AreaType`, `Density`, `DistanceToCentre` |

The key joins each file to the experience data. The attributes are the
CatBoost inputs. Use Parquet to preserve string keys such as `"0061"`.

Save the variables and fit the four models in order. Each catalogue fit
corrects the complete prediction from the earlier models. Inspect its A/E
before continuing.

On **Export**, download the **full lookup CSV** for each fitted catalogue.
It contains every uploaded key, including keys without training rows. Each
file contains a multiplier to use with the preceding models' prediction.
Download a fitted scorer as well if you need the combined prediction.

## 8. Inspect the right diagnostic scope

In **Diagnostics**, choose the fitted version and the part of the model to
inspect:

| Selection | Prediction being checked |
| --- | --- |
| Main effects | GLM alone |
| An accepted interaction | GLM plus tables through that interaction |
| After Vehicle | GLM, kept interactions and Vehicle |
| Complete model | All fitted models and kept tables |

A/E compares actual claims with the combined expected claims at that point.
Training rows are shown first. Choose holdout when you are ready to check the
model on rows kept out of development.

Use **Compare** to compare saved versions. Use **Rate tables** to amend rates:
preview the change, inspect its effect, then apply it to a new version. Choose
**Refit later components** to re-estimate later models using the changed rates,
or **Keep later components** to retain their rates and inspect the combined
effect of your amendment.

## 9. Revise, extend or start again

| What you want to do | Action | What happens |
| --- | --- | --- |
| Change factors or method in an existing model | **Create revised plan** | Keeps the old fitted version. Refit the changed model and those after it; earlier unchanged fits are kept. |
| Add another model after the last one | **Add model** | Adds a model with its own variables, using the combined earlier prediction as offset. |
| Replace the whole working plan | **Start a new plan** | Clears the current plan after confirmation. Saved fitted versions remain available. |
| Change target, exposure or split | **Revise shared assignments** | Creates a plan that needs fitting again from the first model. |

Changing a GLM rate or an accepted interaction also means later models may
need refitting. The workbench shows which models need a new fit. Review the
revised complete model after those fits finish.

## 10. Export the artifact you actually need

| What you need | Download |
| --- | --- |
| Reopen and edit the setup | **Export project** |
| Score using the fitted rates, without refitting | **Download JSON scorer** or **Download Python scorer** |
| Refit the model from source data in Python | **Download Python workflow** |
| Review or implement rating tables | **Download Excel component workbook**, plus each catalogue's **full lookup CSV** |

The Python workflow rebuilds the accepted models and interactions in order,
including amendments and offsets, then writes a scorer and Excel workbook.
Change `SOURCE_PATH` if the data has moved. Fresh fitted values can differ.
Searches and discarded interactions are not included.

For scoring additional datasets, see [Score new data](new_data_scoring.md).
For using the exported project in Python, see [Project workflow](project_workflow.md).
