# Workbench walkthrough

This is the browser workflow for building a pricing model in EasyGLM. It is
written for a pricing actuary who wants to make one modelling decision at a
time, review the result, and then decide what to do next.

> **Current local workbench guide.** The staged-model screens described here
> are newer than the published v0.472 package. Use the version of the
> workbench you are running to follow this guide; do not assume that an
> installed v0.472 package has these screens.

The example route is:

1. open a data set and define the shared response, exposure and split;
2. fit a small GLM of main effects;
3. inspect actual versus expected and search for missing factors;
4. add and review one or more interaction tables inside that GLM;
5. add vehicle, occupation and postcode models with their own variables; and
6. amend, compare and export the accepted complete score.

The workbench has two ways to build a model:

- **Single model** fits one GLM, CatBoost model or catalogue model.
- **Build in stages** fits named models in order. Every later model uses the
  complete score before it as its offset. This is the usual route for a main
  GLM followed by interaction tables and rating files.

Use **Build in stages** for the worked route below. A single model is useful
when you only need one fit or want to test a model family that is not available
for a staged model.

This guide covers the graphical interface. For the same kind of modelling work
in Python, use [Building a pricing model](pricing_walkthrough.md). For the
vehicle, occupation and postcode-file route, use
[Stacking models and building rating files](stacked_models.md). For a portable
project-setup JSON, use [Project workflow](project_workflow.md).

## Worked exercise: French motor claim frequency

Use the built-in French motor data to fit four main effects, add an interaction
table, then fit separate Vehicle and Area CatBoost corrections.

1. In **Project & data**, choose **Example dataset**, then select **Motor
   insurance — claim frequency**. If another dataset is already open, save the
   work you want to retain and tick **Replace this session**. Click **Load example**.
2. Choose **Build in stages**.
3. In the shared setup, confirm `ClaimNb` as the target, `Exposure` as the
   weight, and `IDpol` as the identifier. Leave `DrivAge`, `VehAge`,
   `BonusMalus`, `Density`, `Region`, `VehBrand`, `VehGas`, `VehPower` and
   `Area` available for model selection. Do not ignore them for this exercise.
4. Give the first card the name **Main effects** and select **GLM**. Add a
   second card named **Vehicle** with **CatBoost**, then a third card named
   **Area** with **CatBoost**.
5. In **Variables**, select `DrivAge`, `VehAge`, `BonusMalus` and `Density`
   for **Main effects**. Select `VehBrand`, `VehGas` and `VehPower` for
   **Vehicle**. Select `Area` and `Region` for **Area**. Each selection is for
   that one model only.
6. Fit **Main effects**. In its inline interaction review, add
   `DrivAge × BonusMalus`, click **Fit interaction**, inspect the fitted table,
   and choose **Keep interaction** if you want to include it.
7. Fit **Vehicle**, then **Area**. Leave CatBoost on its default **Optuna
   tuning** setting and choose a small trial or time budget for a quick
   demonstration.
8. In **Diagnostics**, inspect **Main effects**, then **After Vehicle**, then
   the complete score. These are cumulative checkpoints: each includes the
   score before it. A CatBoost stage that selects no change can leave two
   checkpoints with identical predictions.

The sections below explain the controls used at each step. Inspect the values
produced by your local fit.

## 1. Open the data

Open **Project & data** and load a CSV, Parquet or Excel file. The workbench
reads the file; it does not edit it. When replacing a session, first save any
project JSON or fitted scorer you want to retain, then confirm **Replace this
session**.

Once the file is open, choose **Single model** or **Build in stages**. That
choice sets the rest of the setup flow. It is not a request to fit anything.

For this walkthrough, choose **Build in stages**.

## 2. Define the project-wide columns

The next setup step assigns the columns that are shared by every model in the
chain. Set these before choosing rating factors:

| Column purpose | Example | What it controls |
| --- | --- | --- |
| Target | `ClaimNb` | The outcome being modelled. |
| Weight or exposure | `Exposure` | The policy-years used to fit claim frequency and calculate expected claim counts. |
| ID | `IDpol` | Identifies the record and excludes that column from factors. |
| Split | `traintest` | Identifies the training and holdout rows. |
| Ignore | a claims-payment flag or an operational field | Removes a column from factor selection and residual searches. |

If you use an existing split column, choose its **Training value**. The column
must have exactly two distinct values and no missing values; the other value
automatically becomes holdout. Check the displayed training and holdout row
counts before applying the assignments. Training data is used to fit models,
tune CatBoost and search for missing factors and interactions. Holdout data
stays aside until you ask to inspect it. The workbench's random split is
row-based. If repeated IDs must stay together, create and use a grouped split
column in your source data instead of relying on the ID role.

Use **Ignore** deliberately. An ignored column remains in the source data but
cannot appear as a main effect, CatBoost input, catalogue attribute or
interaction candidate. Use it for fields you know must not influence the model,
such as a post-outcome claim field. A column that is merely not selected yet
should remain unassigned so it can still be considered later.

The target, exposure, ID and split are locked for downstream stages. You choose
factors separately for each model later; do not try to repeat the shared column
setup in every stage.

## 3. Define the model route before choosing factors

For a staged model, **Project & data** shows numbered model cards. A card is a
placeholder for one model in the final score. Give each card a clear name and
choose its type:

| Model card | Typical type | Example factors | Why it is separate |
| --- | --- | --- | --- |
| Main effects | GLM | `DrivAge`, `VehAge`, `BonusMalus`, `Density` | Transparent core rating factors. |
| Vehicle | CatBoost or catalogue | `VehBrand`, `VehGas`, `VehPower` | A vehicle correction or a full vehicle rating file. |
| Area | CatBoost | `Area`, `Region` | A separate geographic correction in the French example. |
| Occupation or postcode | CatBoost or catalogue | occupation or postcode attributes | A separate correction or rating file on a different book. |

Click **Add model** to append a card. The order shown is the fitting order. The
second card is fitted with the first card's complete score as its offset; the
third uses the first two; and so on. The card describes that incoming score so
you can see the dependency before fitting.

Use a **GLM** for rate tables fitted from selected factors. Use **CatBoost**
when the stage is a flexible correction on raw values. Use **Catalogue** when
you need to score a complete vehicle, occupation or postcode lookup file,
including codes not represented in the experience data.

Do not add an interaction as a separate planned model. Interactions belong to
the GLM whose score they correct. Fit that GLM first, then add and review its
interaction tables inside the Model page. A pair can include an unassigned
predictor; it does not need to be a GLM main effect. A later model offsets the
GLM *and* every accepted interaction table in it.

The family and link are part of the planned response convention. A single
model can use the supported family and link appropriate to the outcome. Staged
models use a log link so their scores multiply; the current staged route
supports Poisson, Tweedie and Gaussian. If you need a different family or link,
build it as a single model rather than trying to attach a multiplicative stage.

## 4. Select variables for each model

Open **Variables** from the setup flow or use **Choose variables** on a model
card. The page identifies the model being edited and shows its place in the
chain. Its variables belong only to that model.

For a GLM, select the factors and define how each one is represented:

- Numeric factors are grouped into bands. Set a default number of bands, give
  one variable its own number, or enter explicit cut points. Automatic cuts are
  calculated from the training rows. Explicit cuts take precedence over a
  variable-specific band count, which takes precedence over the default.
- Categorical factors use levels. Restricting the number of retained levels
  keeps the most frequent levels in the training data and groups the rest as
  **Other**. Use this only where that treatment makes rating sense.
- The **Table** view is practical for a short list. The **JSON** view is for a
  large set of variables or for pasting a reviewed list. It changes the current
  model only.

For CatBoost, choose inputs from the same source columns but leave numeric
values raw and mark true categories as categorical. CatBoost does not use the
GLM bands. For a catalogue, upload the complete CSV or Parquet file, select its
unique key, then select the catalogue attributes. The key joins the catalogue
to the experience data; it is not itself a model factor.

This is where stage-specific selection matters. `DriverAge` can be a GLM main
effect, while `Make`, `Model` and `Power` appear only in the Vehicle model.
Selecting a factor in Vehicle does not put it into the main GLM. A factor may
also appear in more than one stage, but a later stage is then a correction to
the score already built, not a new estimate of the overall factor effect.

Save the variables for the current model. The setup flow takes you to the next
model or to **Model** when every card has a usable definition.

## 5. Fit the main-effects GLM

On **Model**, the cards are shown in order with their status and incoming
offset. The **Fit Model 1 — Main effects** section shows its fitting settings.
Check the family, link and selected factors, then click **Fit Model 1 — Main
effects**.

The fit is saved as a named checkpoint. It does not overwrite a previous fitted
version. When it completes, inspect the fitted main effects before adding more
complexity:

- Open **Diagnostics** and select the fitted checkpoint. It is a cumulative
  score, so the first model means the main-effects GLM alone.
- Inspect training actual versus expected overall, by each main factor and by
  predicted-risk band.
- Open **Explore** to look at one-way patterns and residual-factor suggestions.
  A residual search suggests things to investigate; it does not add them to the
  model for you.

Keep the first model small enough that its rate tables can be read and
challenged. If a missing-factor result looks useful, return to that model's
Variables setup, create a revised plan, add the factor and refit from that
model. The earlier fitted version remains available for comparison.

## 6. Search, fit and accept interaction tables

Once the main GLM is fitted, its interaction review appears directly beneath
that GLM on **Model**. The review belongs to the named GLM; it is not a global
interaction screen and it does not change the later Vehicle, Occupation or
Postcode cards until you accept a table.

Use **Manage interactions** to open a fitted GLM's review. Then use **Find
interactions** to rank residual pairs from the complete accepted training score.
The search can consider a main-effect factor and another eligible unassigned
predictor. Choosing a pair for an interaction does not add either variable as a
GLM main effect.

For a pair you want to investigate:

1. Choose the first and second variables, then click **Add interaction**.
2. Click **Fit interaction**. When the fit finishes, review the training loss
   comparison, the table or heatmap, the cell corrections and the supporting
   exposure or weight.
3. Choose **Keep interaction** to add that reviewed table to the GLM, or
   **Discard interaction** to remove it.

Only **Keep interaction** changes the accepted score. A candidate is separate
from the fitted chain, so fitting or discarding it never silently changes a
later model. A second accepted table offsets the GLM and the first accepted
table. Main-effect relativities stay fixed while the interaction tables are
being fitted.

If you change an accepted interaction or a main-effect table later, the
workbench marks downstream models as needing a refit. The existing fitted chain
is retained; the revised chain is a new version. Do not judge an unfinished
downstream suffix as though it were part of the accepted score.

## 7. Fit a CatBoost or catalogue stage

After the GLM and any accepted interactions are in place, fit the Vehicle card.
The fit heading tells you it starts from the complete preceding score. That
offset is automatic; you do not type predictions into a column.

For a new CatBoost or catalogue stage, **Optuna tuning** is the default. It
uses training rows only. Choose either an **Optuna trials** budget or a
**CatBoost tuning time** budget. The time budget covers that stage's candidate
CatBoost fits across training folds. It does not include fitting the earlier
GLM, data preparation, scoring, or the final fit on all training rows.

Choose **Manual settings** only when you want to fit a specified number of
iterations, depth, learning rate and regularisation without tuning. Results
record the completed trials and selected settings. “No change selected” means
the preceding score was not improved by the completed candidates; it is not a
missing result.

For a full catalogue file, check its coverage before you continue. The lookup
must contain every code you intend to score, including codes absent from
training experience. The exported Vehicle file is an incremental correction:
it needs the preceding main/interaction score when used in production. The
**complete** model score includes the core GLM, accepted interaction tables and
every accepted later stage.

Repeat the same sequence for Occupation and Postcode. Each can use a different
set of variables and a different model type. The final Postcode stage offsets
the complete accepted score through Occupation.

## Separate catalogue exercise: complete vehicle, occupation and postcode files

This exercise uses the synthetic experience and catalogue files made in
[Stacking models and building rating files](stacked_models.md#load-experience-and-the-complete-catalogues).
Run its loading section through the four `write_parquet` calls first. They write
these local files:

| File to open or upload | Purpose | Key or selected attributes |
| --- | --- | --- |
| `motor_experience.parquet` | Experience data opened in **Project & data** | `PolicyID`, `Claims`, `Exposure`, `traintest` |
| `vehicle_catalogue.parquet` | Complete Vehicle catalogue | Key: `VehicleCode`; attributes: `Make`, `Model`, `Power`, `BodyType` |
| `occupation_catalogue.parquet` | Complete Occupation catalogue | Key: `OccupationCode`; attributes: `OccupationGroup`, `PhysicalWork`, `DrivingForWork` |
| `postcode_catalogue.parquet` | Complete Postcode catalogue | Key: `Postcode`; attributes: `AreaType`, `Density`, `DistanceToCentre` |

All four are Parquet files so string keys, including vehicle code `"0061"`, are
kept exactly. Start a new staged plan with `motor_experience.parquet` and make
these shared assignments: `PolicyID` is the ID, `Claims` is the target,
`Exposure` is the weight, and `traintest` is the split. In the **Training value**
dropdown, choose **1 · 4,500 rows**. The other value, `0`, becomes holdout
(1,500 rows). Click **Preview changes**, then **Apply changes** before continuing.

Set up the following cards in this order. The workbench supplies the common
family, response, split and predecessor offset; do not create an offset column
or re-enter those settings in a later card.

| Card name | Type | Variables or file setup |
| --- | --- | --- |
| Main effects | GLM | `DriverAge`, `BonusMalus`, `Region` |
| Vehicle | Catalogue | Upload `vehicle_catalogue.parquet`, select `VehicleCode`, then select `Make`, `Model`, `Power`, `BodyType` |
| Occupation | Catalogue | Upload `occupation_catalogue.parquet`, select `OccupationCode`, then select `OccupationGroup`, `PhysicalWork`, `DrivingForWork` |
| Postcode | Catalogue | Upload `postcode_catalogue.parquet`, select `Postcode`, then select `AreaType`, `Density`, `DistanceToCentre` |

Save the variables for each card, then fit Main effects, Vehicle, Occupation
and Postcode in that order. Leave each catalogue on **Optuna tuning** for the
first run and choose a trial or time budget. After each fit, inspect its named
cumulative checkpoint in **Diagnostics**. The Vehicle lookup is a correction
to Main effects; Occupation corrects Main effects plus Vehicle; Postcode
corrects all three earlier cards.

On **Export**, download each **full lookup CSV** only after the relevant
catalogue stage is accepted. Those lookups cover the complete uploaded catalogue,
including keys with no training experience. They are still incremental rating
files: production scoring needs the preceding score as well as the lookup. Save
**Export project** to reopen the plan, and download a frozen scorer when you
need to score the full accepted model without refitting.

## 8. Inspect the right diagnostic scope

Open **Diagnostics** after each meaningful decision. Select the named fitted
version and then the checkpoint to inspect:

- **Main effects** is the core GLM alone.
- A named accepted interaction shows the GLM through that interaction.
- **After Vehicle** includes the core GLM, its accepted tables and the vehicle
  correction.
- The complete model includes every fitted stage in that version.

These are actual-versus-expected results for the original response. They are
not isolated correction multipliers. Training is selected first. Choose
holdout only for an accepted candidate or saved version that you are ready to
evaluate independently.

Use **Compare** to put two saved versions side by side. Use **Rate tables** to
review or amend table values. An amendment starts with a preview. Applying it
creates a new version; it never changes the saved fitted version you started
from. When the workbench asks whether to refit later components, choose:

- **Refit later components** when a changed earlier score should be the offset
  used to refit Vehicle, Occupation and Postcode.
- **Keep later components** when you want to preserve their existing fitted
  tables and inspect the resulting combined score before deciding whether to
  refit them.

## 9. Revise, extend or start again

Use **Create revised plan** when changing a fitted model's factors, method or
interaction structure. It creates a new working version and protects the
previous fitted chain. That model and the models after it must be fitted again;
an unchanged earlier prefix is kept.

Use **Add model** to extend a plan after its current final stage. The new card
inherits the shared response and split and starts from the complete existing
score. It still needs its own variables and fitting settings.

Use **Start a new plan** when you want to replace the whole working structure.
It clears the working plan after confirmation but does not delete saved fitted
versions. Use **Revise shared assignments** only when changing the target,
exposure/weight or split. It creates a new plan that must be fitted again from
the first model.

## 10. Export the artifact you actually need

**Export project** saves the setup: data location, shared assignments, model
plan, variable definitions and other project settings. It is the right artifact
for reopening and editing the workbench plan. It is not a deployable fitted
model by itself.

For a selected accepted fitted version, the Export page offers different
artifacts for different jobs:

| Need | Export | What it does |
| --- | --- | --- |
| Score new data now | **Download JSON scorer** or **Download Python scorer** | Uses the saved fitted model without refitting. |
| Rebuild the accepted model later | **Download Python workflow** | Refits the accepted models and interaction tables from source data in order, then writes a new scorer and Excel workbook. Change `SOURCE_PATH` if the source moved. Refit values can change. |
| Give a rating-file owner the tables | **Download Excel component workbook** and **Download full lookup CSV** for catalogue stages | Supplies the component tables or full catalogue lookup. Preserve the preceding score required by an incremental file. |

The workflow export contains accepted components, amendments and offset order.
It does not include exploratory searches or discarded interaction candidates.

For the exact current rules behind this interface, see
[Model flow](../docs/MODEL_FLOW.md). For a portable setup file and its Python
validation route, see [Project workflow](project_workflow.md).
