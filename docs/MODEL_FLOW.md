# Build models with different variables

All models use the project's starting data and train/holdout split. Each new
stage has its own variable selection. A variable does not need the global
`predictor` role to be selected here: unassigned columns are available too.
Targets, weights, IDs, split columns and ignored columns are excluded.

Open **Model → Build a stacked model** to see the model flow.

1. Choose **Add standalone model**. Name the model and its first stage, choose
   GLM or CatBoost, select its variables, then save and fit.
2. On a fitted card, choose **Add model after this**. Select the new stage's
   variables and settings. Its offset is the complete fitted model on that card,
   including earlier accepted interaction tables, catalogue files and amendments.
3. Repeat to add further stages. For example, a main GLM can feed a vehicle
   CatBoost model, which then feeds an occupation model or another GLM.
4. Choose **Create alternative** on a child to try different variables or settings
   against the same fitted parent. The last stage is copied into the editor;
   the original named model remains unchanged.

Search the variable list to find a column. **Show selected** displays the current
selection even when it does not match the search. GLMs and interaction tables use
the shared bands on **Variables**. CatBoost uses raw input values.

Click a model card to inspect that complete model. Expand **Full recipe** to see
every component. Selecting a card does not replace an unfinished stage draft;
the add and alternative buttons explicitly start a new draft.

A connection records the fitted parent used to build a candidate. Refitting a
parent does not change its existing children. When that exact parent fit is no
longer available, the connection is marked historical. **Rebuild full model**
fits the child's saved recipe afresh, including its earlier stages. It does not
reuse the historical parent. Saving project JSON preserves the recipe and these
connections; JSON alone does not contain fitted model artifacts.

Existing single models remain available in **Single model**. Older multi-stage
models without parent information can be extended after fitting. Their alternative
action is disabled because the original parent cannot be identified safely.
