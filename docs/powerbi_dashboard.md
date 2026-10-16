# Power BI dashboard

Power BI Desktop is free and connects to Azure SQL natively. Install it from the
Microsoft Store ("Power BI Desktop").

## 1. Connect

**Quickest:** double-click a connection file in `warehouse/`:

- `FairnessGates_local.pbids`: LocalDB on this machine (works now)
- `FairnessGates_azure.pbids`: Azure SQL. First open it in a text editor and replace
  `<server-name>` with your server name.

Power BI asks how to sign in. For **LocalDB** choose *Windows → Use my current
credentials*. For **Azure** choose *Database* and enter `dashboard_reader` and its
password (see `docs/azure_setup.md` §6).

**Or manually:** Home, then **Get data → SQL Server**. Server is `(localdb)\MSSQLLocalDB`
or `<server-name>.database.windows.net`, database `FairnessGates`, mode **Import**.

In the Navigator, tick all eight tables: `DimDate`, `DimRun`, `DimDataset`,
`DimModel`, `DimGroup`, `DimMetric`, `FactFairnessAudit`, `FactModelEvaluation`.
Then **Load**.

## 2. Model (the star schema)

Open **Model view**. Power BI usually detects the relationships from the matching
key names. Check that each fact table has a one-to-many line *from* each dimension:

| Dimension key | -> FactFairnessAudit | -> FactModelEvaluation |
|---|---|---|
| `DimRun[RunKey]` | `RunKey` | `RunKey` |
| `DimDate[DateKey]` | `DateKey` | `DateKey` |
| `DimDataset[DatasetKey]` | `DatasetKey` | `DatasetKey` |
| `DimModel[ModelKey]` | `ModelKey` | `ModelKey` |
| `DimGroup[GroupKey]` | `GroupKey` | (none) |
| `DimMetric[MetricKey]` | `MetricKey` | (none) |

Delete any relationship *between* the two fact tables or between `DimRun` and
`DimDate`. Facts connect only through dimensions. Then select `DimDate`, choose
**Table tools → Mark as date table**, and pick `FullDate`.

## 3. Measures

In the **Data** pane, right-click `FactFairnessAudit`, choose **New measure**, and add
each of these:

```DAX
Gap = AVERAGE ( FactFairnessAudit[MetricValue] )
Gap CI Lower = AVERAGE ( FactFairnessAudit[CiLower] )
Gap CI Upper = AVERAGE ( FactFairnessAudit[CiUpper] )
Threshold = MAX ( FactFairnessAudit[Threshold] )
Gated Failures = CALCULATE ( COUNTROWS ( FactFairnessAudit ), FactFairnessAudit[Passed] = FALSE () )
```

On `FactModelEvaluation`:

```DAX
Builds = DISTINCTCOUNT ( FactModelEvaluation[RunKey] )
Builds Passed = CALCULATE ( DISTINCTCOUNT ( DimRun[RunKey] ), DimRun[BuildPassed] = TRUE () )
Build Pass Rate = DIVIDE ( [Builds Passed], DISTINCTCOUNT ( DimRun[RunKey] ) )
Accuracy = AVERAGE ( FactModelEvaluation[Accuracy] )
Approval Rate = AVERAGE ( FactModelEvaluation[ApprovalRate] )
```

Format `Gap`, `Accuracy`, `Approval Rate` and `Build Pass Rate` as percentages, or
as decimals with 3 places.

## 4. Pages (from the learning plan)

Add **slicers** on every page (copy them with "sync slicers"):
`DimDataset[DatasetName]`, `DimModel[ModelName]`, `DimMetric[MetricName]`,
`DimGroup[AttributeName]`.

### Page 1: Fairness over time
- **Line chart.** X-axis: `DimRun[RunStartedAt]`. Y-axis: `Gap`. Legend: `DimGroup[AttributeName]`.
  - Visual-level filter: `DimMetric[IsGated] = True`, `DimGroup[GroupName] = all`.
  - Analytics pane: **Error bars**. Upper bound: `Gap CI Upper`; lower bound: `Gap CI Lower`.
  - Analytics pane: **Constant line** at 0.10, labeled "threshold". Or add `Threshold` as a second line.
- Read it as: a line crossing the threshold means that attribute failed the gate in that build.

### Page 2: Pass/fail by build
- **Cards:** `Builds`, `Builds Passed`, `Build Pass Rate`.
- **Table:** `DimRun[RunStartedAt]`, `DimRun[RunId]`, `DimRun[Branch]`, `DimRun[CommitSha]`,
  `DimRun[ChampionModel]`, `DimRun[BuildPassed]`, `Gated Failures`.
  - Conditional formatting on `BuildPassed`: green for True, red for False.
- **Matrix:** rows `DimModel[ModelName]`, columns `DimGroup[AttributeName]`, values `Gated Failures`.
  This shows exactly which vendor failed on which attribute.

### Page 3: Per-vendor comparison
- **Clustered bar.** Axis: `DimModel[Vendor]` and `DimModel[ModelName]`. Value: `Gap`.
  Legend: `DimMetric[MetricName]`. Filter: `IsGated = True`. Add the 0.10 constant line.
- **Scatter.** X: `Accuracy`. Y: `Gap` (filter one metric, e.g. equalized odds). Details:
  `DimModel[ModelName]`. This shows the accuracy-vs-fairness trade-off.
- **Table** from `FactModelEvaluation`: `ModelName`, `Role`, `Accuracy`, `RocAuc`,
  `Approval Rate`, `GatePassed`, `Recommended`. Highlight the recommended row.

### Page 4 (optional): Group detail
- **Bar chart.** Axis: `DimGroup[GroupName]`. Value: `Gap`. Filter `DimMetric[MetricName] = selection_rate`
  (or `false_positive_rate`), one attribute at a time, with CI error bars.
- Add `FactFairnessAudit[GateEligible]` to tooltips, so small groups show as "reported, not gated".

## 5. Refresh

**Home → Refresh** pulls the latest builds. Each CI run adds one build to `DimRun`.
For the demo, run the workflow twice from the Actions tab: once with the default
champion (fails) and once with `vendor_a_mitigated` (passes). Then refresh. Both builds
appear on page 2.

Save the report as `dashboard/FairnessGates.pbix`. It holds a copy of the data, so
it's fine to commit it as long as you're comfortable with the audit numbers being in
the repo. No credentials are stored in the file.
