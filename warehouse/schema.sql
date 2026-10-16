/*
  Fairness Gates warehouse -- star schema for Azure SQL (also runs on SQL Server / LocalDB).
  Safe to re-run: every object is created only if it does not exist.

  FACT GRAIN
    FactFairnessAudit   one row per build x dataset x model x protected group x metric
                        (gated gaps use the group "all" of their attribute)
    FactModelEvaluation one row per build x dataset x model (accuracy, gate result, role)

  DIMENSIONS (surrogate INT keys; natural keys are UNIQUE)
    DimDate     calendar day of the build             natural key: DateKey = yyyymmdd
    DimRun      one CI build / local run              natural key: RunId
    DimDataset  german_credit, hmda                    natural key: DatasetName
    DimModel    vendor models                          natural key: ModelName   (SCD type 1)
    DimGroup    protected attribute + group            natural key: (AttributeName, GroupName)
    DimMetric   accuracy ... equalized_odds_difference natural key: MetricName

  Slowly changing dimensions: DimModel is type 1 (a new description overwrites the old).
  Which model was champion at the time is NOT a model attribute -- it is recorded per build
  in DimRun.ChampionModel and FactModelEvaluation.Role, so history is never lost.
*/

IF OBJECT_ID('dbo.DimDate') IS NULL
CREATE TABLE dbo.DimDate (
    DateKey     INT          NOT NULL PRIMARY KEY,   -- yyyymmdd
    FullDate    DATE         NOT NULL UNIQUE,
    [Year]      SMALLINT     NOT NULL,
    [Quarter]   TINYINT      NOT NULL,
    [Month]     TINYINT      NOT NULL,
    MonthName   NVARCHAR(10) NOT NULL,
    [Day]       TINYINT      NOT NULL,
    DayName     NVARCHAR(10) NOT NULL
);
GO

IF OBJECT_ID('dbo.DimRun') IS NULL
CREATE TABLE dbo.DimRun (
    RunKey        INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    RunId         NVARCHAR(64)  NOT NULL UNIQUE,   -- GitHub run id.attempt, or local-<timestamp>
    RunStartedAt  DATETIME2(0)  NOT NULL,
    DateKey       INT           NOT NULL REFERENCES dbo.DimDate(DateKey),
    CommitSha     NVARCHAR(40)  NULL,
    Branch        NVARCHAR(200) NULL,
    [Trigger]     NVARCHAR(40)  NOT NULL,          -- push, pull_request, workflow_dispatch, local
    ChampionModel NVARCHAR(100) NOT NULL,
    BuildPassed   BIT           NOT NULL            -- the champion's gate result = build result
);
GO

IF OBJECT_ID('dbo.DimDataset') IS NULL
CREATE TABLE dbo.DimDataset (
    DatasetKey  INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    DatasetName NVARCHAR(50) NOT NULL UNIQUE
);
GO

IF OBJECT_ID('dbo.DimModel') IS NULL
CREATE TABLE dbo.DimModel (
    ModelKey    INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    ModelName   NVARCHAR(100) NOT NULL UNIQUE,
    Vendor      NVARCHAR(50)  NOT NULL,
    Description NVARCHAR(300) NOT NULL
);
GO

IF OBJECT_ID('dbo.DimGroup') IS NULL
CREATE TABLE dbo.DimGroup (
    GroupKey      INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    AttributeName NVARCHAR(50)  NOT NULL,          -- race, ethnicity, sex, age_group
    GroupName     NVARCHAR(100) NOT NULL,          -- e.g. Black or African American; 'all' for gaps
    CONSTRAINT UQ_DimGroup UNIQUE (AttributeName, GroupName)
);
GO

IF OBJECT_ID('dbo.DimMetric') IS NULL
CREATE TABLE dbo.DimMetric (
    MetricKey   INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    MetricName  NVARCHAR(60) NOT NULL UNIQUE,
    IsGated     BIT          NOT NULL,
    MetricLevel NVARCHAR(10) NOT NULL              -- 'group' (a rate) or 'gap' (between groups)
);
GO

IF OBJECT_ID('dbo.FactFairnessAudit') IS NULL
CREATE TABLE dbo.FactFairnessAudit (
    AuditKey         BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    RunKey           INT   NOT NULL REFERENCES dbo.DimRun(RunKey),
    DateKey          INT   NOT NULL REFERENCES dbo.DimDate(DateKey),
    DatasetKey       INT   NOT NULL REFERENCES dbo.DimDataset(DatasetKey),
    ModelKey         INT   NOT NULL REFERENCES dbo.DimModel(ModelKey),
    GroupKey         INT   NOT NULL REFERENCES dbo.DimGroup(GroupKey),
    MetricKey        INT   NOT NULL REFERENCES dbo.DimMetric(MetricKey),
    MetricValue      FLOAT NULL,      -- NULL when a gap could not be computed (too few groups)
    CiLower          FLOAT NULL,      -- 95% confidence interval
    CiUpper          FLOAT NULL,
    Threshold        FLOAT NULL,      -- gated metrics only
    Passed           BIT   NULL,      -- gated metrics only; NULL = not evaluated
    GateEligible     BIT   NULL,      -- group-level rows: was the group large enough to gate?
    GroupSize        INT   NOT NULL,
    RowsMissingAttr  INT   NOT NULL,  -- rows excluded because the attribute was not reported
    CONSTRAINT UQ_FactFairnessAudit UNIQUE (RunKey, DatasetKey, ModelKey, GroupKey, MetricKey)
);
GO

IF OBJECT_ID('dbo.FactModelEvaluation') IS NULL
CREATE TABLE dbo.FactModelEvaluation (
    EvaluationKey BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    RunKey        INT   NOT NULL REFERENCES dbo.DimRun(RunKey),
    DateKey       INT   NOT NULL REFERENCES dbo.DimDate(DateKey),
    DatasetKey    INT   NOT NULL REFERENCES dbo.DimDataset(DatasetKey),
    ModelKey      INT   NOT NULL REFERENCES dbo.DimModel(ModelKey),
    [Role]        NVARCHAR(20) NOT NULL,           -- champion / challenger
    TestRows      INT   NOT NULL,
    Accuracy      FLOAT NOT NULL,
    RocAuc        FLOAT NOT NULL,
    ApprovalRate  FLOAT NOT NULL,
    GatePassed    BIT   NOT NULL,
    GateFailures  INT   NOT NULL,
    Recommended   BIT   NOT NULL,                  -- champion/challenger decision
    CONSTRAINT UQ_FactModelEvaluation UNIQUE (RunKey, DatasetKey, ModelKey)
);
GO

/* Convenience views for Power BI (the star tables can also be imported directly). */

CREATE OR ALTER VIEW dbo.vw_GatedMetrics AS
SELECT r.RunId, r.RunStartedAt, d.FullDate, r.Branch, r.CommitSha, r.[Trigger],
       r.ChampionModel, r.BuildPassed,
       ds.DatasetName, m.ModelName, m.Vendor, f2.[Role],
       g.AttributeName, mt.MetricName,
       f.MetricValue, f.CiLower, f.CiUpper, f.Threshold, f.Passed
FROM dbo.FactFairnessAudit f
JOIN dbo.DimRun r      ON r.RunKey = f.RunKey
JOIN dbo.DimDate d     ON d.DateKey = f.DateKey
JOIN dbo.DimDataset ds ON ds.DatasetKey = f.DatasetKey
JOIN dbo.DimModel m    ON m.ModelKey = f.ModelKey
JOIN dbo.DimGroup g    ON g.GroupKey = f.GroupKey
JOIN dbo.DimMetric mt  ON mt.MetricKey = f.MetricKey
LEFT JOIN dbo.FactModelEvaluation f2
       ON f2.RunKey = f.RunKey AND f2.DatasetKey = f.DatasetKey AND f2.ModelKey = f.ModelKey
WHERE mt.IsGated = 1;
GO

CREATE OR ALTER VIEW dbo.vw_BuildSummary AS
SELECT r.RunId, r.RunStartedAt, d.FullDate, r.Branch, r.CommitSha, r.[Trigger],
       r.ChampionModel, r.BuildPassed, ds.DatasetName,
       m.ModelName, m.Vendor, e.[Role], e.Accuracy, e.RocAuc, e.ApprovalRate,
       e.GatePassed, e.GateFailures, e.Recommended
FROM dbo.FactModelEvaluation e
JOIN dbo.DimRun r      ON r.RunKey = e.RunKey
JOIN dbo.DimDate d     ON d.DateKey = e.DateKey
JOIN dbo.DimDataset ds ON ds.DatasetKey = e.DatasetKey
JOIN dbo.DimModel m    ON m.ModelKey = e.ModelKey;
GO
