# Advanced Predictive Analytics --- Experiment 06

## Time-Series Analysis and Forecasting of Reported Crime Incidents

This project analyzes reported crime incidents as a **weekly time
series** and applies **Naive, AutoReg (AR), and ARIMA forecasting
models** to predict future incident counts by police district.

> **Important data note:** The report states that the official Chicago
> Crimes --- 2001 to Present portal could not be accessed from the
> execution environment. Therefore, the experiment was executed using a
> **synthetic, schema-matched incident-level dataset** generated
> locally. The results are illustrative of the forecasting workflow and
> should not be interpreted as real crime-level estimates.

## 1. Objectives

-   Convert incident-level records into a regular weekly crime-count
    time series.
-   Examine trend, seasonality, autocorrelation, and location
    differences.
-   Use chronological train/test separation without random shuffling.
-   Assess stationarity using visual inspection and the Augmented
    Dickey-Fuller (ADF) test.
-   Use ACF/PACF diagnostics to motivate AR/ARIMA model orders.
-   Compare a naive baseline, AR model, and ARIMA model.
-   Evaluate forecasts using MAE and RMSE.
-   Check residual autocorrelation using the Ljung-Box test.
-   Replicate the workflow on a second district.
-   Compare an advanced SARIMA model with the ARIMA baseline.
-   Save modelling and evaluation artifacts for reproducibility.

## 2. Dataset

### Intended source

**City of Chicago --- Crimes --- 2001 to Present** open-data portal.

### Dataset used for this execution

A synthetic dataset designed to match the documented incident schema was
used because the official portal was not reachable in the execution
environment.

The schema includes:

-   `Date` --- incident occurrence date/time
-   `District` --- police district
-   `Primary Type` --- incident/offense category
-   `Case Number` --- incident identifier
-   `Latitude` --- latitude
-   `Longitude` --- longitude

### Data characteristics

-   Time span: **2020-01-06 to 2026-01-05**
-   Aggregation: **weekly (W-MON)**
-   Rows before cleaning: **15,346**
-   Rows after cleaning: **15,346**
-   Weekly periods: **314**
-   Primary analysis location: **District 1**
-   Replication location: **District 7**

## 3. Data Preprocessing

The preprocessing pipeline performs:

1.  Parse the `Date` field into datetime.
2.  Remove rows with unparseable dates.
3.  Remove exact duplicate `Case Number` values.
4.  Clean and validate the `District` field.
5.  Validate latitude and longitude ranges.
6.  Sort records chronologically.
7.  Audit missing values in the key fields.
8.  Aggregate incidents into weekly counts.
9.  Reindex the time series to regular weekly intervals.
10. Split the series chronologically into training data and a locked
    **12-week test period**.

Identifiers are used only for deduplication/audit purposes and are not
used as predictive features.

## 4. Methodology

### Forecasting workflow

``` text
Incident-level data
        ↓
Data cleaning & validation
        ↓
Select police district
        ↓
Weekly aggregation
        ↓
Exploratory time-series analysis
        ↓
ADF stationarity test
        ↓
ACF / PACF diagnostics
        ↓
Chronological train/test split
        ↓
Naive baseline + AR + ARIMA
        ↓
12-week future forecast
        ↓
MAE / RMSE comparison
        ↓
Residual diagnostics
        ↓
Rolling-origin validation
        ↓
Second-district replication
        ↓
SARIMA extension
```

### Models

#### Naive baseline

The naive model uses persistence as the benchmark for evaluating whether
the statistical models provide useful improvement.

#### AutoReg AR(4)

An autoregressive model with four previous observations as predictors.

#### ARIMA(1,1,1)

The selected ARIMA specification was:

``` text
ARIMA(p,d,q) = ARIMA(1,1,1)
```

The candidate models were ranked using training AIC/BIC, and
ARIMA(1,1,1) had the lowest training AIC.

#### SARIMA extension

An advanced seasonal comparison was performed using:

``` text
SARIMA(1,1,1) × (1,0,1,52)
```

This was compared against the non-seasonal ARIMA baseline.

## 5. Stationarity and Diagnostics

The Augmented Dickey-Fuller test on the training portion produced:

-   ADF statistic: **-5.6934**
-   p-value: **0.0000**

The report interprets this as evidence consistent with stationarity,
meaning only mild differencing was expected to be required.

ACF and PACF plots were calculated using training data only to avoid
leakage when selecting model orders.

## 6. Results

### District 1 --- locked 12-week test period

  Model              Test MAE    Test RMSE
  -------------- ------------ ------------
  Naive                2.0000       2.6771
  AR(4)                2.5034       2.9824
  ARIMA(1,1,1)     **1.9729**   **2.6580**

The **ARIMA(1,1,1)** model achieved the lowest MAE and RMSE among the
three core models.

The ARIMA model's Ljung-Box p-value was **0.5456**, providing no strong
evidence of remaining autocorrelation at the tested lag.

The empirical 95% prediction interval contained all 12 held-out
observations in this particular test period. The report notes that this
single-holdout coverage result is illustrative and is not a guarantee of
future coverage.

## 7. Rolling-Origin Validation

An 18-fold rolling-origin backtest was performed on the
training/validation history using:

-   Forecast horizon: 8 weeks
-   Step: 8 weeks

  Statistic     AR MAE   AR RMSE   ARIMA MAE   ARIMA RMSE
  ----------- -------- --------- ----------- ------------
  Mean          2.8779    3.4441      2.9945       3.5776
  Std. Dev.     0.8261    1.0949      1.0563       1.2104

The report notes that the two models showed broadly comparable mean
error and that forecast performance varied across folds. This
demonstrates why relying on only one train/test split can understate
forecast variability.

## 8. District 7 Replication

The same weekly aggregation, chronological split, and evaluation
protocol was replicated for District 7.

  District       Mean Weekly Count   Std. Dev. Best Model          MAE     RMSE
  ------------ ------------------- ----------- -------------- -------- --------
  District 1                  8.72        3.59 ARIMA(1,1,1)     1.9729   2.6580
  District 7                 15.78        4.42 ARIMA(1,1,1)     4.0601   5.3379

District 7 has a higher mean weekly incident count and consequently
larger absolute forecast errors. These are aggregate location-level
results and do not represent person-level conclusions.

## 9. SARIMA Comparison

The advanced seasonal model:

``` text
SARIMA(1,1,1) × (1,0,1,52)
```

was compared with ARIMA(1,1,1).

  Model                                 MAE
  ---------------------------- ------------
  ARIMA(1,1,1)                       1.9729
  SARIMA(1,1,1) × (1,0,1,52)     **1.9699**

The improvement was only **0.003 MAE**, so the report concludes that the
additional seasonal complexity does not clearly justify preferring
SARIMA over the simpler ARIMA model for this series.

## 10. Technologies and Libraries

The project uses Python and the following libraries:

-   **pandas** --- data loading, cleaning, aggregation, and time
    indexing
-   **numpy** --- numerical operations
-   **matplotlib** --- time-series and diagnostic plots
-   **statsmodels** --- AutoReg, ARIMA, SARIMAX, ADF, ACF/PACF, and
    Ljung-Box
-   **scikit-learn** --- MAE and RMSE calculation
-   **joblib** --- artifact/model persistence
-   **json** --- configuration and metadata

## 11. Reproducibility

The report indicates that the project artifacts include:

-   Notebook
-   CSV result files
-   Manifest/configuration metadata
-   Figures
-   Forecast prediction intervals
-   Residual diagnostics

The pipeline is designed so that it can be run against a real Chicago
crime-data extract by pointing `DATA_PATH` to the corresponding dataset,
while retaining the documented preprocessing and modelling workflow.

## 12. Important Limitations

-   The execution used a **synthetic stand-in dataset**, not live
    Chicago crime records.
-   Therefore, absolute forecast errors in this report are illustrative
    of the method.
-   The models forecast **reported/recorded incidents**, not the true
    underlying prevalence of crime.
-   Reporting behavior, enforcement intensity, legal definitions, and
    data-system changes may vary over time and across locations.
-   Location is treated as a label defining a separate time series, not
    as an ordinal numeric predictive feature.
-   The analysis is intended for academic/decision-support forecasting
    and not for person-level profiling or autonomous policing/resource
    allocation.
-   Negative forecasts can occur with ARIMA-type models for non-negative
    count data and should be treated as a documented model limitation
    rather than silently clipped.

## 13. Project Conclusion

For District 1, **ARIMA(1,1,1)** produced the best test performance
among the naive, AR, and ARIMA models, with a test MAE of **1.9729** and
RMSE of **2.6580**. Residual diagnostics showed no strong remaining
autocorrelation.

The second-district replication produced the same qualitative model
ranking, while the SARIMA extension provided only a negligible
improvement over ARIMA. Overall, the experiment demonstrates a
leakage-safe, chronological time-series forecasting workflow for
aggregate reported-incident counts.

## 14. Repository

GitHub repository:

**Advanced Predictive Analytics --- Submission 6**

https://github.com/avinashvit/Advanced_predictive_submission_6

## 15. Author

**Avinash A**\
Register No.: **23MID0344**\
Integrated M.Tech CS (Data Science)\
VIT Vellore\
Course: **MDI3003 --- Advanced Predictive Analytics**\
Experiment: **06**
