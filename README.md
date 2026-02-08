# My Project

Data description
This page describes the files provided for the competition, what you should expect the formats to be, and what you are predicting.

At a high level, you will forecast daily unit sales (units_sold) for each store–product–date in the test period (the final month of the dataset). The training and test targets are provided in wide format (dates as columns). Your submission is in long format (one row per store–product–date).

Note: The organizer scoring file solution.csv exists internally for evaluation but is not provided to participants.

What am I predicting?
You are predicting:

units_sold: the number of units sold for a given store_id and product_id on a given date.
The test period corresponds to the last month of dates in the dataset. All earlier dates are training.

What files do I need?
Minimum required files:

train.csv to train your model
test.csv to generate forecasts for the required dates
sample_submission.csv to format your predictions correctly
Optional (recommended) files for improving forecasts:

prices.csv
discounts.csv
promotions.csv
competitor_pricing.csv
weather.csv
products.csv
stores.csv
File formats and contents
Core forecasting files (wide format)
Dates are column headers formatted as YYYY-MM-DD.

train.csv
Training target values in wide format.

Columns:
store_id (string)
product_id (string)
one column per date (e.g., 2023-01-01, 2023-01-02, …) containing historical units_sold
test.csv
Forecast targets to be predicted, in the same wide format.

Columns:
store_id (string)
product_id (string)
one column per test date (the final month) containing missing/empty values
Important: Your model must output predictions for all test date columns for each store–product row in test.csv.

Submission file (long format)
sample_submission.csv
The submission template. One row per store–product–date.

Columns:
id (string): unique key in the form store_id_product_id_YYYY-MM-DD
units_sold (numeric): your prediction
Example format (illustrative)
