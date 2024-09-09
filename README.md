Car Price Prediction
Overview
This project involves building a predictive model to estimate car prices based on various features such as car name, company, year of manufacture, kilometers driven, and fuel type. The dataset used in this project contains information about used cars listed for sale, and the objective is to clean the data, analyze it, and build a machine learning model for predicting car prices.

Dataset
The dataset quikr_car.csv includes the following columns:

name: The name of the car.
company: The manufacturer of the car.
year: The year of manufacture.
Price: The listed price of the car.
kms_driven: The number of kilometers the car has been driven.
fuel_type: The type of fuel the car uses (Petrol, Diesel, etc.).
Data Cleaning
The data cleaning steps performed are as follows:

Year Column:

Removed non-numeric values.
Converted the year column from object to integer.
Price Column:

Removed rows with 'Ask For Price'.
Removed commas and converted the column to integer.
Kms Driven Column:

Extracted numeric values and removed 'kms' from the string.
Converted the column to integer.
Fuel Type Column:

Removed rows with missing fuel type values.
Name and Company Columns:

Standardized car names by keeping only the first three words.
Cleaned company names to remove irrelevant values.
Final Dataset:

Saved the cleaned dataset as Cleaned_Car_data.csv.
Data Analysis
Various relationships were analyzed:

Company vs. Price: Visualized using a box plot to see how prices vary across different car companies.
Year vs. Price: Visualized using a swarm plot to understand how prices change with the year of manufacture.
Kilometers Driven vs. Price: Examined using a scatter plot to assess the correlation between kilometers driven and car price.
Fuel Type vs. Price: Analyzed using a box plot to compare prices based on fuel type.
Company, Year, and Fuel Type: Explored with a mixed plot to understand the combined effects of these features on car prices.
Model Building
Feature Selection:

Features: name, company, year, kms_driven, fuel_type
Target: Price
Data Preprocessing:

Used OneHotEncoder for categorical features and ColumnTransformer for preprocessing.
Model:

Applied Linear Regression using sklearn.
Created a pipeline with preprocessing and model fitting.
Evaluation:

The model was evaluated using R² score, with the best model achieving an R² score of approximately 0.92.
Prediction:

Example prediction: For a 'Maruti Suzuki Swift', Maruti company, year 2019, 100 kilometers driven, and Petrol fuel type, the predicted price is approximately ₹416,109.
Model Saving:

The final model was saved as LinearRegressionModel.pkl using pickle.
How to Use
Install Dependencies: Make sure you have the following libraries installed:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn pickle-mixin
Load the Model:

python
Copy code
import pickle
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
Make Predictions:

python
Copy code
import pandas as pd
# Define your input data
input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                           data=[['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']])
# Predict
predicted_price = model.predict(input_data)
print(predicted_price)
Notes
The model performs well on the test data but always consider retraining with new data or more advanced models for better accuracy.
The dataset and the model are simplified for demonstration purposes. Further feature engineering and model tuning can improve predictions.
License
This project is licensed under the MIT License. See the LICENSE file for details.
