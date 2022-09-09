# Regression Project Summary

## Project Goals

> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways.
> - Create modules (acquire.py, prepare.py) that make my process repeateable and my report (notebook) easier to read and follow.
> - Ask exploratory questions of my data that will help me understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.
> - Construct a model to predict assessed home value for single family properties using regression techniques.
> - Make recommendations to a data science team about how to improve predictions.
> - Refine my work into a report, in the form of a jupyter notebook, that I will walk through in a 5 minute presentation to a group of collegues and managers about my goals, the work I did, why, what I found, my methodologies, and my conclusions.
> - Be prepared to answer panel questions about my code, process, findings and key takeaways, and model.

## Project Description

### Business Goals

> - Construct a ML Regression model that predicts property tax assessed values ('taxvaluedollarcnt') of **Single Family Properties** using attributes of the properties.
> - Find the key drivers of property value for single family properties.
> - Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
> - Make recommendations on what works or doesn't work in predicting these homes' values.

### Deliverables

> - **Readme (.md)**
> - **Acquire & Prepare Modeules (Wrangle) (.py)**
> - **Final Report (.ipynb)**
> - 5 min Live Presentation

## Data Dictionary

|Target|Definition
|:-------|:----------|
|tax_value|The total tax assessed value of the property|

|Feature|Definition|
|:-------|:----------|
|bedrooms   |Number of bedrooms in home|
|bathrooms  |Number of bathrooms in home including fractional bathrooms|
|garages    |Total number of garages on the lot including an attached garage|
|pools      |Number of pools on the lot|
|area       |Calculated total finished living area of the home| 
|lot_size   |Area of the lot in square feet|
|fips       |Federal Information Processing Standard code|
|county     |County in which the property is located|
|city       |City in which the property is located|
|zip        |Zip code in which the property is located|
|yearbuilt  |The Year the principal residence was built|

## Initial Hypotheses
> I believe that monthly charges and total charges will play a large part in whether a customer will churn or not.  The questions I need answered are how the other features may influence how much a customer may pay at any given time and if specific demographics affect the price threshold for customers (such as age or gender).

## Executive Summary - Key Findings and Recommendations
> 1. Utilizing the following features outlined in X_train, X_validate, and X_test I was able to narrow down my best model for predicting churn at Telco using a random forest model with a max depth of 5 and an 80% accuracy rate.

> 2. Many features help to predict churn with some of the bigger predictors being age (senior_citizen), how much a customer is paying (monthly_charges, total_charges, contract_type, payment_type) and what's affecting how much they pay (internet_service_type), and finally whether customers have adequate support with their plan (tech_support)

> 3. My recommendations are that customers are encouraged to sign up with a one or two year contract by maybe offering technical support included.  We should also focus on a younger crowd who has an established family because they tend to churn at a lesser rate.

## Project Plan

### Planning Phase

> - Created a README.md file
> - Imported all of my tools and files needed to properly conduct the project

### Acquire Phase

> - Utilized my acquire file to pull telco data from a Codeup database

### Prepare Phase

> - Utilized my prepare file to clean up my telco dataset
> - Split the overall dataset into my train, validate, and test datasets
> - Utilized my explore file to create a list of numerical datatype features and a list of categorical datatype features for future exploration

### Explore Phase

> - Created a for loop that used my explore file to create a visualization for every categorical feature.  The explore file also allowed for hypothesis testing each feature to ensure a relationship between the feature and the target (churn).
> - Asked further questions about that data such as how monthly charges effects churn rate and the relationship between other features and monthly charges.  Also explored how age played a part in churn rate and if any other feature could assist in predicting churn for older customers.

### Model Phase

> - Set up the baseline accuracy for future models to base their success on
> - Trained multiple models for each type of classification technique (Decision Tree, Logistic Regression, Random Forest, KNN)
> - Validated all models to narrow down my selection to the best performing model.
> - Chose the MVP of all created models and used the test data set to ensure the best model worke entirely to expectations

### Deliver Phase

> - Prepped my final notebook with a clean presentation to best present my findings and process to other Data Scientists and stakeholders alike.
> - Ensured that a prediction csv was generated for future proof of my working model

## How To Reproduce My Project

> 1. Read this README.md
> 2. Download the acquire.py, prepare.py, explore.py and final_report.ipynb files into your directory along with your own env file that contains your user, password, and host variables
> 3. Run my final_report.ipynb notebook
> 4. Congratulations! You can predict future churn at Telco!