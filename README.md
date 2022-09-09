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
> I believe that area and rooms alone won't be the best predictor after constructing my MVP.  With the inclusion of features that help outline location based properties of each house we will be able to better predict an accurate property tax assessed values of a sinle family property.  The questions I need answered are which location based features are best to use and if any other features have correlation with location thus influencing how location influences tax value. 

## Executive Summary - Key Findings and Recommendations
> 1. Utilizing the following features outlined in X_train, X_validate, and X_test I was able to narrow down my best model for predicting tax value of properties given using a 2nd degree Polynomial Regression model.

> 2. Many features help to predict tax value with the top features being area, fips, county, and zip which matched my original thoughts that location based features would be the strongest predictors within my model. 

> 3. My recommendations are that we find a better way of predicting tax value because although my best model outperforms baseline predictions, there is still a lot of room for error.  Maybe a different methodology can be used or a better sample of data can be captured since much of the given data had erroneous outliers or missing values.

## Project Plan

### Planning Phase

> - Created a README.md file
> - Imported all of my tools and files needed to properly conduct the project

### Acquire Phase

> - Utilized my wrangle file to pull zillow data from a Codeup database

### Prepare Phase

> - Utilized my wranlge file to clean up my zillow dataset
> - Split the overall dataset into my train, validate, and test datasets
> - Utilized my wrangle file to create a separate copy of scaled datasets for future model creation.

### Explore Phase

> - Created visualizations via pairplots and heatmaps to see the correlation between each feature within my train dataset
> - Created a for loop for my hypothesis testing to see the numbers behind correlation between each independent variable and the target variable using Spearmen Test.
> - Asked further questions about that data such as how each location based feature affected tax value (in which counties are the highest tax value, etc.).

### Model Phase

> - Set up the baseline prediction for future models to base their success on
> - Trained multiple models for each type of Regression technique (Ordinary Least Squares, LASSO + LARS, Generalized Linear Model, and Polynomial Regression Model)
> - Validated all models to narrow down my selection to the best performing model.
> - Chose the MVP of all created models and used the test data set to ensure the best model worke entirely to expectations

### Deliver Phase

> - Prepped my final notebook with a clean presentation to best present my findings and process to other Data Scientists and stakeholders alike.

## How To Reproduce My Project

> 1. Read this README.md
> 2. Download the wrangle.py, evaluate.py, and final_report.ipynb files into your directory along with your own env file that contains your user, password, and host variables
> 3. Run my final_report.ipynb notebook
> 4. Congratulations! You can predict home tax value!