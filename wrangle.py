import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
import os
from env import host, username, password

def get_connection(db, user=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                bedroomcnt as bedrooms, bathroomcnt as bathrooms, garagecarcnt as garages, poolcnt as pools, calculatedfinishedsquarefeet as area,
                lotsizesquarefeet as lot_size, fips, regionidcounty as county, regionidcity as city, regionidzip as zip,
                yearbuilt, taxvaluedollarcnt as tax_value
                From predictions_2017 
                JOIN properties_2017 USING (parcelid)
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
                WHERE transactiondate Like "2017%%"
                AND propertylandusedesc IN('Single Family Residential','Inferred Single Family Residential');
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()

    
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = df.columns

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
    
    

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
  
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def prepare_zillow(df, target, col_list):
    # remove all outliers from dataset
    df = remove_outliers(df, 1.5, col_list)
    
    #remove nulls
    values = {'garages':2.0, 'pools':0}
    df = df.fillna(value=values)
    
    #cleanup and change yearbuilt
    # df['yearbuilt'] = df.yearbuilt.apply(check_decade)
    # df = df.rename(columns={'yearbuilt':'decade'})
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)
    
    # splitting data into train, validate, test
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    scaler = MinMaxScaler()
    
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns.values).set_index([X_train.index.values])
    X_validate = pd.DataFrame(scaler.transform(X_validate), columns = X_validate.columns.values).set_index([X_validate.index.values])
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns.values).set_index([X_test.index.values])
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def wrangle_zillow():

    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = prepare_zillow(get_zillow_data())

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test


def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def check_decade(n):
    if n < 1810:
        return 1800
    elif n < 1820:
        return 1810
    elif n < 1830:
        return 1820
    elif n < 1840:
        return 1830
    elif n < 1850:
        return 1840
    elif n < 1860:
        return 1850
    elif n < 1870:
        return 1860
    elif n < 1880:
        return 1870
    elif n < 1890:
        return 1880
    elif n < 1900:
        return 1890
    elif n < 1910:
        return 1900
    elif n < 1920:
        return 1910
    elif n < 1930:
        return 1920
    elif n < 1940:
        return 1930
    elif n < 1950:
        return 1940
    elif n < 1960:
        return 1950
    elif n < 1970:
        return 1960
    elif n < 1980:
        return 1970
    elif n < 1990:
        return 1980
    elif n < 2000:
        return 1990
    elif n < 2010:
        return 2000
    else:
        return 2010
    
def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    df = pd.read_csv('student_grades.csv')
    # Replace white space values with NaN values.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Drop all rows with NaN values.
    df = df.dropna()
    # Convert all columns to int64 data types.
    df = df.astype('int64')
    return df