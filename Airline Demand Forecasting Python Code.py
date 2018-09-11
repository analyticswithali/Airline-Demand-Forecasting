import pandas as pd
import numpy as np

# IS5201
# Group Project
# Author  : Russell Key, Ali Afsar, Matt Ensseln
# Date    : 3/15/2018
# Purpose : Develop airline forecasting model based upon departure date,
#           booking date and current bookings from training data. Apply
#           calculated coefficients for additive and multiplictive models
#           to validation data.
# Inputs  : airline_booking_trainingData.csv, airline_booking_validationData_revised.csv

def readCSV(file, dateColumns):

    ''' Read passed in csv file name and format column indicated
    by dateColumn as dates. Returns contents of file as dataframe'''

    df = pd.read_csv(file)
    for column in dateColumns:
        df[column] = pd.to_datetime(df[column], format='%m/%d/%Y')
        
    return(df)

def calculateDaysPrior(df, dateColumns):
    
    ''' Using columns indicated by dateColumn, calculate days prior.
    Also creates string day of week as column 'Departure_dow' from dataColumns.
    Returns original dataframe with additional columns. '''

    df['Days_Prior'] = (df[dateColumns[0]] - df[dateColumns[1]]).dt.days
    df['Departure_dow'] = df[dateColumns[0]].dt.weekday_name
    return(df)

def calculateDailyBookings(df, bookingColumn):

    ''' Based upon column in bookingColumn, calculate new bookings by
    subtracting current day's booking by yesterday. Also creates a 3 day rolling
    average of daily bookings as column 'Daily_Bookings_Average'.
    Returns original dataframe with additional columns. '''

    df['Daily_Bookings'] = df[bookingColumn] - df[bookingColumn].shift(1)
    df.loc[df.index[0], 'Daily_Bookings'] = 0
    # Set first row of new departure date set to 0
    df.loc[df.departure_date != df.departure_date.shift(1), 'Daily_Bookings'] = 0

    # Calculate 3 day rolling average to smooth daily bookings
    df['Daily_Bookings_Average'] = (df['Daily_Bookings'] + df['Daily_Bookings'].shift(1) + df['Daily_Bookings'].shift(-1)) / 3
    # Set first row of new departure date set to 0
    df['Daily_Bookings_Average'] = np.where(df['Days_Prior'].shift(1) == 0, 0, df['Daily_Bookings_Average'])

    return(df)

def calculateAverageDailyBookings(df, dailyBookingsColumn, daysPriorColumn):
    
    ''' Creates day of week dataframe that averages daily bookings using daysPriorColumn to
    group by and dailyBookingsColumn for actual values. Returns new dataframe. '''

    df_DOW = createDOWDataFrame(df, daysPriorColumn, dailyBookingsColumn)

    return(df_DOW)

def calculateAverageDailyBookingsRate(df, dailyBookingsColumn, daysPriorColumn):
    
    ''' Merges df dataframe with itself to line up cumulative bookings to create daily booking rate. 
    Creates day of week dataframe that averages daily bookings using daysPriorColumn to group by for actual values. 
    Returns new dataframe. '''

    df = pd.merge(df, pd.DataFrame(df[df['departure_date'] == df['booking_date']][['departure_date','cum_bookings']]), on='departure_date', suffixes=('', '_final'))
    df['Booking_Rate'] = df['cum_bookings'] / df['cum_bookings_final']
    
    df_DOW = createDOWDataFrame(df, daysPriorColumn, 'Booking_Rate')

    return(df_DOW)

def createDOWDataFrame(df, groupBy, dailyColumn):

    ''' Creates day of week dataframe using passed in groupBy column and
    dailyColumn to calculate averages. '''

    df_DOW = pd.DataFrame(index = range(0, 60), columns = ['All', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Setup dataframe to columns of interest only
    df = df[[groupBy, dailyColumn, 'Departure_dow']]
    df_DOW['All'] = df.groupby(groupBy).mean().astype(float)
       
    df_DOW['Monday'] = (df[df['Departure_dow'] == 'Monday'].groupby(groupBy)).mean().astype(float)
    df_DOW['Tuesday'] = (df[df['Departure_dow'] == 'Tuesday'].groupby(groupBy)).mean().astype(float)
    df_DOW['Wednesday'] = (df[df['Departure_dow'] == 'Wednesday'].groupby(groupBy)).mean().astype(float)
    df_DOW['Thursday'] = (df[df['Departure_dow'] == 'Thursday'].groupby(groupBy)).mean().astype(float)
    df_DOW['Friday'] = (df[df['Departure_dow'] == 'Friday'].groupby(groupBy)).mean().astype(float)
    df_DOW['Saturday'] = (df[df['Departure_dow'] == 'Saturday'].groupby(groupBy)).mean().astype(float)
    df_DOW['Sunday'] = (df[df['Departure_dow'] == 'Sunday'].groupby(groupBy)).mean().astype(float)

    return (df_DOW)

def airlineForecast(trainingDataFile, validationDataFile) :

    ''' Calculate airline forecast using trainingDataFile to train model
    and validationDataFile to test and calculate MASE based upon existing
    naive forecast in validationDataFile. '''

    # Identify expected date column in training and validation data
    dateColumns = ['departure_date', 'booking_date']

    # Load Training data
    trainingDataFrame = readCSV(trainingDataFile, dateColumns)
    # Calculate and add 'Days_Prior' column
    trainingDataFrame = calculateDaysPrior(trainingDataFrame, dateColumns)

    # Calculate average daily for additive model
    trainingDataFrame = calculateDailyBookings(trainingDataFrame, 'cum_bookings')
    # Setup average daily bookings by day of week.
    averageDailyBookings = calculateAverageDailyBookings(trainingDataFrame, 'Daily_Bookings_Average', 'Days_Prior')

    # Calculate coefficients for multiplicative model
    averageDailyBookingsRate = calculateAverageDailyBookingsRate(trainingDataFrame, 'Daily_Bookings', 'Days_Prior')

    # Load Validation data
    validationDataFrame = readCSV(validationDataFile, dateColumns)
    # Calculate and add 'Days_Prior' column
    validationDataFrame = calculateDaysPrior(validationDataFrame, dateColumns)

    # Remove day prior of 0 since we do not forcast these days
    validationDataFrame.drop(validationDataFrame[(validationDataFrame['departure_date'] == validationDataFrame['booking_date'])].index, inplace=True)
    
    # Calculate Additive forecast and forecast error.  Using current Days_Prior and day of week, a look up is done in averageDailyBookings to loop from current Days_Prior
    # departure day to add each incremental forecast amount to arrive at final booking forecast
    validationDataFrame['add_forecast'] = validationDataFrame['cum_bookings'] + validationDataFrame[validationDataFrame.Days_Prior > 0][['Days_Prior','Departure_dow']].apply(lambda row: sum([averageDailyBookings.loc[i, row[1]] for i in range(row[0] - 1, -1, -1)]), axis = 1)    
    validationDataFrame['add_forecast_error'] = abs(validationDataFrame['final_demand'] - validationDataFrame['add_forecast'])
    
    # Calculate Multiplicative forecast and forecast error. Using current Days_Prior, a look up is done in averageDailyBookingsRate to divide the current cum_bookings 
    # by forecast ratio to arrive at final booking forecast
    validationDataFrame['mult_forecast'] = validationDataFrame['cum_bookings'] / validationDataFrame[validationDataFrame.Days_Prior > 0][['Days_Prior']].apply(lambda row: averageDailyBookingsRate.loc[row[0], 'All'], axis = 1)
    validationDataFrame['mult_forecast_error'] = abs(validationDataFrame['final_demand'] - validationDataFrame['mult_forecast'])
    
    # Calculate Naive forecast error
    validationDataFrame['naive_forecast_error'] = abs(validationDataFrame['final_demand'] - validationDataFrame['naive_forecast'])

    # Calculate MASE for additive and multiplicative forecast compared to naive forecast
    MASE_add = round(validationDataFrame['add_forecast_error'].sum() * 100 / validationDataFrame['naive_forecast_error'].sum(), 1)
    MASE_mult = round(validationDataFrame['mult_forecast_error'].sum() * 100 / validationDataFrame['naive_forecast_error'].sum(), 1)       

    dfAdditive = pd.DataFrame(validationDataFrame[['departure_date','booking_date','add_forecast']])
    dfMultiplicative = pd.DataFrame(validationDataFrame[['departure_date','booking_date','mult_forecast']])
    return ([dfAdditive, str(MASE_add) + '%', dfMultiplicative, str(MASE_mult) + '%'])


def main():
    mase = airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData_revised.csv')
    print mase

main()
