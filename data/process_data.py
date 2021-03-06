import sys
import re
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''load data given filepath for each set of data

       params:
           message_filepath - a path the disaster message dataset
           categories_filepath - a path to the disaster categories dataset
       returns:
           df - a dataframe of the merged disaster messages and categories dataframes
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    return df
    


def clean_data(df):
    '''clean the disaster messages and categories dataframe

       params:
           df - a dataframe of the messages and categories dataframe
       returns:
           df (copy) - a copy of the cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: re.findall(r'[a-z_A-Z]+', x)[0]).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].str.replace('2', '1')
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
        
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df.copy(deep=True)
    


def save_data(df, database_filename):
    '''save the dataframe to a SQLite database

       params:
           df - a dataframe
           database_filename - a path to where you want df saved
    '''
    # Define database file name
    path = 'sqlite:///' + database_filename
    
    # create a SQLAlchemy engine to the database
    engine = create_engine(path)
    #table_name = database_filename.strip('.db')
    table_name = 'message_categories'
    # save the database
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    pass  


def main():
    '''the driver function
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
