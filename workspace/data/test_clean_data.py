import pandas as pd
from sqlalchemy import create_engine

def main():
    engine_read = create_engine('sqlite:///./message_categories.db')
    test_df = pd.read_sql('SELECT * FROM message_categories', engine_read)
    print(test_df.head())
    
if __name__ == '__main__':
    main()