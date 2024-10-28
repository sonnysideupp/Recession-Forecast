
import pandas as pd
import datetime as dt

def getRecessionSpan(df: pd.DataFrame) -> list[tuple[dt.datetime, dt.datetime]]:
    rec_spans: list[tuple[dt.datetime, dt.datetime]] = []
    started_date: tuple[dt.datetime, float] | None = None
    
    for index, row in df.iterrows():
        if row['Regime'] == 'Recession':
            if started_date is None:
                started_date = (row['Date'], index)
               
        elif row['Regime'] == 'Normal':
           
            if started_date is not None and df.iloc[index-1]['Regime'] == 'Recession':
                rec_spans.append((started_date[0], row['Date']))
                started_date = None


    return rec_spans
    