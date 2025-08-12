## Instructions

You will be provided with two parquet files.

----------------------------------------------------------------------------------------------------------------------------

`sedol_mapping_raw.parquet` contains the mapping between the **SEDOL** symbology, a code provided by the London Stock 
Exchange (LSEG) which uniquely defines a security, and an internal 'adia_id'.
Sedols are unique to each security, but the same security can have multiple Sedols over time.

This file contains the following columns:

- `sedol`: The SEDOL symbology for the security
- `adia_id`: The corresponding internal 'ADIA' symbology for the security
- `start_date`: The first date the mapping is valid
- `end_date`: The last date the mapping is valid

----------------------------------------------------------------------------------------------------------------------------

`sp500_constituents.parquet` contains the prices and weights for the constituents of the S&P 500 index from 2015 to 2020.
This is an index of the 500 largest companies listed on US stock exchanges.
Please note that due to some companies having multiple classes of shares, the same company may appear multiple times in the index
and the index contituent count may occasionally exceed 500.

This file contains the following columns:

- `date`: The date for the data
- `index_name`: The name of the index, in this case S&P 500
- `sedol`: The SEDOL symbology for the security on the date
- `security_name`: The name of the security, for example Apple Inc.
- `price`: The price of the security in USD on the date
- `weight`: The weight of the security in the index on the date as a fraction of 1. For example, a weight of 0.01 means the security makes up 1% of the index.



```python
import sys
!{sys.executable} -m pip install pyarrow

# If you would like to install additional packages that are not installed by default, you may do so here (as was done above)
```

    Requirement already satisfied: pyarrow in /opt/conda/lib/python3.9/site-packages (21.0.0)



```python
# Preamble
import pandas as pd
import matplotlib.pyplot as plt

# Data loading
sedol_mapping = pd.read_parquet('https://hr-projects-assets-prod.s3.amazonaws.com/90qc788c3ec/159b4f78789351d07c81d151908d1488/sedol_mapping_raw.parquet')
sp500_constituents = pd.read_parquet('https://hr-projects-assets-prod.s3.amazonaws.com/90qc788c3ec/40a77e6c2106f4f4d801b64530b675f0/sp500_constituents.parquet')
print(sedol_mapping.describe())
print(sp500_constituents.describe())
```

              sedol   adia_id           start_date             end_date
    count      3645      3645                 3645                 3645
    unique      700       683                  280                  325
    top     2542049  47A9AE8E  2015-01-01 00:00:00  2016-01-01 00:00:00
    freq          9         9                  907                  886
    first       NaN       NaN  2015-01-01 00:00:00  2015-01-21 00:00:00
    last        NaN       NaN  2019-12-09 00:00:00  2020-01-01 00:00:00
                   price         weight
    count  634721.000000  634721.000000
    mean      100.203677       0.001982
    std       136.325992       0.003374
    min         1.590000       0.000009
    25%        43.550000       0.000521
    50%        70.810000       0.000906
    75%       113.520000       0.001890
    max      3892.890000       0.045833


    /tmp/ipykernel_20902/707652608.py:8: FutureWarning: Treating datetime data as categorical rather than numeric in `.describe` is deprecated and will be removed in a future version of pandas. Specify `datetime_is_numeric=True` to silence this warning and adopt the future behavior now.
      print(sedol_mapping.describe())
    /tmp/ipykernel_20902/707652608.py:8: FutureWarning: Treating datetime data as categorical rather than numeric in `.describe` is deprecated and will be removed in a future version of pandas. Specify `datetime_is_numeric=True` to silence this warning and adopt the future behavior now.
      print(sedol_mapping.describe())


1. Load the data from the parquet files into the in-memory sqlite tables `sedol_mapping` and `sp500_constituents`.
Then, using SQL, map the constituents to the ADIA symbology and save the result to a new table `output`
This output table should have the columns: 
`date, index_name, adia_id, security_name, price, weight`
Please refer to the code block below for an example of the interface for sqlite

```
import sqlite3
conn = sqlite3.connect(':memory:') # Initialises an in-memory database
c = conn.cursor() # Initialises a cursor object
c.execute('CREATE TABLE table_one (column_one TEXT, column_two DATE)') # Creates a table
values = [('value_one', '2022-01-01'), ('value_two', '2022-01-02')] # List of tuples
c.executemany('INSERT INTO table_one VALUES (?, ?)', values) # Inserts multiple rows
conn.commit() # Commits the transaction
c.execute('SELECT * FROM table_one') # Selects all rows from the table
print(c.fel())
```

1.1 **Load into SQLite and Join**


```python
import warnings
import sqlite3

# Initialize in-memory database
conn = sqlite3.connect(':memory:')
c = conn.cursor()

# Load data into SQLite tables
sedol_mapping.to_sql('sedol_mapping', conn, if_exists='replace', index=False)
sp500_constituents.to_sql('sp500_constituents', conn, if_exists='replace', index=False)

# Create the output table by joining on SEDOL and large data range
c.execute('DROP TABLE IF EXISTS output;')
c.execute('''
CREATE TABLE output AS
SELECT s.date, s.index_name, m.adia_id, s.security_name, s.price, s.weight
FROM sp500_constituents s
JOIN sedol_mapping m
ON s.sedol = m.sedol
WHERE DATE(s.date) BETWEEN DATE(m.start_date) AND DATE(m.end_date)
''')

# Check the existence of the output table
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='output'")
print(c.fetchall())
```

    [('output',)]


1.2 **Preview joined table**


```python
# Verify the output table
c.execute('SELECT * FROM output LIMIT 15')
print("Sample rows from output table :")
print(c.fetchall())
```

    Sample rows from output table :
    [('2019-06-10 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 70.29, 0.000928120405975081), ('2018-04-26 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 66.37, 0.000942313265273382), ('2015-08-31 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 36.31, 0.00069502350366216), ('2016-08-30 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 47.14, 0.000818989218361911), ('2019-04-26 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 77.42, 0.00100357675254813), ('2016-12-09 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 46.3, 0.000772144732717085), ('2015-07-24 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 39.31, 0.000713237949557318), ('2019-01-07 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 66.85, 0.000995749549773795), ('2015-10-26 00:00:00', 'S&P 500', '9CDDCF1F', 'AGILENT TECHNOLOGIES INC', 36.83, 0.00066845035478791), ('2015-01-22 00:00:00', 'S&P 500', '1CA0D044', 'ALCOA INC', 16.04, 0.00103418785561383), ('2016-07-01 00:00:00', 'S&P 500', '1CA0D044', 'ALCOA INC', 9.55, 0.000688780783839057), ('2017-11-08 00:00:00', 'S&P 500', '04071152', 'AMERICAN AIRLINES GROUP INC.', 46.37, 0.000904460992353169), ('2019-11-18 00:00:00', 'S&P 500', '04071152', 'AMERICAN AIRLINES GROUP INC.', 28.6, 0.000437196266528007), ('2016-05-09 00:00:00', 'S&P 500', '04071152', 'AMERICAN AIRLINES GROUP INC.', 32.94, 0.00110449936053194), ('2016-06-15 00:00:00', 'S&P 500', '04071152', 'AMERICAN AIRLINES GROUP INC.', 30.49, 0.00102026661273578)]



```python
# Check row count
c.execute('SELECT COUNT(*) FROM output')
print(f'Total rows in output table : {c.fetchall()[0]}')
```

    Total rows in output table : (696664,)


2. The three years 2015 to the start of 2018, have a single distinct problem each year within the raw mapping file `sedol_mapping_raw.parquet`. Identify and briefly explain the data problem for each year, and correct them within the raw file using Python. Save the corrected file as `sedol_mapping_corrected` and rerun the pipeline using the corrected file. Have the errors been fixed? What might be a straightforward SQL query to check?


```python
# Quick verification of sedol_mapping values
# sedol_mapping.isna().sum()
# sedol_mapping.adia_id.unique()
# sedol_mapping.sedol.unique()
```

2.1. **Data Quality checks (Before fix)**


```python
# Year masks
def active_in_year(df, year):
    # mapping active at any time in that calendar year
    start = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp(f'{year}-12-31')
    return (df['start_date'] <= end) & (df['end_date'] >= start)

# No issues found with
def check_invalid(df, year):
    ss = df[active_in_year(df, year)]
    return int((ss['end_date'] < ss['start_date']).sum())

# issues found with
def check_bad_adia_ids(df, year):
    ss = df[active_in_year(df, year)]
    bad_values = {'None', 'Null', '**', '', 'A', 'placeholder', '?', '12321', '*', '-', '0', 'test', 'UNKNOWN', '#', '\\n', "''", 'z', '??', 'example', '94835849684392', 'x', 'AKSBFHDIWUEJ', 'X', 'prk#((83', 'owe283DJ', 'mD3hfs83', '*#(@8473', 'dusjwied', 'nnnnnnnn', 'Mdk83Dis'}
    bad_mask = ss['adia_id'].astype(str).str.strip().isin(bad_values)
    return bad_mask.sum()

    
# Issues found with
def check_dupes(df, year):
    ss = df[active_in_year(df, year)]
    return int (ss.duplicated(subset=['sedol','adia_id','start_date','end_date']).sum())

# No issues found with
def check_missing_data(df, year):
    ss = df[(df['start_date'].dt.year <= year) & (df['end_date'].dt.year >= year)]
    return int(ss['adia_id'].isna().sum())

# No issues found with
def check_gaps(df, year):
    ss = df[(df['start_date'].dt.year <= year) & (df['end_date'].dt.year >= year)].copy()
    gaps = 0
    for sed, group in ss.groupby('sedol'):
        group = group.sort_values('start_date')
        prev_end = None
        for _, row in group.iterrows():
            if prev_end and (row['start_date'] > prev_end + pd.Timedelta(days=1)):
                gaps += 1
                prev_end = row['end_date']
    return gaps

# Issues found with
def check_conflicting_ids(df, year):
    ss = df[(df['start_date'].dt.year <= year) & (df['end_date'].dt.year >= year)].copy()
    conflicts = 0
    for sed, group in ss.groupby('sedol'):
        group = group.sort_values('start_date')
        for i in range(len(group) - 1):
            if group.iloc[i]['end_date'] >= group.iloc[i+1]['start_date'] and group.iloc[i]['adia_id'] != group.iloc[i+1]['adia_id']:
                conflicts += 1
    return conflicts
```


```python
print("Checking 2015-2017 mapping for invalid ranges, duplicates, missing adia_id, gaps and conflicting IDs...")
```

    Checking 2015-2017 mapping for invalid ranges, duplicates, missing adia_id, gaps and conflicting IDs...



```python
years = [2015, 2016, 2017]

# Check issues in sedol_mapping
results = []
for y in years:
    results.append({
        "Year": y,
        "Total": active_in_year(sedol_mapping, y).sum(),
        "Bad_adia_ids": check_bad_adia_ids(sedol_mapping, y),
        "Invalid_date_ranges": check_invalid(sedol_mapping, y),
        "Duplicates": check_dupes(sedol_mapping, y),
        "Missing_adia_id": check_missing_data(sedol_mapping, y),
        "Coverage_Gaps": check_gaps(sedol_mapping, y),
        "Conflicting_IDs": check_conflicting_ids(sedol_mapping, y)
    })
df_checks_before = pd.DataFrame(results)
df_checks_before
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Total</th>
      <th>Bad_adia_ids</th>
      <th>Invalid_date_ranges</th>
      <th>Duplicates</th>
      <th>Missing_adia_id</th>
      <th>Coverage_Gaps</th>
      <th>Conflicting_IDs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>930</td>
      <td>0</td>
      <td>0</td>
      <td>302</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>1571</td>
      <td>65</td>
      <td>0</td>
      <td>289</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1431</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
if df_checks_before[['Bad_adia_ids', 'Invalid_date_ranges','Duplicates','Missing_adia_id','Coverage_Gaps','Conflicting_IDs']].sum().sum() == 0:
    print("No issues remain.")
else:
    print("Issues found!")
```

    Issues found!



```python
print("""Checked 2015-2017 mapping:
2015 -> 302 duplicates
2016 -> 289 duplicates and 65 conflicting IDs and 65 bad_adia_ids
2017 -> no issues
""")
```

    Checked 2015-2017 mapping:
    2015 -> 302 duplicates
    2016 -> 289 duplicates and 65 conflicting IDs and 65 bad_adia_ids
    2017 -> no issues
    


2.2. **Data Cleaning and Corrections**


```python
def fix_conflicting_overlaps(df):
    df = df.sort_values(['sedol', 'start_date', 'end_date']).copy()

    def _resolve_conflicts(group):
        rows = group.copy()
        i = 0
        while i < len(rows) - 1:
            cur_end = rows.iloc[i]['end_date']
            next_start = rows.iloc[i+1]['start_date']
            cur_id = rows.iloc[i]['adia_id']
            next_id = rows.iloc[i+1]['adia_id']

            # conflict: overlap and different IDs
            if cur_end >= next_start and cur_id != next_id:
                # trim the earlier mapping
                rows.iat[i, rows.columns.get_loc('end_date')] = next_start - pd.Timedelta(days=1)
            i += 1
        return rows
    return df.groupby('sedol', group_keys=False).apply(_resolve_conflicts).reset_index(drop=True)

def fix_exact_duplicates(df):
    return df.drop_duplicates(subset=['sedol','adia_id','start_date','end_date']).reset_index(drop=True)

def fix_bad_adia_ids(df):
    bad_values = {'None', 'Null', '**', '', 'A', 'placeholder', '?', '12321', '*', '-', '0', 'test', 'UNKNOWN', '#', '\\n', "''", 'z', '??', 'example', '94835849684392', 'x', 'AKSBFHDIWUEJ', 'X', 'prk#((83', 'owe283DJ', 'mD3hfs83', '*#(@8473', 'dusjwied', 'nnnnnnnn', 'Mdk83Dis'}
    df = df.copy()
    df.loc[df['adia_id'].isin(bad_values), 'adia_id'] = pd.NA
    return df.dropna().reset_index(drop=True)
```


```python
# Fix conflicts first, then duplicates
corrected = fix_conflicting_overlaps(sedol_mapping)
corrected = fix_exact_duplicates(corrected)
corrected = fix_bad_adia_ids(corrected)

# Save corrected file
corrected.to_parquet("sedol_mapping_corrected.parquet")
print("Corrected mapping saved.")
```

    Corrected mapping saved.


2.3. **Data Quality checks (After fix)**


```python
years = [2015, 2016, 2017]

# check if issues were corrected
results = []
for y in years:
    results.append({
        "Year": y,
        "Total": active_in_year(corrected, y).sum(),
        "Bad_adia_ids": check_bad_adia_ids(corrected, y),
        "Invalid_date_ranges": check_invalid(corrected, y),
        "Duplicates": check_dupes(corrected, y),
        "Missing_adia_id": check_missing_data(corrected, y),
        "Coverage_Gaps": check_gaps(corrected, y),
        "Conflicting_IDs": check_conflicting_ids(corrected, y)
    })
df_checks_after = pd.DataFrame(results)
df_checks_after
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Total</th>
      <th>Bad_adia_ids</th>
      <th>Invalid_date_ranges</th>
      <th>Duplicates</th>
      <th>Missing_adia_id</th>
      <th>Coverage_Gaps</th>
      <th>Conflicting_IDs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>628</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>1217</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1366</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
if df_checks_after[['Bad_adia_ids', 'Invalid_date_ranges','Duplicates','Missing_adia_id','Coverage_Gaps','Conflicting_IDs']].sum().sum() == 0:
    print("No issues remain.")
else:
    print("Issues found!")
```

    No issues remain.


2.4. **Re-run join with corrected mapping**


```python
# Push corrected mapping into SQLite
corrected.to_sql('sedol_mapping', conn, if_exists='replace', index=False)

# Drop old output & re-join
c.execute('DROP TABLE IF EXISTS output_corrected;')
c.execute('''
CREATE TABLE output_corrected AS
SELECT s.date, s.index_name, m.adia_id, s.security_name, s.price, s.weight
FROM sp500_constituents s
JOIN sedol_mapping m
ON s.sedol = m.sedol
WHERE DATE(s.date) BETWEEN DATE(m.start_date) AND DATE(m.end_date)
''')
```




    <sqlite3.Cursor at 0x7f481857b0a0>



2.5. **SQL queries to verify fixes**


```python
# Check for duplicates in mapping
c.execute('''
SELECT COUNT(*) AS duplicates_rows
FROM (
SELECT sedol, adia_id, start_date, end_date, COUNT(*) AS cnt
FROM sedol_mapping
GROUP BY sedol, adia_id, start_date, end_date
HAVING cnt > 1
) t;
''')
print(c.fetchall())

# Check for conflicting overlaps
c.execute('''
SELECT COUNT(*) AS conflicting_rows
FROM sedol_mapping AS a
JOIN sedol_mapping AS b
ON a.sedol = b.sedol
AND a.adia_id <> b.adia_id
AND DATE(a.start_date) <= DATE(b.end_date)
AND DATE(b.start_date) <= DATE(a.end_date)
''')
print(c.fetchall())
```

    [(0,)]
    [(0,)]


3. Please retrieve the two-year period between 2018 to the start of 2020 from the `output` table as a pandas DataFrame. Using the Python code below, please calculate the signal for each of the securities for the two-year period.

```
price_threshold = 20
result = result[result['price'] > price_threshold]
result['signal_score'] = result.groupby(['adia_id'])['weight'].transform(lambda x: (x - x.mean()) / x.std())
```

3.1. **Signal Calculation**


```python
result = pd.read_sql_query("SELECT * FROM output WHERE date >= '2018-01-01' AND date < '2020-01-01'", conn)

price_threshold = 20
result = result[result['price'] > price_threshold]
result['signal_score'] = result.groupby(['adia_id'])['weight'].transform(lambda x: (x - x.mean()) / x.std())
```

3.2. **Brief evaluation of robustness**


```python
# Sanity summary
print(f"""result dataset has:
                {len(result):,} rows 
                {result['adia_id'].nunique()} securities
                NaN signals: {int(result['signal_score'].isna().sum())}
                """)
```

    result dataset has:
                    241,989 rows 
                    544 securities
                    NaN signals: 8
                    



```python
print("Task 3: Sample of result with signal_score:")
display(result.head(10))
```

    Task 3: Sample of result with signal_score:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>index_name</th>
      <th>adia_id</th>
      <th>security_name</th>
      <th>price</th>
      <th>weight</th>
      <th>signal_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-06-10 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>9CDDCF1F</td>
      <td>AGILENT TECHNOLOGIES INC</td>
      <td>70.29</td>
      <td>0.000928</td>
      <td>-0.372500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-26 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>9CDDCF1F</td>
      <td>AGILENT TECHNOLOGIES INC</td>
      <td>66.37</td>
      <td>0.000942</td>
      <td>-0.144120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-04-26 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>9CDDCF1F</td>
      <td>AGILENT TECHNOLOGIES INC</td>
      <td>77.42</td>
      <td>0.001004</td>
      <td>0.841684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-07 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>9CDDCF1F</td>
      <td>AGILENT TECHNOLOGIES INC</td>
      <td>66.85</td>
      <td>0.000996</td>
      <td>0.715735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-11-18 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>04071152</td>
      <td>AMERICAN AIRLINES GROUP INC.</td>
      <td>28.60</td>
      <td>0.000437</td>
      <td>-1.186945</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-10-03 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>04071152</td>
      <td>AMERICAN AIRLINES GROUP INC.</td>
      <td>38.80</td>
      <td>0.000644</td>
      <td>0.025634</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-06-05 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>79860674</td>
      <td>ADVANCE AUTO PARTS INC</td>
      <td>129.95</td>
      <td>0.000410</td>
      <td>-0.748866</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-09-06 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>19FE06DF</td>
      <td>APPLE INC.</td>
      <td>223.10</td>
      <td>0.044940</td>
      <td>1.856071</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-02-02 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>19FE06DF</td>
      <td>APPLE INC.</td>
      <td>160.50</td>
      <td>0.034958</td>
      <td>-1.034446</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-04-16 00:00:00</td>
      <td>S&amp;P 500</td>
      <td>19FE06DF</td>
      <td>APPLE INC.</td>
      <td>199.25</td>
      <td>0.036853</td>
      <td>-0.485599</td>
    </tr>
  </tbody>
</table>
</div>


Where 'result' is the pandas DataFrame.
Does this implementation align with what you’d expect for generating a robust signal? Why or why not?
What would you change or improve? Only a few sentences are necessary.


3.3. **Findings**


```python
print("""
No, this implementation is a basic z-score of index weight over the full two-year period, 
which makes it sensitive to long-term trends and stable for securities with little variation 
or short history (as shown by the NaN scores. A more robust signal would use rolling window 
for mean/std to adapt over time, filter out groups with too few observation, and normalize 
cross-sectionally to account for market-wide shift
""")
```

    
    No, this implementation is a basic z-score of index weight over the full two-year period, 
    which makes it sensitive to long-term trends and stable for securities with little variation 
    or short history (as shown by the NaN scores. A more robust signal would use rolling window 
    for mean/std to adapt over time, filter out groups with too few observation, and normalize 
    cross-sectionally to account for market-wide shift
    



```python

```
