# pandas
https://youtu.be/vmEHCJofslg  

```python
import pandas as pd

# create a new dataframe:
df = pd.DataFrame(columns=[])
```
df - dataframe

## Reading/saving data from/to file
```python
# read a CSV file:
df = pd.read_csv(path.csv)
df = pd.read_csv(path.csv, delimiter='\t')

# if the file is really big, read it in chunks:
for df in pd.read_csv(path, chunksize=1000):

# save to csv:
df.to_csv(path)
df.to_csv(path, separator=';')

# read from/save to an excel file:
df = pd.read_excel(path.xlsx)
df.to_excel(path)
```

## Accessing data:
```python
# read first/last n rows:
df.head(n)
df.tail(n)

# list columns:
df.columns

# read column(s) only:
df.col_name
df[str/list]

# integer location, access a scpecific row, a range [i:j:k] can be used
df.iloc[int]

# access a single datum
df.iloc[row, col]
```

## Iterating row by row:
```python
for index, row in df.iterrows():
# index - row index
# row - data
```

## Conditional selection:
```python
# select all rows that meet a condition:
df.loc(condition)
df.loc(df['col_name'] == value)

# combining contitions:
df.loc((condition0) & (condition1) | (condition2))
```

## Sorting data: 
```python
# sort by column(s):
df.sort_values(str/list)
df.sort_values(str/list, ascending=bool/list)

df.sort_values('col_name')
df.sort_values('col_name', ascending=False)

df.sort_values(['col0', 'col1'])
df.sort_values(['col0', 'col1'], ascending=[False, True])
```

## Modifying data:
```python
# drop columns:
df = df.drop(colums=[])

# reorder columns:
df = df[[new_columns_order]]

# add new index:
df = df.reset_index()
df.reset_index(inplace=True)

# reset index (add new, drop the old):
df = df.reset_index(drop=True)

# add a new column:
df['new_col'] = df['other_col'] * 2
```

## Conditional modification:
```python
# modify rows where a condition is met:
df.loc[condition, [columns]] = [new_values]

df.loc[df['col0'] == val, 'col1'] = new_val
```


df.sum(axis=1) - sum for every row


## Grouping data:
df.groupby([column(s)]).mean()

df.groupby(['column']).count()['count'] - super useful?

## Statistics:
```python
# get full stats:
df.describe()

# get single metrics:
df.count()
df.mean()
df.median()
df.max()
df.min()
df.std()
```

df = pd.concat([dataframe0, dataframe1, ...])
