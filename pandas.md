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
df = pd.read_csv(path.csv, delimiter=char)

# if the file is really big, read it in chunks:
for df in pd.read_csv(path, chunsize=int):

# save to csv:
df.to_csv(path, [separator='char'])

# read from/save to an excel file:
df = pd.read_excel(path.xlsx)
df.to_excel(path)
```

## Accessing data:
```python
# read first/last n rows:
df.head(n) - first n rows
df.tail(n) - last n rows

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

### Iterating row by row:
```python
for index, row in df.iterrows():
# index - row index
# row - data
```

### Conditional selection:
```python
# select rows based on a condition:
df.loc(condition) - only rows for which the condition is true
df.loc(df['col_name'] == value)

# joining contitions:
df.loc((condition0) & (condition1) | (condition2))
```

## Sorting data: 
```python
# sort by column(s):
df.sort_values(str/list)
df.sort_values(str/list, ascending=bool/list)

df.sort_values('col_name')
df.sort_values(['col0', 'col1'])
df.sort_values('col_name', ascending=False)
df.sort_values(['col0', 'col1'], ascending=[False, True])
```

CHANGING DATA
df['new_col'] = df['othercol'] + 2 - add new column based on other columns' values
df = df.drop(colums=[]) - drop columns
df.sum(axis=1) - sum for every row
df = df[[colums]] - reorder by columns
df = df.reset_index() - add new index
df = df.reset_index(drop=True) - add new index, drop the old index
df.reset_index(inplace=True)

CONDITIONAL CHANGES
df.loc[df['col'] == val, 'col'] = 'new_val' - change value of col where it has a specified value
We can also change a different column while keeping the same condition
df.loc[df['col'] == val, 'other_col'] = 'new_other_col_val'
Also we can change multiple columns at onece
df.loc[condition, [columns]] = [new_values]
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
