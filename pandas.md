# pandas
https://youtu.be/vmEHCJofslg  
```python
import pandas as pd
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

# 
df.columns - list of column names
df.col_name OR df[str/list] - get a specific column(s)

df.iloc[int] - integer location, access a scpecific row, you can use a range [i:j]
df.iloc[row_no, col_no] - access a single datum/'cell'
```

### Iterating row by row:
```
for index, row in df.iterrows():
index - row index
row - data
```

### Conditional selection:
```
df.loc(condition) - only rows for which the condition is true
df.loc(df['col_name'] == value)
df.loc((condition0) & (condition1) | (condition2))
```

## Sorting 
df.sort_values('col_name')
df.sort_values(['col0', 'col1'])
df.sort_values('col_name', ascending=False)
df.sort_values([columns], ascending=[False, True, False...])
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
GROUPBY
df.describe() - stats like min, max etc for each column
df.groupby([column(s)]).mean()
.count()
.mean()
.max()
.min()
.std()
df.groupby(['column']).count()['count'] - super useful?
READ A BIG FILE IN CHUNKS
new_df = pd.DataFrame(columns=[])
df = pd.concat([dataframe0, dataframe1, ...])
