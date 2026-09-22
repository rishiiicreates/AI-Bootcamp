import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)


print("Here's a basic pandas DataFrame:")
print(df)


print("\nAccessing the 'Name' column:")
print(df['Name'])


print("\nDataFrame Info:")
df.info()

print("\nDataFrame Description (numerical columns):")
print(df.describe())
