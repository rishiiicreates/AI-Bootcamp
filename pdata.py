import pandas as pd

data = {
    "Name": ["Rahul", "Aman", "Riya"],
    "Age": [20, 21, 19],
    "Marks": [85, 92, 78]
}

df = pd.DataFrame(data)

print(df)

# Show first rows
print(df.head())

# Get one column
print(df["Name"])

# Get a specific row
print(df.iloc[0])

# Find average marks
print(df["Marks"].mean())

# Students with marks greater than 80
print(df[df["Marks"] > 80])
