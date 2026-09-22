import pandas as pd

data = {
    "Name": ["Rahul", "Aman", "Riya"],
    "Age": [20, 21, 19],
    "Marks": [85, 92, 78]
}

df = pd.DataFrame(data)

print(df)



print(df.head())

print(df["Name"])
print(df["Marks"].mean())

print(df.iloc[0])
print(df[df["Marks"] > 80])
