import matplotlib.pyplot as plt

# Data
students = ["Aman", "Riya", "Rahul", "Neha"]
marks = [85, 92, 78, 88]

# Create a bar graph
plt.bar(students, marks)

# Add labels
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks")

# Display graph
plt.show()


plt.plot(x,y)
plt.bar(x,y)
plt.scatter(x,y)
plt.hist(data)
plt.pie(data)
