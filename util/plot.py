import matplotlib.pyplot as plt

results = [0.205, 0.9067, 0.8667]

# Labels for the bars
labels = ['Pure QNN', 'Classical NN', 'Hybrid NN']

# Create a figure and axis
fig, ax = plt.subplots()

# Create the bar plot
ax.bar(labels, results)

# Set the y-axis limits from 0 to 1
ax.set_ylim(0, 1)

# Add labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracies of Different Models 10-digit MNIST Classification')

# Display the plot
plt.savefig("plot.png")
