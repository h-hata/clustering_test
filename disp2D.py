import pickle
import matplotlib.pyplot as plt

with open("points_ring.pkl", "rb") as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

print(class_1)
print(class_2)
print(labels)
for i in range(len(class_1)):
    plt.plot(class_1[i][0], class_1[i][1], "*")
for i in range(len(class_2)):
    plt.plot(class_2[i][0], class_2[i][1], "+")

plt.show()
