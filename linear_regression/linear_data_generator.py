import random

with open("linear_data.csv", "w") as data_file:
    for i in range(10_000):
        value1 = random.randint(0, 101)
        value2 = 3 * value1 + 10 + random.randint(-3, 3)
        data_file.write(f"{value1},{value2}\n")