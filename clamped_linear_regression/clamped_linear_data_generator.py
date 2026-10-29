import random

with open("clamped_linear_data.csv", "w") as data_file:
    for i in range(200_000):
        value1 = random.randint(0, 101)
        value2 = min(3 * value1 + 10 + random.randint(-3, 3), 200)
        data_file.write(f"{value1},{value2}\n")