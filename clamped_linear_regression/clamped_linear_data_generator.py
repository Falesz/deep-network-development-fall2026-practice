import sys
import random

try:
    output_name = sys.argv[1]
    sample_count = int(sys.argv[2])
    true_line_slope = float(sys.argv[3])
    true_line_intercept = float(sys.argv[4])
    true_output_cap = float(sys.argv[5])
except:
    print("Exception thrown while parsing clamped linear data generation arguments! Using default values...")
    output_name = "clamped_linear_data.csv"
    sample_count = 100_000
    true_line_slope = 3
    true_line_intercept = 10
    true_output_cap = 200

with open(output_name, "w") as data_file:
    for i in range(sample_count):
        value1 = random.randint(0, 101) # These are fixed because I don't want this project to get TOO convoluted / overambitious
        value2 = min(true_line_slope * value1 + true_line_intercept + random.randint(-3, 3), true_output_cap)
        data_file.write(f"{value1},{value2}\n")