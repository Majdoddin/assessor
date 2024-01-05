import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python script.py inputfile")
    sys.exit(1)

file_path = sys.argv[1]
with open(file_path, 'r') as file:
    lines = file.readlines()

# Re-extracting depth from each relevant line
depths = []
for line in lines:
    if 'depth:' in line:
        match = re.search(r'depth:(\d+)', line)
        if match:
            depth = match.groups()[0]
            depths.append(int(depth))

var_nums = []
for line in lines:
    if 'var_num:' in line:
        match = re.search(r'var_num:(\d+)', line)
        if match:
            var_num = match.groups()[0]
            var_nums.append(int(var_num))



# Calculating the percentage of entries with var_num >= 6

num_entries_var_ge_6 = sum(1 for var_num in var_nums if var_num >= 6)

percent_var_ge_6 = (num_entries_var_ge_6 / len(var_nums)) * 100

# Calculating the percentage of files with depth >= 6
num_files_depth_ge_6 = sum(1 for d in depths if d >= 6)
total_files = len(depths)
percent_depth_ge_6 = (num_files_depth_ge_6 / total_files) * 100

# Re-creating the histograms for both Depth and Variable Number with percentages
plt.figure(figsize=(12, 6))

# Histogram for Depth with percentages
plt.subplot(1, 2, 1)
plt.hist(depths, bins=range(min(depths), max(depths) + 1), color='skyblue', weights=np.ones(len(depths)) / len(depths))
plt.title('Percentage Histogram of Depths')
plt.xlabel('Depth')
plt.ylabel('Percentage')
plt.xticks(range(min(depths), max(depths) + 1))  # Ensuring integer x-axis for depth
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))  # Convert y-axis to percentage

# Histogram for Variable Numbers with percentages
plt.subplot(1, 2, 2)
plt.hist(var_nums, bins=range(min(var_nums), max(var_nums) + 1), color='lightgreen', weights=np.ones(len(var_nums)) / len(var_nums))
plt.title('Percentage Histogram of Variable Numbers')
plt.xlabel('Variable Number')
plt.ylabel('Percentage')
plt.xticks(range(min(var_nums), max(var_nums) + 1))  # Ensuring integer x-axis for variable numbers
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))  # Convert y-axis to percentage

plt.tight_layout()
plt.savefig('histogram.png')
plt.show()

# Creating a joint plot for both Depth and Variable Number
sns.jointplot(x=depths, y=var_nums, kind="hex", color="purple")
plt.xlabel('Depth')
plt.ylabel('Variable Number')
plt.suptitle('Joint Distribution of Depth and Variable Number', y=1.02)
plt.savefig('joint_dist.png')
plt.show()
