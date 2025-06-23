import numpy as np
import h5py

filename = "data.h5"
y_train_temp = []
with h5py.File(filename, "r") as f:
    for i in f["train_data"].keys():
        y = np.array(f["train_data"][i]["branch"]["status"][()])
        y_train_temp.append(y)


# Convert to NumPy array: shape = (num_samples, num_branches)
y_train_arr = np.stack(y_train_temp)

# Count ON and OFF per branch
branch_on_counts = y_train_arr.sum(axis=0)  # total 1s per column
branch_off_counts = y_train_arr.shape[0] - branch_on_counts  # total 0s

# Percentage ON per branch
branch_on_pct = branch_on_counts / y_train_arr.shape[0]

# Print results
print("Branch\t% ON\tCount ON\tCount OFF")
for i in range(len(branch_on_pct)):
    print(f"{i}\t{branch_on_pct[i]*100:.2f}%\t{int(branch_on_counts[i])}\t\t{int(branch_off_counts[i])}")
