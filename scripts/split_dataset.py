import sys; sys.path.append("..")
import pandas as pd
import gvpgnn.paths as paths

# Open the training data sequences and structure
df = pd.read_csv(paths.data_folder('cath_w_seqs_share.csv'), index_col=0)

FRACTION_TEST = 0.2 # of the entire set
FRACTION_VAL = 0.2 # of the REMAINING training set

n_test = round(len(df) * FRACTION_TEST)
n_training = len(df) - n_test
n_val = round(n_training * FRACTION_VAL)
n_train = n_training - n_val

print("\n-- NOMINAL SPLITS:")
print("Dataset size:", len(df))
print("test:", n_test)
print("train:", n_train)
print("val:", n_val)

sf_count_low_to_high: list[tuple[int, int]] = \
  sorted(df.groupby(by="superfamily").size().to_dict().items(), key=lambda x: x[1])

split_superfamilies = dict(train=[], val=[], test=[])
split_counts = dict(train=0, val=0, test=0)

idx = 0

# Build the test set first so that it receives many small superfamilies.
while split_counts["test"] < n_test:
  sf, count = sf_count_low_to_high[idx]
  split_superfamilies["test"].append(sf)
  split_counts["test"] += count
  idx += 1

# Then build the validation set, which we also want to have a diverse set of families.
while split_counts["val"] < n_val:
  sf, count = sf_count_low_to_high[idx]
  split_superfamilies["val"].append(sf)
  split_counts["val"] += count
  idx += 1

# Every remaining example goes in the training set.
split_superfamilies["train"].extend([
  sf_count_low_to_high[i][0] for i in range(idx, len(sf_count_low_to_high))
])

split_counts["train"] = sum([sf_count_low_to_high[i][1] for i in range(idx, len(sf_count_low_to_high))])

print("\n-- ACTUAL SPLITS:")
print("* test:", split_counts["test"])
print("* val:", split_counts["val"])
print("* train:", split_counts["train"])
print("(Note: Due to the size of superfamilies, the actual splits may be slightly different in size.)")

# Make sure that every example is accounted for!
assert(sum(split_counts.values()) == len(df))

# Write the splits to files:
for split_name in ("train", "val", "test"):
  sf_list = split_superfamilies[split_name]
  df_split = df[df.superfamily.isin(sf_list)]
  df_split.to_csv(paths.data_folder(f"{split_name}_cath_w_seqs_share.csv"), index=False)


def check_disjoint_dataset_splits():
  """Ensure that no examples are shared across splits!"""
  print("Checking that dataset splits are disjoint...")
  cath_ids = dict(train=set(), val=set(), test=set())

  for split_name in cath_ids:
    df_split = pd.read_csv(paths.data_folder(f"{split_name}_cath_w_seqs_share.csv"))
    cath_ids[split_name] = set(df_split.cath_id.unique())

  # Just to be really sure...
  assert(cath_ids["train"].isdisjoint(cath_ids["test"]))
  assert(cath_ids["train"].isdisjoint(cath_ids["val"]))
  assert(cath_ids["val"].isdisjoint(cath_ids["test"]))

  print("OK")


check_disjoint_dataset_splits()