import pickle
import numpy as np

with open("/cluster/scratch/pangyi/reto/submission_result/expt3/submission_results.pkl", "rb") as f:
    predictions = pickle.load(f)

print("type(predictions):", type(predictions))
print("len(predictions):", len(predictions))

first = predictions[0]
print("type(predictions[0]):", type(first))
print("shape of first:", np.array(first).shape)
print(predictions[0] if isinstance(predictions[0], str) else np.array(predictions[0]).shape)
print("dtype of first:", np.array(first).dtype)
print("min/max of first:", np.array(first).min(), np.array(first).max())


# print("len:", len(predictions))
# print("first path:", predictions[0])

# arr = np.load(predictions[0])
# print("loaded shape:", arr.shape)
# print("dtype:", arr.dtype)
# print("min/max:", arr.min(), arr.max())