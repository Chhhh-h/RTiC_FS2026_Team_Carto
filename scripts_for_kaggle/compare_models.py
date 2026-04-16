import torch
import sys

def compare_models(file1, file2, num_elements=10):
    """
    Compares two PyTorch model state dictionaries.

    Args:
        file1 (str): Path to the first .pth file.
        file2 (str): Path to the second .pth file.
        num_elements (int): Number of elements to compare from each tensor.
    """
    print(f"Comparing '{file1}' and '{file2}'")

    # Load the state dictionaries
    try:
        state_dict1 = torch.load(file1, map_location='cpu')
        state_dict2 = torch.load(file2, map_location='cpu')
    except Exception as e:
        print(f"Error loading model files: {e}")
        return

    # If the state dict is nested (e.g., in 'state_dict' or 'model'), get it.
    if 'state_dict' in state_dict1:
        state_dict1 = state_dict1['state_dict']
    if 'model' in state_dict1:
        state_dict1 = state_dict1['model']
        
    if 'state_dict' in state_dict2:
        state_dict2 = state_dict2['state_dict']
    if 'model' in state_dict2:
        state_dict2 = state_dict2['model']


    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Compare parameter structures (keys)
    print("\n--- Key Comparison ---")
    if keys1 == keys2:
        print("Parameter structures are identical.")
    else:
        print("Parameter structures are DIFFERENT.")
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"\nKeys only in '{file1}':")
            for k in sorted(list(only_in_1)):
                print(f"  - {k}")
        if only_in_2:
            print(f"\nKeys only in '{file2}':")
            for k in sorted(list(only_in_2)):
                print(f"  - {k}")

    # Compare parameter values for common keys
    print("\n--- Tensor Value Comparison (first {} elements) ---".format(num_elements))
    common_keys = sorted(list(keys1.intersection(keys2)))

    if not common_keys:
        print("No common keys to compare.")
        return

    all_tensors_equal = True
    for key in common_keys:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if torch.equal(tensor1, tensor2):
            print(f"\n[OK] Tensors for key '{key}' are identical.")
        else:
            all_tensors_equal = False
            print(f"\n[DIFF] Tensors for key '{key}' are DIFFERENT.")
            print(f"  Shape 1: {tensor1.shape}, Shape 2: {tensor2.shape}")
            
            # Flatten and compare first N elements
            flat_tensor1 = tensor1.flatten()
            flat_tensor2 = tensor2.flatten()

            print(f"  Values from '{file1}': {flat_tensor1[:num_elements].tolist()}")
            print(f"  Values from '{file2}': {flat_tensor2[:num_elements].tolist()}")
    
    if all_tensors_equal:
        print("\nAll common tensors have identical values.")
    else:
        print("\nSome common tensors have different values.")


if __name__ == "__main__":
    model_path1 = "/cluster/scratch/pangyi/reto/SegFormer/pretrained/mit_b5.pth"
    model_path2 = "/cluster/scratch/pangyi/reto/SegFormer/pretrained/segformer_b5_backbone_weights.pth"
    
    if len(sys.argv) == 3:
        model_path1 = sys.argv[1]
        model_path2 = sys.argv[2]

    compare_models(model_path1, model_path2)
