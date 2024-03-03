import yaml


def load_yaml_file(filepath):
    with open(filepath) as file:
        return yaml.safe_load(file)


def compare_structures(data1, data2, path=""):
    if type(data1) != type(data2):
        print(
            f"Type mismatch at {path}: {type(data1).__name__} vs {type(data2).__name__}",
        )
        return False

    if isinstance(data1, dict):
        for key in data1:
            if key not in data2:
                print(f"Missing key in second structure at {path}: {key}")
                continue
            compare_structures(
                data1[key],
                data2[key],
                path=f"{path}.{key}" if path else key,
            )
        for key in data2:
            if key not in data1:
                print(f"Missing key in first structure at {path}: {key}")
        return True
    elif isinstance(data1, list):
        # For simplicity, just compare the first item if it exists, assuming homogeneous lists
        if data1 and data2:
            compare_structures(data1[0], data2[0], path=f"{path}[0]")
        elif not data1 and data2 or data1 and not data2:
            print(f"List length mismatch or one is empty at {path}")
        return True
    else:
        # This part ignores values if they are not container types
        return True


def test_yaml_structure(filepath1, filepath2):
    data1 = load_yaml_file(filepath1)
    data2 = load_yaml_file(filepath2)

    print("Comparing structure...")
    if compare_structures(data1, data2):
        print("The structures match.")
    else:
        print("The structures do not match.")


# Example usage
test_yaml_structure(
    "../config/tests/config.test_simple.yaml",
    "../config/config.default.yaml",
)
