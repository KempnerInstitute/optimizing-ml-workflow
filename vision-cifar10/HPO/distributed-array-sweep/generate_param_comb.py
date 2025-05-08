import csv
import itertools

def generate_parameter_combinations(parameters):
    """
    Generate all possible combinations of parameters from a dictionary of parameter values.
    
    Args:
        parameters (dict): Dictionary with parameter names as keys and lists of possible values
        
    Returns:
        list: List of dictionaries, each representing a parameter combination
    """
    # Extract parameter names and their possible values
    param_names = list(parameters.keys())
    param_values = [parameters[name]['values'] for name in param_names]
    
    # Generate all combinations
    combinations = []
    for values in itertools.product(*param_values):
        combination = dict(zip(param_names, values))
        combinations.append(combination)
    
    return combinations

def write_combinations_to_csv(combinations, output_file='parameter_combinations.csv'):
    """
    Write parameter combinations to a CSV file.
    
    Args:
        combinations (list): List of parameter combination dictionaries
        output_file (str): Path to the output CSV file
    """
    if not combinations:
        print("No combinations to write.")
        return
    
    # Get field names from the first combination
    fieldnames = list(combinations[0].keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write parameter combinations
        writer.writerows(combinations)
    
    print(f"Successfully wrote {len(combinations)} combinations to {output_file}")

if __name__ == "__main__":
    # Define parameters
    parameters = {
        'learning_rate': {'values': [0.1, 0.01, 0.001]},
        'batch_size': {'values': [32, 64, 128]},
        'optimizer': {'values': ['adam', 'sgd']},
        'weight_decay': {'values': [0.0001, 0.001]},
        'model_name': {'values': ['resnet18', 'resnet50']},
    }
    
    # Generate all combinations
    combinations = generate_parameter_combinations(parameters)
    
    # Print summary
    print(f"Generated {len(combinations)} parameter combinations")
    
    # Write to CSV file
    write_combinations_to_csv(combinations)

