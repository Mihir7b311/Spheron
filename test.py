# import re

# def extract_functions_from_code(function_code):
#     # Regular expression to match function definitions
#     pattern = r"def\s+(\w+)\s*\(([^)]*)\):"

#     functions = []

#     # Split the code by lines and check for function definitions
#     lines = function_code.strip().split("\n")

#     for line in lines:
#         match = re.match(pattern, line.strip())

#         if match:
#             funcid = match.group(1)
#             args = match.group(2).strip().split(',') if match.group(2).strip() else []
#             # Clean arguments to remove any unnecessary spaces
#             args = [arg.strip() for arg in args]
#             functions.append({"funcid": funcid, "args": args})

#     return functions

# def write_function_data_to_file(functions, output_file):
#     with open(output_file, "w") as f:
#         for idx, func in enumerate(functions):
#             f.write(f"fn{idx+1}: {func['funcid']}\n")
#             f.write(f"args: {func['args']}\n\n")

# # Example Python code (simulating multiple functions)
# python_code = """
# def example_function(x, y, z):
#     return x + y + z

# def add(a, b):
#     return a + b

# def subtract(m, n):
#     return m - n
# """

# # Extract function details
# functions = extract_functions_from_code(python_code)

# # Write the formatted function data to a file
# write_function_data_to_file(functions, "function_data.txt")


# print("Function data has been written to 'function_data.txt'.")
def test_function(x):
    y = x + 1
    return y
