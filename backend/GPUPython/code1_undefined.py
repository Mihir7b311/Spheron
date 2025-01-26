def add(a, b):
    """Returns the sum of two numbers."""
    return a + b

def subtract(a, b):
    """Returns the difference between two numbers."""
    return a - b

if __name__ == "__main__":
    x = 10
    y = 5
    print(f"Sum of {x} and {y}: {add(x, y)}")
    print(f"Difference between {x} and {y}: {subtract(x, y)}")
