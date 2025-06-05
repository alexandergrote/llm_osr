import time
import functools

def measure_time(func=None, *, unit='seconds', precision=4):
    """
    Decorator to measure and print the execution time of a function.
    
    Args:
        func: The function to decorate (when used without parentheses)
        unit: Time unit for display ('seconds', 'milliseconds', 'microseconds')
        precision: Number of decimal places to display
    
    Usage:
        @measure_time
        def my_function():
            pass
        
        @measure_time(unit='milliseconds', precision=2)
        def my_function():
            pass
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            # Convert to requested unit
            if unit == 'milliseconds':
                execution_time *= 1000
                unit_abbrev = 'ms'
            elif unit == 'microseconds':
                execution_time *= 1_000_000
                unit_abbrev = 'μs'
            else:  # seconds
                unit_abbrev = 's'
            
            print(f"{f.__name__} executed in {execution_time:.{precision}f} {unit_abbrev}")
            return result
        return wrapper
    
    # Handle both @measure_time and @measure_time() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


# Alternative version that returns the time instead of printing
def time_it(func=None, *, return_time=False):
    """
    Decorator that can optionally return execution time along with result.
    
    Args:
        func: The function to decorate
        return_time: If True, returns (result, execution_time) tuple
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            if return_time:
                return result, execution_time
            else:
                print(f"{f.__name__} executed in {execution_time:.4f} seconds")
                return result
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


# Example usage
if __name__ == "__main__":
    # Basic usage with regular function
    @measure_time
    def slow_function():
        time.sleep(0.1)
        return "Done!"
    
    # Class with decorated methods
    class Calculator:
        def __init__(self, name):
            self.name = name
        
        @measure_time
        def compute_sum(self, n):
            """Instance method with decorator"""
            return sum(range(n))
        
        @measure_time(unit='milliseconds', precision=2)
        def compute_factorial(self, n):
            """Instance method with custom timing settings"""
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        
        @classmethod
        @measure_time
        def class_method_example(cls):
            """Class method with decorator"""
            time.sleep(0.05)
            return "Class method executed"
        
        @staticmethod
        @measure_time(unit='microseconds')
        def static_method_example():
            """Static method with decorator"""
            return [i**2 for i in range(100)]
    
    # Test regular function
    print("=== Testing regular function ===")
    result1 = slow_function()
    print(f"Result: {result1}\n")
    
    # Test class methods
    print("=== Testing class instance methods ===")
    calc = Calculator("MyCalculator")
    
    result2 = calc.compute_sum(100000)
    print(f"Sum result: {result2}")
    
    result3 = calc.compute_factorial(10)
    print(f"Factorial result: {result3}")
    
    print("\n=== Testing class and static methods ===")
    result4 = Calculator.class_method_example()
    print(f"Class method result: {result4}")
    
    result5 = Calculator.static_method_example()
    print(f"Static method result length: {len(result5)}")
    
    # You can also create multiple instances
    print("\n=== Testing multiple instances ===")
    calc1 = Calculator("Calc1")
    calc2 = Calculator("Calc2")
    
    calc1.compute_sum(50000)
    calc2.compute_sum(75000)