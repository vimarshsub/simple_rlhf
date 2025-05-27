"""
Debug script to print trl and transformers versions
"""

import importlib.metadata
import sys

def print_versions():
    """Print version information for key packages"""
    try:
        trl_version = importlib.metadata.version('trl')
        transformers_version = importlib.metadata.version('transformers')
        
        print("=" * 50)
        print(f"TRL VERSION: {trl_version}")
        print(f"TRANSFORMERS VERSION: {transformers_version}")
        print("=" * 50)
        
        # Also print Python version
        print(f"PYTHON VERSION: {sys.version}")
        print("=" * 50)
        
        return True
    except Exception as e:
        print(f"Error getting version info: {str(e)}")
        return False

if __name__ == "__main__":
    print_versions()
