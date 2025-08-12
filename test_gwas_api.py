#!/usr/bin/env python3
"""
Test script for GWAS API functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))

from tools import query_gwas_api

def test_gwas_api():
    """Test the GWAS API function with various endpoints"""
    
    print("🧬 Testing GWAS API...\n")
    
    # Test 1: Search for associations with diabetes
    print("Test 1: Searching for diabetes associations...")
    try:
        result = query_gwas_api("/associations", {"trait": "diabetes"})
        print(f"✅ Success! Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Search for gene-specific associations
    print("Test 2: Searching for TCF7L2 gene associations...")
    try:
        result = query_gwas_api("/genes/TCF7L2/associations")
        print(f"✅ Success! Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Search for studies
    print("Test 3: Searching for studies...")
    try:
        result = query_gwas_api("/studies", {"size": 5})
        print(f"✅ Success! Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Search for traits
    print("Test 4: Searching for traits...")
    try:
        result = query_gwas_api("/traits", {"size": 5})
        print(f"✅ Success! Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_gwas_api()
