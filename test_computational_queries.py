#!/usr/bin/env python3
"""
Test script to demonstrate computational capabilities of the enhanced QTL chatbot.
"""

import sys
import os
sys.path.append(os.getcwd())

from enhanced_computational_chatbot import EnhancedQTLChatbot

def test_computational_queries():
    """Test various computational queries."""
    print("🧪 Testing Enhanced QTL Chatbot Computational Capabilities")
    print("=" * 60)
    
    # Initialize chatbot
    chatbot = EnhancedQTLChatbot()
    
    if chatbot.qtl_data.empty:
        print("❌ No QTL data available")
        return
    
    print(f"✅ Loaded {len(chatbot.qtl_data)} QTL records")
    print("\n🔍 Testing computational queries...\n")
    
    # Test queries
    test_queries = [
        "What is the average LOD score?",
        "What is the average LOD score for cis QTLs?",
        "How many QTLs are on each chromosome?",
        "What is the highest LOD score and which gene?",
        "Show me statistics for the LOD scores",
        "How many cis-acting QTLs are there?",
        "What is the sum of all LOD scores?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔬 Query {i}: {query}")
        print("-" * 50)
        
        try:
            response = chatbot.process_query(query)
            print(f"🤖 Response:\n{response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        print("=" * 60)
        print()

if __name__ == "__main__":
    test_computational_queries() 