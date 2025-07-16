#!/usr/bin/env python3
"""
Test the simple multi-file chatbot
"""

from simple_multi_file_chatbot import SimpleMultiFileQTLChatbot

def test_top_lod_query():
    """Test the specific query that was failing."""
    
    print("🧪 Testing Simple Multi-File QTL Chatbot...")
    
    try:
        # Initialize chatbot
        chatbot = SimpleMultiFileQTLChatbot()
        
        # Check if system is ready
        info = chatbot.get_system_info()
        if info['status'] != 'ready':
            print(f"❌ System not ready: {info}")
            return
        
        print("✅ System initialized successfully!")
        print(f"📊 {info['total_qtls']:,} QTLs, {info['total_genes']:,} genes")
        print(f"📈 Max LOD: {info['max_lod']:.2f}")
        
        # Test the problematic query
        test_queries = [
            "what is the top lod",
            "highest lod score", 
            "top 5 genes",
            "count by trait type"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            result = chatbot.answer_question(query)
            print(f"📋 Method: {result['method']} | Time: {result['response_time']:.2f}s")
            print(f"📊 Answer: {result['answer'][:200]}...")
            if result.get('sql_query'):
                print(f"🔧 SQL: {result['sql_query'].strip()[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_top_lod_query()