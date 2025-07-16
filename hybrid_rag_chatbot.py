import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Import the hybrid system
from hybrid_qtl_system import HybridQTLSystem

# Import existing components
import google.generativeai as genai
import openai
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRAGChatbot:
    """
    Enhanced RAG chatbot using hybrid 2-layer architecture:
    - Layer 1: Semantic search on summary documents (10k-20k docs)
    - Layer 2: SQL analytics on raw QTL data (800k+ rows)
    """
    
    def __init__(self, csv_file_path: str, 
                 google_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        
        # Initialize hybrid QTL system
        self.qtl_system = HybridQTLSystem(csv_file_path)
        
        # Setup API keys
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        
        # Initialize models
        self.google_model = None
        self.openai_client = None
        
        # Setup models
        self.setup_models()
        
        # Setup hybrid system
        self.setup_hybrid_system()
        
    def setup_models(self):
        """Setup text generation models."""
        # Setup Google Gemini
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self.google_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Google Gemini 2.0 Flash ready")
            except Exception as e:
                logger.warning(f"⚠️ Google model setup failed: {e}")
        
        # Setup OpenAI as backup
        if self.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("✅ OpenAI client ready as backup")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI setup failed: {e}")
    
    def setup_hybrid_system(self):
        """Setup the hybrid QTL system with embeddings and vector store."""
        logger.info("Setting up hybrid QTL system...")
        
        # Setup embedding models
        self.qtl_system.setup_embedding_models(self.google_api_key)
        
        # Setup vector store with local embeddings (to avoid quota issues)
        self.qtl_system.setup_vector_store(use_google_embeddings=False)
        
        logger.info("✅ Hybrid QTL system ready")
    
    def detect_query_intent(self, query: str) -> str:
        """
        Detect whether the query needs semantic search or analytical processing.
        """
        query_lower = query.lower()
        
        # Keywords that suggest analytical queries
        analytical_keywords = [
            'list all', 'show all', 'count', 'how many', 'average', 'mean', 
            'maximum', 'minimum', 'top', 'bottom', 'rank', 'sort', 'filter',
            'lod >', 'lod <', 'lod =', 'p-value', 'correlation', 'compare',
            'statistics', 'statistical', 'chromosome', 'position', 'range',
            'between', 'greater than', 'less than', 'equal to'
        ]
        
        # Keywords that suggest semantic search
        semantic_keywords = [
            'what is', 'what are', 'explain', 'describe', 'tell me about',
            'similar', 'related', 'like', 'function', 'role', 'pathway',
            'biological', 'mechanism', 'regulation', 'metabolism', 'disease',
            'why', 'how does', 'what does', 'significance', 'meaning'
        ]
        
        analytical_score = sum(1 for kw in analytical_keywords if kw in query_lower)
        semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)
        
        # Special patterns for analytical queries
        if any(pattern in query_lower for pattern in ['>', '<', '=', 'top ', 'best ', 'worst ', 'highest ', 'lowest ']):
            analytical_score += 2
        
        if analytical_score > semantic_score:
            return "analytical"
        else:
            return "semantic"
    
    def generate_sql_query(self, query: str) -> Optional[str]:
        """
        Generate SQL query using LLM for analytical questions.
        """
        # Define the database schema for the LLM
        schema_info = """
        Table: qtl_peaks
        Columns:
        - gene_symbol (text): Gene identifier
        - qtl_lod (numeric): LOD score (strength of QTL association)
        - qtl_chr (text): Chromosome number (1-19, X, Y)
        - qtl_pos (numeric): Position in Mb on chromosome
        - qtl_pval (numeric): P-value
        - qtl_qval (numeric): Q-value (FDR)
        - cis (text): 'TRUE' for cis-acting, 'FALSE' for trans-acting
        - gene_type (text): protein_coding, lncRNA, pseudogene, etc.
        - qtl_ci_lo, qtl_ci_hi (numeric): Confidence interval bounds
        """
        
        sql_prompt = f"""
        Generate a SQL query for the following question about QTL data:
        
        Question: "{query}"
        
        Database schema:
        {schema_info}
        
        Guidelines:
        - Return only the SQL query, no explanation
        - Use proper SQL syntax for DuckDB
        - Filter out NULL/empty gene_symbol values: WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
        - For "highest/top/best" questions, use ORDER BY ... DESC LIMIT 1
        - For "top N" questions, use ORDER BY ... DESC LIMIT N
        - For chromosome queries, filter with WHERE qtl_chr = 'X'
        - For cis/trans comparisons, GROUP BY cis
        - For gene-specific queries, use WHERE gene_symbol = 'GENE_NAME'
        - For counting, use COUNT(*) and GROUP BY as appropriate
        
        SQL Query:
        """
        
        # Try to generate SQL using available LLM
        try:
            if self.google_model:
                response = self.google_model.generate_content(sql_prompt)
                sql_query = response.text.strip()
                
                # Clean up the response (remove markdown, etc.)
                if '```sql' in sql_query:
                    sql_query = sql_query.split('```sql')[1].split('```')[0].strip()
                elif '```' in sql_query:
                    sql_query = sql_query.split('```')[1].strip()
                
                # Basic validation - should contain SELECT and FROM
                if 'SELECT' in sql_query.upper() and 'FROM' in sql_query.upper():
                    return sql_query
                    
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Generate only SQL queries, no explanations."},
                        {"role": "user", "content": sql_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                sql_query = response.choices[0].message.content.strip()
                
                # Clean up the response
                if '```sql' in sql_query:
                    sql_query = sql_query.split('```sql')[1].split('```')[0].strip()
                elif '```' in sql_query:
                    sql_query = sql_query.split('```')[1].strip()
                
                # Basic validation
                if 'SELECT' in sql_query.upper() and 'FROM' in sql_query.upper():
                    return sql_query
        
        except Exception as e:
            logger.warning(f"LLM SQL generation failed: {e}")
        
        # Fallback to simple pattern matching for critical queries
        query_lower = query.lower()
        
        # Basic fallbacks for common patterns
        if 'highest' in query_lower and 'lod' in query_lower and 'chromosome' in query_lower:
            import re
            chr_match = re.search(r'chromosome\s*(\d+|[XY])', query_lower)
            if chr_match:
                chr_num = chr_match.group(1)
                return f"""
                SELECT gene_symbol, qtl_lod, qtl_chr, qtl_pos
                FROM qtl_peaks 
                WHERE qtl_chr = '{chr_num}' AND gene_symbol IS NOT NULL AND gene_symbol != 'nan'
                ORDER BY qtl_lod DESC 
                LIMIT 1
                """
        
        elif any(word in query_lower for word in ['highest', 'top', 'maximum']) and 'lod' in query_lower:
            return """
            SELECT gene_symbol, MAX(qtl_lod) as max_lod, COUNT(*) as qtl_count
            FROM qtl_peaks 
            WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
            GROUP BY gene_symbol 
            ORDER BY max_lod DESC 
            LIMIT 1
            """
        
        return None
    
    def format_analytical_results(self, query: str, df: pd.DataFrame) -> str:
        """Format SQL query results into human-readable text."""
        if df.empty:
            return "No results found for this query."
        
        # Get basic info
        num_rows = len(df)
        columns = df.columns.tolist()
        
        # Start building response
        response_parts = [
            f"📊 **Analytical Query Results**",
            f"Query: {query}",
            f"Found {num_rows} result{'s' if num_rows != 1 else ''}",
            ""
        ]
        
        # Format based on query type
        if 'gene_symbol' in columns and 'max_lod' in columns:
            # Check if this is an ordinal query (second, third, etc.)
            query_lower = query.lower()
            ordinal_map = {
                'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
            }
            ordinal_found = None
            for ordinal, num in ordinal_map.items():
                if ordinal in query_lower:
                    ordinal_found = (ordinal, num)
                    break
            
            if ordinal_found and num_rows >= ordinal_found[1]:
                # Handle ordinal queries (e.g., "second highest")
                ordinal_name, ordinal_num = ordinal_found
                target_row = df.iloc[ordinal_num - 1]  # -1 because list is 0-indexed
                gene = target_row['gene_symbol']
                lod = target_row['max_lod']
                count = target_row.get('qtl_count', 'N/A')
                response_parts = [
                    f"🎯 **ANSWER: The gene with the {ordinal_name} highest LOD score is {gene}**",
                    f"📊 LOD Score: {lod:.2f}",
                    f"🔢 Number of QTLs: {count}",
                    "",
                    f"Ranking #{ordinal_num} out of {len(df)} genes by LOD score."
                ]
            elif num_rows == 1:
                # Single result (highest)
                gene = df.iloc[0]['gene_symbol']
                lod = df.iloc[0]['max_lod']
                count = df.iloc[0].get('qtl_count', 'N/A')
                response_parts = [
                    f"🎯 **ANSWER: The gene with the highest LOD score is {gene}**",
                    f"📊 LOD Score: {lod:.2f}",
                    f"🔢 Number of QTLs: {count}",
                    "",
                    "This means this gene has the strongest quantitative trait locus association in the dataset."
                ]
            else:
                # Multiple results
                response_parts.append("🧬 **Top Genes by LOD Score:**")
                for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                    gene = row['gene_symbol']
                    lod = row['max_lod']
                    count = row.get('qtl_count', 'N/A')
                    response_parts.append(f"{i}. {gene}: LOD {lod:.2f} ({count} QTLs)")
        
        elif 'gene_symbol' in columns and 'qtl_lod' in columns and 'qtl_chr' in columns:
            # Single chromosome highest LOD query
            if num_rows >= 1:
                row = df.iloc[0]
                gene = row['gene_symbol']
                lod = row['qtl_lod']
                chr_num = row['qtl_chr']
                pos = row.get('qtl_pos', 'N/A')
                response_parts = [
                    f"🎯 **ANSWER: The highest LOD score on chromosome {chr_num} is {lod:.2f}**",
                    f"🧬 Gene: {gene}",
                    f"📍 Position: {pos} Mb" if pos != 'N/A' else f"📍 Position: {pos}",
                    "",
                    f"This QTL on chromosome {chr_num} has the strongest association with the trait."
                ]
        
        elif 'qtl_chr' in columns and 'qtl_count' in columns:
            # Chromosome statistics
            response_parts.append("🧬 **QTLs by Chromosome:**")
            for _, row in df.iterrows():
                chr_name = row['qtl_chr']
                qtl_count = row['qtl_count']
                unique_genes = row.get('unique_genes', 'N/A')
                response_parts.append(f"• Chr {chr_name}: {qtl_count} QTLs ({unique_genes} genes)")
        
        elif 'cis' in columns:
            # Cis vs trans statistics
            response_parts.append("🔗 **Cis vs Trans QTL Statistics:**")
            for _, row in df.iterrows():
                reg_type = "Cis-acting" if row['cis'] == 'TRUE' else "Trans-acting"
                count = row['qtl_count']
                avg_lod = row['avg_lod']
                genes = row['unique_genes']
                response_parts.append(f"• {reg_type}: {count} QTLs, avg LOD {avg_lod:.2f}, {genes} genes")
        
        elif len(df) == 1 and 'average_lod' in columns:
            # Overall statistics
            row = df.iloc[0]
            response_parts.extend([
                "📈 **Overall QTL Statistics:**",
                f"• Average LOD Score: {row['average_lod']:.2f}",
                f"• Maximum LOD Score: {row['max_lod']:.2f}",
                f"• Minimum LOD Score: {row['min_lod']:.2f}",
                f"• Total QTLs: {row['total_qtls']:,}"
            ])
        
        else:
            # Generic table format
            response_parts.append("📋 **Results:**")
            for _, row in df.head(20).iterrows():  # Limit to first 20 rows
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                response_parts.append(f"• {row_str}")
            
            if len(df) > 20:
                response_parts.append(f"... and {len(df) - 20} more rows")
        
        return "\n".join(response_parts)
    
    def generate_response(self, query: str, context: str, intent: str) -> str:
        """Generate response using available LLM (Google or OpenAI backup)."""
        
        # Create enhanced prompt based on intent
        if intent == "semantic":
            prompt = f"""
            You are a QTL genetics expert assistant. Based on the provided context about quantitative trait loci (QTLs), 
            answer the user's question clearly and scientifically.

            Context from QTL database:
            {context}

            User Question: {query}

            Guidelines:
            - Use the provided context to answer accurately
            - Explain genetic concepts clearly
            - Include specific LOD scores, chromosomes, and gene names when relevant
            - Use scientific terminology appropriately
            - If the context doesn't contain enough information, say so

            Answer:
            """
        else:  # analytical
            prompt = f"""
            The user asked: "{query}"

            Here are the direct results from the QTL database:
            {context}

            Please provide a brief, clear response that:
            1. First directly answers their question with the specific data
            2. Then adds a short biological interpretation (2-3 sentences max)

            Keep it concise and focus on the actual numbers they requested.
            """
        
        # Try Google Gemini first
        if self.google_model:
            try:
                response = self.google_model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Google model failed: {e}")
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful QTL genetics expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI model failed: {e}")
        
        # Final fallback
        return f"""
        Based on the QTL data analysis for your query: "{query}"

        {context}

        Note: Advanced interpretation is currently unavailable due to API limitations. 
        The data above shows the direct results from the QTL database query.
        """
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Main method to answer questions using hybrid architecture.
        """
        start_time = datetime.now()
        
        # Detect intent
        intent = self.detect_query_intent(query)
        logger.info(f"Detected intent: {intent} for query: {query}")
        
        context = ""
        method_used = ""
        
        try:
            if intent == "semantic":
                # Layer 1: Semantic search on summaries
                semantic_results = self.qtl_system.semantic_search(query, n_results=5)
                
                # Combine context from top results
                context_parts = []
                for result in semantic_results:
                    context_parts.append(f"Document: {result['id']}")
                    context_parts.append(result['content'])
                    context_parts.append("")
                
                context = "\n".join(context_parts)
                method_used = "semantic_search"
                
            else:  # analytical
                # Layer 2: SQL analytics on raw data
                sql_query = self.generate_sql_query(query)
                
                if sql_query and 'GENE_NAME' not in sql_query:
                    # Execute the SQL query
                    df_results = self.qtl_system.analytical_query(sql_query)
                    context = self.format_analytical_results(query, df_results)
                    method_used = "sql_analytics"
                    
                else:
                    # Fallback to semantic search for complex analytical queries
                    semantic_results = self.qtl_system.semantic_search(query, n_results=3)
                    context_parts = []
                    for result in semantic_results:
                        context_parts.append(result['content'])
                    context = "\n".join(context_parts)
                    method_used = "semantic_fallback"
            
            # Generate response
            answer = self.generate_response(query, context, intent)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'question': query,
                'answer': answer,
                'intent': intent,
                'method': method_used,
                'response_time': response_time,
                'context_used': context[:500] + "..." if len(context) > 500 else context,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'question': query,
                'answer': f"I encountered an error processing your question: {str(e)}",
                'intent': intent,
                'method': 'error',
                'response_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Interactive chat interface
def interactive_chat():
    """Interactive chat interface for the hybrid QTL system."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('config.env')
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize chatbot
    print("🚀 Initializing Hybrid RAG Chatbot...")
    print("This may take 15-30 seconds to load embeddings and vector store...")
    
    try:
        chatbot = HybridRAGChatbot(
            csv_file_path="/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv",
            google_api_key=google_api_key,
            openai_api_key=openai_api_key
        )
        
        print("\n✅ Hybrid RAG Chatbot ready!")
        print(f"📊 Layer 1: {len(chatbot.qtl_system.summary_docs):,} summary documents")
        print(f"🗃️ Layer 2: {len(chatbot.qtl_system.raw_data):,} raw QTL records")
        
        # Welcome message
        print("\n" + "="*70)
        print("🧬 INTERACTIVE QTL ANALYSIS CHATBOT")
        print("="*70)
        print("Ask me anything about the QTL data! I can handle:")
        print("📚 Semantic queries: 'What are QTLs?', 'Tell me about metabolism'")
        print("📊 Analytical queries: 'Top genes by LOD score', 'Count by chromosome'")
        print("\nType 'quit', 'exit', or 'bye' to stop")
        print("Type 'examples' to see sample questions")
        print("Type 'help' for more information")
        print("-" * 70)
        
        # Chat loop
        chat_count = 0
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 Your question: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for using the Hybrid QTL Chatbot! Goodbye!")
                    break
                
                # Handle special commands
                if user_input.lower() == 'examples':
                    print("\n📝 Example Questions:")
                    print("\n🔬 Semantic/Conceptual:")
                    print("  • What are QTLs and how do they work?")
                    print("  • Tell me about cis-acting regulation")
                    print("  • What genes are involved in metabolism?")
                    print("  • Explain the biological significance of LOD scores")
                    
                    print("\n📊 Analytical/Data:")
                    print("  • What are the top 10 genes by LOD score?")
                    print("  • How many QTLs are on each chromosome?")
                    print("  • Compare cis vs trans-acting QTLs")
                    print("  • What's the average LOD score?")
                    continue
                
                if user_input.lower() == 'help':
                    print("\n❓ How to use this chatbot:")
                    print("• Ask natural language questions about QTL genetics")
                    print("• The system automatically detects if you want:")
                    print("  - Semantic search (concepts, explanations)")
                    print("  - Analytical queries (statistics, data)")
                    print("• No special syntax required - just ask normally!")
                    print("• Questions are routed to the optimal layer automatically")
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the question
                print("\n🤖 Processing your question...")
                
                result = chatbot.answer_question(user_input)
                chat_count += 1
                
                # Display results
                print(f"\n📋 Query #{chat_count}")
                print(f"Intent: {result['intent']} | Method: {result['method']} | Time: {result['response_time']:.2f}s")
                print("-" * 50)
                print(result['answer'])
                
                # Show context info for debugging (optional)
                if result['method'] == 'sql_analytics':
                    print(f"\n🔍 [Debug] SQL query executed successfully")
                elif result['method'] == 'semantic_search':
                    print(f"\n🔍 [Debug] Found relevant summary documents")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try rephrasing your question or type 'help' for guidance.")
    
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        print("Please check your config.env file and data path.")

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run demonstration mode
        from dotenv import load_dotenv
        load_dotenv('config.env')
        
        google_api_key = os.getenv('GOOGLE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize chatbot
        print("🚀 Initializing Hybrid RAG Chatbot for demonstration...")
        chatbot = HybridRAGChatbot(
            csv_file_path="/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv",
            google_api_key=google_api_key,
            openai_api_key=openai_api_key
        )
        
        print("✅ Hybrid RAG Chatbot ready!")
        print(f"📊 Layer 1: {len(chatbot.qtl_system.summary_docs)} summary documents")
        print(f"🗃️ Layer 2: {len(chatbot.qtl_system.raw_data)} raw QTL records")
        
        # Example queries
        test_queries = [
            # Semantic queries
            "What are QTLs and what do they tell us about genetics?",
            "Tell me about genes with strong metabolic effects",
            
            # Analytical queries  
            "What are the top 5 genes by LOD score?",
            "Compare cis-acting vs trans-acting QTLs"
        ]
        
        print("\n" + "="*60)
        print("DEMONSTRATION: Hybrid RAG Chatbot")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 50)
            
            result = chatbot.answer_question(query)
            
            print(f"Intent: {result['intent']}")
            print(f"Method: {result['method']}")
            print(f"Response time: {result['response_time']:.2f}s")
            print(f"\nAnswer:\n{result['answer']}")
            
            if i < len(test_queries):
                print("\n" + "."*50)
        
        print(f"\n🎉 Hybrid RAG demonstration complete!")
        print(f"💡 The system automatically routes queries between semantic search and SQL analytics")
    
    else:
        # Run interactive mode by default
        interactive_chat() 