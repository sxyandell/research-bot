#!/usr/bin/env python3
"""
Simple Multi-File QTL Chatbot

A working chatbot that uses all 40 QTL files with SQL queries only.
No vector store complexity - just direct SQL analytics.
"""

import os
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from hybrid_qtl_system import HybridQTLSystem

# Import LLM libraries with fallbacks
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

class SimpleMultiFileQTLChatbot:
    """Simple chatbot for multi-file QTL analysis using SQL only."""
    
    def __init__(self):
        self.system = None
        self.google_model = None
        self.openai_client = None
        
        # Setup LLMs
        self.setup_llms()
        
        # Initialize system
        self.initialize_system()
    
    def setup_llms(self):
        """Setup LLM clients."""
        from dotenv import load_dotenv
        load_dotenv('config.env')
        
        # Setup Google Gemini
        if GOOGLE_AVAILABLE:
            try:
                google_api_key = os.getenv('GOOGLE_API_KEY')
                if google_api_key:
                    genai.configure(api_key=google_api_key)
                    self.google_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    logger.info("✅ Google Gemini ready")
            except Exception as e:
                logger.warning(f"Google setup failed: {e}")
        
        # Setup OpenAI
        if OPENAI_AVAILABLE:
            try:
                openai_api_key = os.getenv('OPENAI_API_KEY')
                if openai_api_key:
                    self.openai_client = openai.OpenAI(api_key=openai_api_key)
                    logger.info("✅ OpenAI ready")
            except Exception as e:
                logger.warning(f"OpenAI setup failed: {e}")
    
    def initialize_system(self):
        """Initialize the QTL system with unified dataset."""
        logger.info("🚀 Initializing Simple Multi-File QTL System...")
        
        try:
            # Use the unified dataset created by the adapter
            self.system = HybridQTLSystem("./unified_qtl_data.csv")
            
            logger.info("✅ Simple Multi-File QTL System Ready!")
            logger.info(f"📊 Total QTLs: {len(self.system.raw_data):,}")
            logger.info(f"🧬 Total Genes: {self.system.raw_data['gene_symbol'].nunique():,}")
            logger.info(f"🏷️ Trait Types: {list(self.system.raw_data['trait_type'].unique())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    def generate_sql_query(self, query: str) -> Optional[str]:
        """Generate SQL query using LLM or patterns."""
        
        # Simple pattern matching first
        query_lower = query.lower()
        
        # Highest/top LOD score
        if any(word in query_lower for word in ['highest', 'top', 'maximum']) and 'lod' in query_lower:
            return """
            SELECT gene_symbol, trait_type, MAX(qtl_lod) as max_lod, COUNT(*) as qtl_count
            FROM qtl_peaks 
            WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
            GROUP BY gene_symbol, trait_type
            ORDER BY max_lod DESC 
            LIMIT 1
            """
        
        # Top N genes
        top_match = re.search(r'top\s+(\d+)', query_lower)
        if top_match:
            n = int(top_match.group(1))
            return f"""
            SELECT gene_symbol, trait_type, MAX(qtl_lod) as max_lod, COUNT(*) as qtl_count
            FROM qtl_peaks 
            WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
            GROUP BY gene_symbol, trait_type
            ORDER BY max_lod DESC 
            LIMIT {n}
            """
        
        # Count by trait type
        if ('count' in query_lower or 'how many' in query_lower) and 'trait' in query_lower:
            return """
            SELECT trait_type, COUNT(*) as qtl_count, COUNT(DISTINCT gene_symbol) as unique_genes,
                   AVG(qtl_lod) as avg_lod, MAX(qtl_lod) as max_lod
            FROM qtl_peaks 
            WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
            GROUP BY trait_type 
            ORDER BY qtl_count DESC
            """
        
        # Gene-specific query
        gene_match = re.search(r'\b([A-Z][a-z0-9]+[0-9]*)\b', query)
        if gene_match:
            gene_name = gene_match.group(1)
            return f"""
            SELECT gene_symbol, trait_type, qtl_lod, qtl_chr, qtl_pos
            FROM qtl_peaks 
            WHERE gene_symbol = '{gene_name}'
            ORDER BY qtl_lod DESC
            LIMIT 10
            """
        
        # Overall statistics
        if any(word in query_lower for word in ['overview', 'summary', 'statistics', 'stats']):
            return """
            SELECT 
                COUNT(*) as total_qtls,
                COUNT(DISTINCT gene_symbol) as unique_genes,
                COUNT(DISTINCT trait_type) as trait_types,
                AVG(qtl_lod) as avg_lod,
                MAX(qtl_lod) as max_lod,
                MIN(qtl_lod) as min_lod
            FROM qtl_peaks 
            WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
            """
        
        # Try LLM if available
        if self.google_model:
            try:
                schema_prompt = """
                Table: qtl_peaks
                Columns: gene_symbol, qtl_lod, qtl_chr, qtl_pos, qtl_pval, cis, trait_type, source_file, analysis_type, cohort, gene_type
                """
                
                prompt = f"""
                Generate a DuckDB SQL query for: {query}
                
                Schema: {schema_prompt}
                
                Rules:
                - Return ONLY the SQL query
                - Always filter: WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
                - Use LIMIT for top/highest queries
                
                SQL:
                """
                
                response = self.google_model.generate_content(prompt)
                sql_query = response.text.strip()
                
                # Clean the response
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*```$', '', sql_query)
                
                if 'SELECT' in sql_query.upper() and 'FROM' in sql_query.upper():
                    return sql_query
                    
            except Exception as e:
                logger.warning(f"LLM SQL generation failed: {e}")
        
        return None
    
    def format_results(self, query: str, df: pd.DataFrame) -> str:
        """Format SQL results into readable response."""
        if df.empty:
            return "No results found for this query."
        
        response_parts = [
            f"📊 **Query Results for: {query}**",
            f"Found {len(df)} result{'s' if len(df) != 1 else ''}",
            ""
        ]
        
        # Format based on columns present
        if 'max_lod' in df.columns and len(df) == 1:
            # Single highest result
            row = df.iloc[0]
            gene = row.get('gene_symbol', 'Unknown')
            lod = safe_float(row.get('max_lod', 0))
            trait = row.get('trait_type', 'Unknown')
            count = row.get('qtl_count', 'N/A')
            
            response_parts = [
                f"🎯 **ANSWER: The gene with the highest LOD score is {gene}**",
                f"📊 LOD Score: {lod:.2f}",
                f"🏷️ Trait Type: {trait.replace('_', ' ').title()}",
                f"🔢 QTL Count: {count}",
                "",
                "This represents the strongest quantitative trait locus association across all trait types in the dataset."
            ]
        
        elif 'trait_type' in df.columns and 'qtl_count' in df.columns:
            # Trait comparison
            response_parts.append("🏷️ **QTLs by Trait Type:**")
            for _, row in df.iterrows():
                trait = row['trait_type'].replace('_', ' ').title()
                qtl_count = row['qtl_count']
                gene_count = row.get('unique_genes', 'N/A')
                avg_lod = safe_float(row.get('avg_lod', 0))
                max_lod = safe_float(row.get('max_lod', 0))
                response_parts.append(f"• {trait}: {qtl_count:,} QTLs, {gene_count} genes, avg LOD: {avg_lod:.2f}, max LOD: {max_lod:.2f}")
        
        elif 'gene_symbol' in df.columns and 'qtl_lod' in df.columns:
            # Gene-specific or top genes results
            if len(df) == 1:
                gene_name = df.iloc[0]['gene_symbol']
                response_parts.append(f"🧬 **QTL Results for Gene {gene_name}:**")
            else:
                response_parts.append(f"🧬 **Top Genes by LOD Score:**")
            
            for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                gene = row.get('gene_symbol', 'Unknown')
                trait = row.get('trait_type', 'Unknown').replace('_', ' ').title()
                lod = safe_float(row.get('qtl_lod') or row.get('max_lod', 0))
                
                if 'qtl_chr' in row:
                    pos = safe_float(row.get('qtl_pos', 0))
                    chr_pos = f"Chr{row.get('qtl_chr', '?')}:{pos:.1f}Mb"
                    response_parts.append(f"{i}. {gene} ({trait}): LOD {lod:.2f} at {chr_pos}")
                else:
                    count = row.get('qtl_count', 'N/A')
                    response_parts.append(f"{i}. {gene} ({trait}): LOD {lod:.2f} ({count} QTLs)")
        
        elif 'total_qtls' in df.columns:
            # Overall statistics
            row = df.iloc[0]
            response_parts = [
                f"📈 **Overall QTL Dataset Statistics:**",
                f"• Total QTLs: {row['total_qtls']:,}",
                f"• Unique Genes: {row['unique_genes']:,}",
                f"• Trait Types: {row['trait_types']}",
                f"• Average LOD Score: {safe_float(row['avg_lod']):.2f}",
                f"• Maximum LOD Score: {safe_float(row['max_lod']):.2f}",
                f"• Minimum LOD Score: {safe_float(row['min_lod']):.2f}",
                "",
                "This multi-file dataset combines QTL analyses across multiple biological trait types."
            ]
        
        else:
            # Generic table format
            response_parts.append("📋 **Results:**")
            for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                row_items = []
                for col, val in row.items():
                    if col == 'trait_type':
                        val = str(val).replace('_', ' ').title()
                    row_items.append(f"{col}: {val}")
                response_parts.append(f"{i}. {' | '.join(row_items)}")
            
            if len(df) > 10:
                response_parts.append(f"... and {len(df) - 10} more results")
        
        return "\n".join(response_parts)
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """Answer a question using SQL analytics."""
        if not self.system:
            return {
                'question': query,
                'answer': "❌ System not initialized.",
                'error': 'system_not_ready'
            }
        
        start_time = datetime.now()
        
        try:
            # Generate SQL query
            sql_query = self.generate_sql_query(query)
            
            if sql_query:
                # Execute SQL query
                df_results = self.system.analytical_query(sql_query)
                answer = self.format_results(query, df_results)
                method = "sql_analytics"
            else:
                answer = "I couldn't understand that query. Try asking about:\n• 'highest lod score'\n• 'top 10 genes'\n• 'count by trait type'\n• 'overview'\n• Specific gene names like 'Gsdma3'"
                method = "no_pattern"
            
            return {
                'question': query,
                'answer': answer,
                'method': method,
                'response_time': (datetime.now() - start_time).total_seconds(),
                'sql_query': sql_query if sql_query else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'question': query,
                'answer': f"I encountered an error: {str(e)}",
                'method': 'error',
                'error': str(e),
                'response_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        if not self.system:
            return {'status': 'not_initialized'}
        
        try:
            data = self.system.raw_data
            return {
                'status': 'ready',
                'total_qtls': len(data),
                'total_genes': data['gene_symbol'].nunique(),
                'trait_types': list(data['trait_type'].unique()),
                'max_lod': data['qtl_lod'].max(),
                'avg_lod': data['qtl_lod'].mean()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

def interactive_chat():
    """Interactive chat interface."""
    print("🚀 Initializing Simple Multi-File QTL Chatbot...")
    print("Using SQL analytics on unified 40-file dataset...")
    
    try:
        chatbot = SimpleMultiFileQTLChatbot()
        
        # Get system info
        info = chatbot.get_system_info()
        if info['status'] != 'ready':
            print(f"❌ System initialization failed: {info.get('error', 'Unknown error')}")
            return
        
        print("\n✅ Simple Multi-File QTL Chatbot Ready!")
        print(f"📊 Dataset: {info['total_qtls']:,} QTLs across {info['total_genes']:,} genes")
        print(f"🏷️ Trait types: {', '.join(info['trait_types'])}")
        print(f"📈 Max LOD: {info['max_lod']:.2f}, Avg LOD: {info['avg_lod']:.2f}")
        
        # Welcome message
        print("\n" + "="*70)
        print("🧬 SIMPLE MULTI-FILE QTL ANALYSIS CHATBOT")
        print("="*70)
        print("Ask me about QTLs using natural language!")
        print("\n📚 Example questions:")
        print("• 'highest lod score'")
        print("• 'top 10 genes by lod score'")
        print("• 'count by trait type'")
        print("• 'tell me about Gsdma3'")
        print("• 'what is Tdpoz2'")
        print("• 'overview of the dataset'")
        print("\nType 'quit' to exit, 'info' for system stats")
        print("-" * 70)
        
        # Chat loop
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for using the Simple Multi-File QTL Chatbot!")
                    break
                
                if user_input.lower() == 'info':
                    info = chatbot.get_system_info()
                    print(f"\n📊 System Information:")
                    print(f"Total QTLs: {info['total_qtls']:,}")
                    print(f"Total Genes: {info['total_genes']:,}")
                    print(f"Trait Types: {', '.join(info['trait_types'])}")
                    print(f"Max LOD Score: {info['max_lod']:.2f}")
                    print(f"Average LOD Score: {info['avg_lod']:.2f}")
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤖 Processing your question...")
                result = chatbot.answer_question(user_input)
                
                print(f"\n📋 Method: {result['method']} | Time: {result['response_time']:.2f}s")
                if result.get('sql_query'):
                    print(f"🔍 SQL: {result['sql_query'].strip()[:100]}...")
                print("-" * 50)
                print(result['answer'])
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try rephrasing your question.")
    
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        print("Please check your data files and configuration.")

if __name__ == "__main__":
    interactive_chat() 