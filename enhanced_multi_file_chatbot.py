#!/usr/bin/env python3
"""
Enhanced Multi-File QTL Chatbot

This chatbot uses the MultiFileAdapter to work with all 40 QTL files
while leveraging your existing working hybrid_qtl_system.py components.
"""

import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

# Import our adapter
from multi_file_adapter import MultiFileAdapter, EnhancedHybridQTLSystem

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

class EnhancedMultiFileQTLChatbot:
    """Enhanced chatbot for multi-file QTL analysis using the adapter approach."""
    
    def __init__(self, data_directory: str = "/data/dev/miniViewer_3.0/"):
        self.data_directory = data_directory
        self.enhanced_system = None
        self.google_model = None
        self.openai_client = None
        
        # Setup LLMs
        self.setup_llms()
        
        # Initialize the multi-file system
        self.initialize_system()
    
    def setup_llms(self):
        """Setup LLM clients."""
        # Load API keys
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
        """Initialize the enhanced multi-file QTL system."""
        logger.info("🚀 Initializing Enhanced Multi-File QTL System...")
        logger.info("This will combine all 40 QTL files into a unified system...")
        
        try:
            # Create adapter
            adapter = MultiFileAdapter(self.data_directory)
            
            # Create enhanced system with Google API key if available
            google_api_key = os.getenv('GOOGLE_API_KEY') if GOOGLE_AVAILABLE else None
            self.enhanced_system = adapter.create_enhanced_hybrid_system(google_api_key)
            
            # Setup vector store
            logger.info("Setting up enhanced vector store...")
            self.enhanced_system.setup_enhanced_vector_store()
            
            # Get system statistics
            stats = self.enhanced_system.get_trait_statistics()
            
            logger.info("✅ Enhanced Multi-File QTL System Ready!")
            logger.info(f"📊 Total QTLs: {stats['total_qtls']:,}")
            logger.info(f"🧬 Total Genes: {stats['total_genes']:,}")
            logger.info(f"🏷️ Trait Types: {len(stats['trait_types'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    def detect_query_intent(self, query: str) -> Tuple[str, Optional[str]]:
        """Detect query intent and trait filter."""
        query_lower = query.lower()
        
        # Extract trait type filter
        trait_filter = None
        trait_mappings = {
            'liver genes': 'liver_genes',
            'liver gene': 'liver_genes',
            'hepatic': 'liver_genes',
            'liver lipids': 'liver_lipids',
            'liver lipid': 'liver_lipids',
            'lipid': 'liver_lipids',
            'liver isoforms': 'liver_isoforms',
            'liver isoform': 'liver_isoforms',
            'isoform': 'liver_isoforms',
            'splice junction': 'liver_splice_juncs',
            'splice': 'liver_splice_juncs',
            'splicing': 'liver_splice_juncs',
            'plasma metabolites': 'plasma_metabolites',
            'plasma metabolite': 'plasma_metabolites',
            'metabolite': 'plasma_metabolites',
            'clinical traits': 'clinical_traits',
            'clinical trait': 'clinical_traits',
            'clinical': 'clinical_traits',
            'phenotype': 'clinical_traits'
        }
        
        for phrase, trait_type in trait_mappings.items():
            if phrase in query_lower:
                trait_filter = trait_type
                break
        
        # Detect intent
        analytical_patterns = [
            r'\b(top|highest|best|maximum|max|count|number|how many)\b',
            r'\b(average|mean|total|sum|statistics)\b',
            r'\b(list|show|give me|find)\b.*\b(genes?|qtls?)\b',
            r'\b(compare|correlation|vs|versus)\b',
            r'\blod\s*(score|>|<|=)\b',
            r'\bchromosome\s*\d+\b',
            r'\btop.*lod\b',
            r'\bhighest.*lod\b'
        ]
        
        semantic_patterns = [
            r'\b(what is|what are|explain|describe|tell me about)\b',
            r'\b(function|role|mechanism|biology|biological)\b',
            r'\b(genetics|genetic|regulation|metabolic)\b'
        ]
        
        analytical_score = sum(1 for p in analytical_patterns if re.search(p, query_lower))
        semantic_score = sum(1 for p in semantic_patterns if re.search(p, query_lower))
        
        intent = 'analytical' if analytical_score > semantic_score else 'semantic'
        
        return intent, trait_filter
    
    def generate_sql_query(self, query: str) -> Optional[str]:
        """Generate SQL query using LLM."""
        if not self.enhanced_system:
            return None
        
        # Get schema information
        schema_prompt = """
        Table: qtl_data
        Columns:
        - gene_symbol (text): Gene name/identifier
        - qtl_lod (numeric): LOD score (genetic association strength)
        - qtl_chr (text): Chromosome (1-19, X, Y)
        - qtl_pos (numeric): Position in Mb
        - qtl_pval (numeric): P-value
        - cis (text): 'TRUE' for cis-acting, 'FALSE' for trans-acting
        - trait_type (text): liver_genes, liver_lipids, clinical_traits, etc.
        - source_file (text): Original filename
        - analysis_type (text): additive, diet_interactive, sex_interactive, etc.
        - cohort (text): all_mice, male_mice, female_mice, HC_mice, HF_mice
        - gene_type (text): protein_coding, lncRNA, etc.
        """
        
        prompt = f"""
        Generate a DuckDB SQL query for this question about QTL data.
        
        SCHEMA:
        {schema_prompt}
        
        QUESTION: {query}
        
        Rules:
        - Return ONLY the SQL query, no explanation
        - Always filter: WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
        - For "top/highest/maximum" questions, use ORDER BY ... DESC LIMIT 1
        - For "top N" questions, use ORDER BY ... DESC LIMIT N
        - Include trait_type in results when relevant
        
        SQL:
        """
        
        # Try Google first
        if self.google_model:
            try:
                response = self.google_model.generate_content(prompt)
                sql_query = response.text.strip()
                
                # Clean the response
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*```$', '', sql_query)
                
                if 'SELECT' in sql_query.upper() and 'FROM' in sql_query.upper():
                    return sql_query
                    
            except Exception as e:
                logger.warning(f"Google SQL generation failed: {e}")
        
        # Try OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Generate only SQL queries, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                sql_query = response.choices[0].message.content.strip()
                
                # Clean the response
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*```$', '', sql_query)
                
                if 'SELECT' in sql_query.upper() and 'FROM' in sql_query.upper():
                    return sql_query
                    
            except Exception as e:
                logger.warning(f"OpenAI SQL generation failed: {e}")
        
        # Fallback patterns
        return self._fallback_sql_patterns(query)
    
    def _fallback_sql_patterns(self, query: str) -> Optional[str]:
        """Fallback SQL patterns for common queries."""
        query_lower = query.lower()
        
        # Top genes by LOD score (including "top lod", "highest lod", etc.)
        if any(word in query_lower for word in ['highest', 'top', 'maximum']) and 'lod' in query_lower:
            return """
            SELECT gene_symbol, trait_type, MAX(qtl_lod) as max_lod, COUNT(*) as qtl_count
            FROM qtl_data 
            WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
            GROUP BY gene_symbol, trait_type
            ORDER BY max_lod DESC 
            LIMIT 1
            """
        
        # Count by trait type
        if ('count' in query_lower or 'how many' in query_lower) and 'trait' in query_lower:
            return """
            SELECT trait_type, COUNT(*) as qtl_count, COUNT(DISTINCT gene_symbol) as unique_genes
            FROM qtl_data 
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
            FROM qtl_data 
            WHERE gene_symbol = '{gene_name}'
            ORDER BY qtl_lod DESC
            """
        
        return None
    
    def format_analytical_results(self, query: str, df: pd.DataFrame) -> str:
        """Format SQL results into readable response."""
        if df.empty:
            return "No results found for this query."
        
        response_parts = [
            f"📊 **Query Results**",
            f"Question: {query}",
            f"Found {len(df)} result{'s' if len(df) != 1 else ''}",
            ""
        ]
        
        # Format based on columns present
        if len(df) == 1 and 'max_lod' in df.columns:
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
                "This gene shows the strongest quantitative trait locus association in the entire multi-file dataset."
            ]
        
        elif 'trait_type' in df.columns and 'qtl_count' in df.columns:
            # Trait comparison
            response_parts.append("🏷️ **QTLs by Trait Type:**")
            for _, row in df.iterrows():
                trait = row['trait_type'].replace('_', ' ').title()
                qtl_count = row['qtl_count']
                gene_count = row.get('unique_genes', 'N/A')
                response_parts.append(f"• {trait}: {qtl_count:,} QTLs ({gene_count} genes)")
        
        elif 'gene_symbol' in df.columns and 'qtl_lod' in df.columns:
            # Gene-specific results
            gene_name = df.iloc[0]['gene_symbol']
            response_parts.append(f"🧬 **QTL Results for Gene {gene_name}:**")
            for _, row in df.head(10).iterrows():
                trait = row.get('trait_type', 'Unknown').replace('_', ' ').title()
                lod = safe_float(row['qtl_lod'])
                pos = safe_float(row.get('qtl_pos', 0))
                chr_pos = f"Chr{row.get('qtl_chr', '?')}:{pos:.1f}Mb"
                response_parts.append(f"• {trait}: LOD {lod:.2f} at {chr_pos}")
        
        else:
            # Generic table format
            response_parts.append("📋 **Results:**")
            for _, row in df.head(10).iterrows():
                row_items = []
                for col, val in row.items():
                    if col in ['gene_symbol', 'trait_type', 'qtl_lod', 'qtl_chr']:
                        if col == 'trait_type':
                            val = str(val).replace('_', ' ').title()
                        row_items.append(f"{col}: {val}")
                response_parts.append(f"• {' | '.join(row_items)}")
            
            if len(df) > 10:
                response_parts.append(f"... and {len(df) - 10} more results")
        
        return "\n".join(response_parts)
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """Main method to answer questions."""
        if not self.enhanced_system:
            return {
                'question': query,
                'answer': "❌ System not initialized. Please restart the chatbot.",
                'error': 'system_not_ready'
            }
        
        start_time = datetime.now()
        
        # Detect intent and trait filter
        intent, trait_filter = self.detect_query_intent(query)
        
        try:
            if intent == "analytical":
                # Generate and execute SQL query
                sql_query = self.generate_sql_query(query)
                
                if sql_query:
                    df_results = self.enhanced_system.analytical_query(sql_query)
                    context = self.format_analytical_results(query, df_results)
                    method = "sql_analytics"
                else:
                    # Fallback to semantic search
                    results = self.enhanced_system.trait_filtered_search(query, trait_filter, n_results=3)
                    context = "\n\n".join([r['content'] for r in results])
                    method = "semantic_fallback"
            
            else:  # semantic
                # Use semantic search
                try:
                    results = self.enhanced_system.trait_filtered_search(query, trait_filter, n_results=5)
                    context = "\n\n".join([r['content'] for r in results])
                    method = "semantic_search"
                except Exception as e:
                    # If semantic search fails, try a simple SQL approach
                    logger.warning(f"Semantic search failed: {e}")
                    fallback_sql = self._fallback_sql_patterns(query)
                    if fallback_sql:
                        df_results = self.enhanced_system.analytical_query(fallback_sql)
                        context = self.format_analytical_results(query, df_results)
                        method = "sql_fallback"
                    else:
                        context = "I'm sorry, I couldn't process that query due to a system issue."
                        method = "error_fallback"
            
            # Simple response for analytical queries
            if intent == "analytical" and method == "sql_analytics":
                answer = context
            else:
                answer = f"Based on the multi-file QTL analysis:\n\n{context}"
            
            return {
                'question': query,
                'answer': answer,
                'intent': intent,
                'trait_filter': trait_filter,
                'method': method,
                'response_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'question': query,
                'answer': f"I encountered an error processing your question: {str(e)}",
                'intent': intent,
                'method': 'error',
                'error': str(e),
                'response_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system."""
        if not self.enhanced_system:
            return {'status': 'not_initialized'}
        
        try:
            stats = self.enhanced_system.get_trait_statistics()
            return {
                'status': 'ready',
                'total_qtls': stats['total_qtls'],
                'total_genes': stats['total_genes'],
                'trait_types': stats['trait_types'],
                'trait_breakdown': stats['by_trait']
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

def interactive_chat():
    """Interactive chat interface."""
    print("🚀 Initializing Enhanced Multi-File QTL Chatbot...")
    print("This combines all 40 QTL files into a unified analysis system...")
    print("Please wait while the system loads...")
    
    try:
        chatbot = EnhancedMultiFileQTLChatbot()
        
        # Get system info
        info = chatbot.get_system_info()
        if info['status'] != 'ready':
            print(f"❌ System initialization failed: {info.get('error', 'Unknown error')}")
            return
        
        print("\n✅ Enhanced Multi-File QTL Chatbot Ready!")
        print(f"📊 Dataset: {info['total_qtls']:,} QTLs across {info['total_genes']:,} genes")
        print(f"��️ Trait types: {', '.join(info['trait_types'])}")
        
        # Welcome message
        print("\n" + "="*70)
        print("🧬 ENHANCED MULTI-FILE QTL ANALYSIS CHATBOT")
        print("="*70)
        print("Ask me anything about QTLs across all trait types!")
        print("\n📚 Example questions:")
        print("• 'highest lod score'")
        print("• 'count by trait type'")
        print("• 'tell me about Gsdma3'")
        print("• 'top 10 genes by lod score'")
        print("\nType 'quit' to exit, 'info' for system stats")
        print("-" * 70)
        
        # Chat loop
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for using the Enhanced Multi-File QTL Chatbot!")
                    break
                
                if user_input.lower() == 'info':
                    info = chatbot.get_system_info()
                    print(f"\n📊 System Information:")
                    print(f"Total QTLs: {info['total_qtls']:,}")
                    print(f"Total Genes: {info['total_genes']:,}")
                    print(f"Trait Types: {len(info['trait_types'])}")
                    for trait, stats in info['trait_breakdown'].items():
                        print(f"  {trait}: {stats['qtl_count']:,} QTLs, {stats['gene_count']:,} genes")
                    continue
                
                if not user_input:
                    continue
                
                print("\n�� Processing your question...")
                result = chatbot.answer_question(user_input)
                
                print(f"\n📋 Intent: {result['intent']} | Method: {result['method']} | Time: {result['response_time']:.2f}s")
                if result.get('trait_filter'):
                    print(f"🎯 Trait filter: {result['trait_filter']}")
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
        print("Please check your data directory and configuration.")

if __name__ == "__main__":
    interactive_chat()
