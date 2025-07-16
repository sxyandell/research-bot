#!/usr/bin/env python3
"""
Multi-File QTL Chatbot

Interactive chatbot for analyzing all 40 QTL data files across different
trait types (clinical traits, liver genes, isoforms, lipids, splice junctions, metabolites).
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from multi_file_qtl_system import MultiFileQTLSystem

# Try imports with fallbacks
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

class MultiFileQTLChatbot:
    """Enhanced chatbot for multi-file QTL analysis."""
    
    def __init__(self):
        self.qtl_system = None
        self.setup_llm()
        
        # Query patterns
        self.analytical_patterns = [
            r'\b(count|number of|how many)\b',
            r'\b(top|highest|best|maximum|max)\b',
            r'\b(average|mean|median|total)\b',
            r'\b(list|show me|give me)\b.*\b(genes?|qtls?)\b',
            r'\bcompare.*\b(traits?|categories?)\b',
            r'\b(liver|plasma|clinical).*\b(vs|versus)\b',
            r'\bchromosome\s+\d+\b',
            r'\blod\s+(score|>|<|greater)\b'
        ]
        
        self.semantic_patterns = [
            r'\b(what is|what are|explain|describe)\b',
            r'\b(function|role|mechanism)\b',
            r'\b(biology|biological|genetics)\b'
        ]
    
    def setup_llm(self):
        """Set up LLM clients."""
        self.google_model = None
        self.openai_client = None
        
        if GOOGLE_AVAILABLE:
            try:
                google_api_key = os.getenv('GOOGLE_API_KEY')
                if google_api_key:
                    genai.configure(api_key=google_api_key)
                    self.google_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            except Exception as e:
                logger.warning(f"Google setup failed: {e}")
        
        if OPENAI_AVAILABLE:
            try:
                openai_api_key = os.getenv('OPENAI_API_KEY')
                if openai_api_key:
                    self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except Exception as e:
                logger.warning(f"OpenAI setup failed: {e}")
    
    def initialize_qtl_system(self):
        """Initialize the QTL system."""
        logger.info("Initializing Multi-File QTL System...")
        self.qtl_system = MultiFileQTLSystem()
        self.qtl_system.initialize_system()
        logger.info("System ready!")
    
    def detect_intent(self, query: str) -> Tuple[str, Optional[str]]:
        """Detect query intent and extract trait filter."""
        query_lower = query.lower()
        
        # Extract trait filter
        trait_filter = None
        if 'liver genes' in query_lower:
            trait_filter = 'liver_genes'
        elif 'liver lipids' in query_lower:
            trait_filter = 'liver_lipids'
        elif 'liver isoforms' in query_lower:
            trait_filter = 'liver_isoforms'
        elif 'splice junctions' in query_lower:
            trait_filter = 'liver_splice_juncs'
        elif 'plasma metabolites' in query_lower:
            trait_filter = 'plasma_metabolites'
        elif 'clinical traits' in query_lower:
            trait_filter = 'clinical_traits'
        
        # Determine intent
        analytical_score = sum(1 for p in self.analytical_patterns if re.search(p, query_lower))
        semantic_score = sum(1 for p in self.semantic_patterns if re.search(p, query_lower))
        
        intent = 'analytical' if analytical_score > semantic_score else 'semantic'
        return intent, trait_filter
    
    def generate_sql_query(self, user_query: str) -> Optional[str]:
        """Generate SQL query using LLM."""
        schema = self.qtl_system.get_schema()
        
        prompt = f"""
Generate a DuckDB SQL query for this QTL database question.

SCHEMA:
{schema}

USER QUESTION: {user_query}

Generate ONLY the SQL query:
"""
        
        # Try Google first
        if self.google_model:
            try:
                response = self.google_model.generate_content(prompt)
                sql_query = response.text.strip()
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*```$', '', sql_query)
                return sql_query.strip()
            except Exception as e:
                logger.warning(f"Google SQL generation failed: {e}")
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                sql_query = response.choices[0].message.content.strip()
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*```$', '', sql_query)
                return sql_query.strip()
            except Exception as e:
                logger.warning(f"OpenAI SQL generation failed: {e}")
        
        return None
    
    def fallback_sql(self, query: str) -> Optional[str]:
        """Fallback SQL patterns."""
        query_lower = query.lower()
        
        if 'count' in query_lower or 'how many' in query_lower:
            if 'trait' in query_lower:
                return "SELECT trait_type, COUNT(*) as count FROM qtl_data GROUP BY trait_type ORDER BY count DESC"
            else:
                return "SELECT COUNT(*) FROM qtl_data"
        
        top_match = re.search(r'top\s+(\d+)', query_lower)
        if top_match:
            n = int(top_match.group(1))
            return f"SELECT gene_symbol, trait_type, qtl_lod FROM qtl_data ORDER BY qtl_lod DESC LIMIT {n}"
        
        return None
    
    def process_analytical_query(self, query: str, trait_filter: Optional[str] = None) -> str:
        """Process analytical queries."""
        try:
            sql_query = self.generate_sql_query(query)
            if not sql_query:
                sql_query = self.fallback_sql(query)
            
            if not sql_query:
                return self.process_semantic_query(query, trait_filter)
            
            result_df = self.qtl_system.sql_query(sql_query)
            
            if result_df.empty:
                return "No results found."
            
            response = f"Query: {query}\n\nResults:\n"
            response += self.format_results(result_df)
            return response
            
        except Exception as e:
            logger.error(f"Analytical query error: {e}")
            return self.process_semantic_query(query, trait_filter)
    
    def process_semantic_query(self, query: str, trait_filter: Optional[str] = None) -> str:
        """Process semantic queries."""
        try:
            results = self.qtl_system.semantic_search(
                query=query,
                n_results=5,
                trait_filter=trait_filter
            )
            
            if not results:
                return "No relevant information found."
            
            response = f"Query: {query}\n\n"
            if trait_filter:
                response += f"Filtered to: {trait_filter.replace('_', ' ').title()}\n\n"
            
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['content'][:300]}...\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Semantic query error: {e}")
            return f"Error processing query: {e}"
    
    def format_results(self, df: pd.DataFrame) -> str:
        """Format SQL results."""
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            return f"{value:,}" if isinstance(value, (int, float)) else str(value)
        elif len(df) <= 15:
            return df.to_string(index=False)
        else:
            return df.head(15).to_string(index=False) + f"\n... ({len(df)} total rows)"
    
    def process_query(self, query: str) -> str:
        """Main query processing."""
        if not self.qtl_system:
            return "Please initialize the system first with 'initialize' command."
        
        intent, trait_filter = self.detect_intent(query)
        
        if intent == 'analytical':
            return self.process_analytical_query(query, trait_filter)
        else:
            return self.process_semantic_query(query, trait_filter)
    
    def get_help(self) -> str:
        """Get help message."""
        return """
Multi-File QTL Analysis Chatbot

TRAIT TYPES:
• Clinical Traits - physiological measurements
• Liver Genes - liver gene expression QTLs
• Liver Lipids - liver lipid QTLs
• Liver Isoforms - liver isoform QTLs
• Liver Splice Junctions - liver splicing QTLs
• Plasma Metabolites - plasma metabolite QTLs

EXAMPLE QUERIES:
• "Count QTLs by trait type"
• "Top 10 genes by LOD score"
• "What are QTLs?"
• "Liver genes with LOD > 20"
• "How many QTLs on chromosome 5?"

COMMANDS:
• 'help' - Show this help
• 'initialize' - Initialize system
• 'quit' - Exit
        """.strip()
    
    def run_chat(self):
        """Run interactive chat."""
        print("🧬 Multi-File QTL Analysis Chatbot")
        print("=" * 50)
        print("Type 'help' for examples, 'quit' to exit")
        print("\nInitializing system...")
        
        try:
            self.initialize_qtl_system()
            print("✅ System ready!")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Type 'initialize' to try again.")
        
        while True:
            try:
                user_input = input("\n🔬 Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                elif user_input.lower() in ['help', 'h']:
                    print(self.get_help())
                    continue
                elif user_input.lower() == 'initialize':
                    print("Initializing...")
                    try:
                        self.initialize_qtl_system()
                        print("✅ Initialized!")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue
                
                print("\n🔍 Processing...")
                response = self.process_query(user_input)
                print(f"\n📊 Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    chatbot = MultiFileQTLChatbot()
    chatbot.run_chat()