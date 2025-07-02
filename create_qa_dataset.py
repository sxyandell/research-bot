import chromadb
import json
from typing import List, Dict, Any
import random
from rag_chatbot import GoogleEmbeddingFunction, QTLChatbot
from datetime import datetime
import time

class QADatasetGenerator:
    def __init__(self, wait_time: float = 1.0):
        """Initialize the QA dataset generator.
        
        Args:
            wait_time: Time to wait between API calls in seconds
        """
        # Initialize the chatbot to access the data
        self.chatbot = QTLChatbot()
        self.wait_time = wait_time
        
    def _parse_document(self, doc: str) -> Dict[str, Any]:
        """Parse a document to extract gene information."""
        try:
            lines = doc.split('\n')
            gene_info = {}
            
            for line in lines:
                if 'Gene:' in line:
                    gene_info['gene'] = line.split('Gene:')[1].strip()
                elif 'LOD:' in line:
                    gene_info['lod'] = line.split('LOD:')[1].strip()
                elif 'Chromosome:' in line:
                    gene_info['chromosome'] = line.split('Chromosome:')[1].strip()
                elif 'Position:' in line:
                    gene_info['position'] = line.split('Position:')[1].strip()
                elif 'Type:' in line:
                    gene_info['type'] = line.split('Type:')[1].strip()
                    
            return gene_info
        except Exception as e:
            print(f"Error parsing document: {str(e)}")
            return None
    
    def generate_qa_pairs(self, num_pairs: int = 50) -> List[Dict[str, str]]:
        """Generate question-answer pairs from the QTL data."""
        qa_pairs = []
        
        # Get all documents from the collection
        results = self.chatbot.collection.get()
        documents = results['documents']
        
        # Template questions for different aspects of QTL data
        question_templates = [
            # Single gene questions
            "What is the LOD score for {gene}?",
            "Where is {gene} located on the chromosome?",
            "Is {gene} a cis or trans-acting QTL?",
            "What is the statistical significance (p-value) for {gene}?",
            "What type of gene is {gene}?",
            "What is the confidence interval for {gene}?",
            "Which chromosome contains {gene}?",
            "What is the position of {gene} on the chromosome?",
            "What are the key characteristics of {gene}?",
            "How significant is the QTL effect for {gene}?"
        ]
        
        # Process each document to create QA pairs
        for doc in documents:
            try:
                gene_info = self._parse_document(doc)
                
                if gene_info:
                    # For single gene questions
                    single_gene_templates = question_templates[:10]
                    for template in random.sample(single_gene_templates, min(2, len(single_gene_templates))):
                        # Extract just the gene name from the document
                        gene_name = gene_info.get('gene', 'this gene')
                        if isinstance(gene_name, str):
                            # Remove any additional information after the gene name
                            gene_name = gene_name.split('|')[0].strip()
                        question = template.format(gene=gene_name)
                        answer = self.chatbot.process_query(question)
                        
                        # Create a simplified QA pair
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "ground_truth_context": doc
                        })
                        
                        # Wait to avoid rate limits
                        time.sleep(self.wait_time)
                        print(f"Generated QA pair {len(qa_pairs)}/{num_pairs}. Waiting {self.wait_time} seconds...")
                    
                    if len(qa_pairs) >= num_pairs:
                        break
                            
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                continue
                
        return qa_pairs[:num_pairs]  # Ensure we don't exceed the requested number of pairs
    
    def save_dataset(self, qa_pairs: List[Dict[str, str]], filename: str = "qtl_qa_dataset.json"):
        """Save the QA pairs to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": qa_pairs,
                "metadata": {
                    "total_pairs": len(qa_pairs),
                    "source": "QTL Database",
                    "generation_date": datetime.now().strftime("%Y-%m-%d"),
                    "wait_time_between_calls": self.wait_time
                }
            }, f, indent=2)
            
def main():
    # Initialize the generator with a 4-second wait time between calls (15 requests per minute)
    generator = QADatasetGenerator(wait_time=4.0)
    
    # Generate QA pairs
    print("Generating QA pairs...")
    qa_pairs = generator.generate_qa_pairs(num_pairs=20)
    
    # Save the dataset
    print(f"\nSaving {len(qa_pairs)} QA pairs to file...")
    generator.save_dataset(qa_pairs)
    print("Dataset generation complete!")

if __name__ == "__main__":
    main() 