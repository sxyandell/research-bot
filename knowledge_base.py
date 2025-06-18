import json
from typing import Dict, List

class BiologicalKnowledgeBase:
    """Add biological context and knowledge to enhance QTL interpretation."""
    
    def __init__(self):
        self.gene_functions = {}
        self.pathway_descriptions = {}
        self.disease_associations = {}
        self.chromosome_info = {}
        self.qtl_interpretation_guide = {}
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Initialize with curated biological knowledge."""
        
        # QTL interpretation guidelines
        self.qtl_interpretation_guide = {
            "lod_score_interpretation": {
                "very_high": {"threshold": 100, "meaning": "Extremely strong genetic association, very high confidence"},
                "high": {"threshold": 50, "meaning": "Strong genetic association, high confidence"},
                "significant": {"threshold": 7.5, "meaning": "Significant genetic linkage, standard threshold"},
                "suggestive": {"threshold": 6, "meaning": "Suggestive linkage, requires validation"}
            },
            "cis_trans_explanation": {
                "cis": "Local regulation - QTL affects nearby gene expression, typically within 10Mb",
                "trans": "Distant regulation - QTL affects gene expression on different chromosome or distant location"
            },
            "p_value_interpretation": {
                "highly_significant": {"threshold": 1e-10, "meaning": "Extremely significant result"},
                "significant": {"threshold": 0.05, "meaning": "Statistically significant"},
                "nominal": {"threshold": 0.1, "meaning": "Nominally significant, borderline"}
            }
        }
        
        # Chromosome-specific information
        self.chromosome_info = {
            "1": {"size_mb": 195, "features": "Largest autosome, contains many metabolic genes"},
            "2": {"size_mb": 182, "features": "Contains major histocompatibility complex (MHC)"},
            "3": {"size_mb": 160, "features": "Rich in olfactory receptor genes"},
            "4": {"size_mb": 156, "features": "Contains clustered gene families"},
            "5": {"size_mb": 151, "features": "Important for development and metabolism"},
            "6": {"size_mb": 149, "features": "Contains immunoglobulin gene clusters"},
            "7": {"size_mb": 145, "features": "Rich in transcription factors"},
            "8": {"size_mb": 129, "features": "Contains many enzyme-coding genes"},
            "9": {"size_mb": 124, "features": "Important for neural development"},
            "10": {"size_mb": 130, "features": "Contains protocadherin gene clusters"},
            "11": {"size_mb": 122, "features": "Rich in metabolic pathway genes"},
            "12": {"size_mb": 120, "features": "Contains many regulatory genes"},
            "13": {"size_mb": 120, "features": "Important for immune function"},
            "14": {"size_mb": 125, "features": "Contains T-cell receptor genes"},
            "15": {"size_mb": 104, "features": "Rich in ribosomal protein genes"},
            "16": {"size_mb": 98, "features": "Contains many housekeeping genes"},
            "17": {"size_mb": 95, "features": "Dense in protein-coding genes"},
            "18": {"size_mb": 90, "features": "Contains important developmental genes"},
            "19": {"size_mb": 61, "features": "Highest gene density, many zinc finger proteins"},
            "X": {"size_mb": 171, "features": "X-linked inheritance, dosage compensation"}
        }
        
        # Common liver-relevant gene functions
        self.gene_functions = {
            "metabolism": [
                "Glucose metabolism", "Lipid metabolism", "Amino acid metabolism",
                "Glycogen synthesis", "Gluconeogenesis", "Fatty acid synthesis"
            ],
            "detoxification": [
                "Drug metabolism", "Xenobiotic processing", "Phase I metabolism",
                "Phase II metabolism", "Cytochrome P450 activity"
            ],
            "transport": [
                "Lipid transport", "Cholesterol transport", "Bile acid transport",
                "Amino acid transport", "Glucose transport"
            ],
            "regulation": [
                "Transcriptional regulation", "Metabolic regulation", "Circadian rhythm",
                "Stress response", "Inflammation response"
            ]
        }
        
        # Disease associations
        self.disease_associations = {
            "metabolic_disorders": [
                "Type 2 diabetes", "Obesity", "Metabolic syndrome",
                "Non-alcoholic fatty liver disease", "Insulin resistance"
            ],
            "cardiovascular": [
                "Atherosclerosis", "Hyperlipidemia", "Coronary artery disease",
                "Hypertension", "Stroke"
            ],
            "liver_diseases": [
                "Hepatitis", "Cirrhosis", "Liver cancer",
                "Cholestasis", "Wilson disease"
            ]
        }
    
    def interpret_lod_score(self, lod_score: float) -> str:
        """Provide interpretation of LOD score."""
        if lod_score >= 100:
            return f"LOD {lod_score:.1f}: {self.qtl_interpretation_guide['lod_score_interpretation']['very_high']['meaning']}"
        elif lod_score >= 50:
            return f"LOD {lod_score:.1f}: {self.qtl_interpretation_guide['lod_score_interpretation']['high']['meaning']}"
        elif lod_score >= 3:
            return f"LOD {lod_score:.1f}: {self.qtl_interpretation_guide['lod_score_interpretation']['significant']['meaning']}"
        else:
            return f"LOD {lod_score:.1f}: {self.qtl_interpretation_guide['lod_score_interpretation']['suggestive']['meaning']}"
    
    def get_chromosome_context(self, chromosome: str) -> str:
        """Get biological context for a chromosome."""
        chr_str = str(chromosome)
        if chr_str in self.chromosome_info:
            info = self.chromosome_info[chr_str]
            return f"Chromosome {chr_str} ({info['size_mb']} Mb): {info['features']}"
        return f"Chromosome {chr_str}: Standard autosome"
    
    def explain_cis_trans(self, is_cis: bool) -> str:
        """Explain cis vs trans regulation."""
        if is_cis:
            return self.qtl_interpretation_guide['cis_trans_explanation']['cis']
        else:
            return self.qtl_interpretation_guide['cis_trans_explanation']['trans']
    
    def get_liver_context(self) -> str:
        """Get general liver biology context."""
        return """
        LIVER BIOLOGY CONTEXT:
        The liver is the body's largest internal organ and performs over 500 functions:
        
        🧬 GENETIC REGULATION:
        - Highly expressed genes include albumin, transferrin, and cytochrome P450s
        - Many genes show circadian expression patterns
        - Sex differences in gene expression are common
        
        🔬 METABOLIC FUNCTIONS:
        - Glucose homeostasis and glycogen storage
        - Lipid synthesis and cholesterol metabolism  
        - Protein synthesis and amino acid metabolism
        - Bile acid production and fat digestion
        
        🛡️ DETOXIFICATION:
        - Phase I and Phase II drug metabolism
        - Cytochrome P450 enzyme family
        - Glutathione conjugation pathways
        - Alcohol and toxin processing
        
        🩺 DISEASE RELEVANCE:
        - Genetic variants affect drug response
        - Key organ in metabolic diseases
        - Central to cardiovascular risk factors
        - Target for therapeutic interventions
        """
    
    def get_qtl_significance_context(self, lod_score: float, p_value: float = None) -> str:
        """Provide context about QTL significance."""
        context_parts = []
        
        # LOD score context
        context_parts.append(self.interpret_lod_score(lod_score))
        
        # P-value context if available
        if p_value is not None:
            if p_value < 1e-10:
                context_parts.append(f"P-value {p_value:.2e}: Extremely significant, very low chance of false positive")
            elif p_value < 0.001:
                context_parts.append(f"P-value {p_value:.3f}: Highly significant result")
            elif p_value < 0.05:
                context_parts.append(f"P-value {p_value:.3f}: Statistically significant")
            else:
                context_parts.append(f"P-value {p_value:.3f}: Not statistically significant")
        
        # General QTL interpretation
        context_parts.append("\nQTL INTERPRETATION GUIDE:")
        context_parts.append("- LOD > 100: Extremely strong genetic effect")
        context_parts.append("- LOD 50-100: Very strong genetic association") 
        context_parts.append("- LOD 10-50: Strong genetic association")
        context_parts.append("- LOD 3-10: Significant genetic linkage")
        context_parts.append("- LOD < 3: Suggestive, needs validation")
        
        return "\n".join(context_parts)
    
    def create_knowledge_chunks(self) -> List[Dict]:
        """Create knowledge base chunks for the RAG system."""
        knowledge_chunks = []
        
        # General QTL knowledge
        qtl_chunk = {
            'id': 'knowledge_qtl_basics',
            'type': 'knowledge_base',
            'content': f"""
            QTL (QUANTITATIVE TRAIT LOCI) FUNDAMENTALS:
            
            {self.get_qtl_significance_context(50)}  # Example with high LOD
            
            CIS vs TRANS REGULATION:
            - Cis-acting: {self.explain_cis_trans(True)}
            - Trans-acting: {self.explain_cis_trans(False)}
            
            STATISTICAL INTERPRETATION:
            - LOD scores measure evidence for genetic linkage
            - Higher LOD = stronger evidence for QTL at that location
            - P-values measure statistical significance
            - Q-values control for multiple testing (FDR)
            """,
            'metadata': {
                'type': 'qtl_fundamentals',
                'topics': ['lod_scores', 'cis_trans', 'statistics']
            }
        }
        knowledge_chunks.append(qtl_chunk)
        
        # Liver biology knowledge
        liver_chunk = {
            'id': 'knowledge_liver_biology',
            'type': 'knowledge_base', 
            'content': self.get_liver_context(),
            'metadata': {
                'type': 'liver_biology',
                'topics': ['metabolism', 'detoxification', 'disease']
            }
        }
        knowledge_chunks.append(liver_chunk)
        
        # Chromosome-specific knowledge
        for chr_num, info in self.chromosome_info.items():
            chr_chunk = {
                'id': f'knowledge_chromosome_{chr_num}',
                'type': 'knowledge_base',
                'content': f"""
                CHROMOSOME {chr_num} CHARACTERISTICS:
                
                Size: {info['size_mb']} Mb
                Features: {info['features']}
                
                QTL INTERPRETATION FOR CHROMOSOME {chr_num}:
                - QTLs on this chromosome may affect {info['features'].lower()}
                - Physical size influences mapping resolution
                - Gene density affects cis vs trans classification
                """,
                'metadata': {
                    'type': 'chromosome_info',
                    'chromosome': chr_num,
                    'topics': ['genomics', 'mapping']
                }
            }
            knowledge_chunks.append(chr_chunk)
        
        return knowledge_chunks
    
    def enhance_qtl_chunk_with_knowledge(self, qtl_chunk: Dict) -> Dict:
        """Add biological knowledge to an existing QTL chunk."""
        enhanced_chunk = qtl_chunk.copy()
        
        # Extract information from the chunk
        metadata = qtl_chunk.get('metadata', {})
        lod_range = metadata.get('lod_range', [0, 0])
        chromosomes = metadata.get('chromosomes', [])
        cis_count = metadata.get('cis_count', 0)
        trans_count = metadata.get('trans_count', 0)
        
        # Add biological context
        knowledge_context = ["\n=== BIOLOGICAL CONTEXT ==="]
        
        # LOD score interpretation
        if lod_range[1] > 0:
            knowledge_context.append(f"\n📊 SIGNIFICANCE INTERPRETATION:")
            knowledge_context.append(self.interpret_lod_score(lod_range[1]))  # Use max LOD
        
        # Cis/trans interpretation
        if cis_count > 0 or trans_count > 0:
            knowledge_context.append(f"\n🔗 REGULATION TYPES:")
            if cis_count > 0:
                knowledge_context.append(f"- {cis_count} cis-acting QTLs: {self.explain_cis_trans(True)}")
            if trans_count > 0:
                knowledge_context.append(f"- {trans_count} trans-acting QTLs: {self.explain_cis_trans(False)}")
        
        # Chromosome context
        if chromosomes:
            knowledge_context.append(f"\n🧬 CHROMOSOME CONTEXT:")
            for chr_num in chromosomes[:3]:  # Limit to first 3 chromosomes
                knowledge_context.append(self.get_chromosome_context(chr_num))
        
        # Add liver biology context
        knowledge_context.append(f"\n🫀 LIVER RELEVANCE:")
        knowledge_context.append("These QTLs may affect liver metabolism, detoxification, or gene regulation.")
        knowledge_context.append("Genetic variants in liver genes can influence drug response and disease risk.")
        
        # Append knowledge to existing content
        enhanced_chunk['content'] += "\n".join(knowledge_context)
        
        # Update metadata
        if 'topics' not in enhanced_chunk['metadata']:
            enhanced_chunk['metadata']['topics'] = []
        enhanced_chunk['metadata']['topics'].extend(['biological_context', 'interpretation'])
        enhanced_chunk['metadata']['enhanced_with_knowledge'] = True
        
        return enhanced_chunk

# Example usage
if __name__ == "__main__":
    print("🧠 Biological Knowledge Base System")
    print("=" * 50)
    
    kb = BiologicalKnowledgeBase()
    
    # Create knowledge chunks
    knowledge_chunks = kb.create_knowledge_chunks()
    print(f"✅ Created {len(knowledge_chunks)} knowledge base chunks")
    
    # Save knowledge chunks
    with open("biological_knowledge_chunks.json", 'w') as f:
        json.dump(knowledge_chunks, f, indent=2)
    print("💾 Saved knowledge chunks to biological_knowledge_chunks.json")
    
    # Example: enhance existing QTL chunk
    print("\n📋 Example knowledge enhancement:")
    example_chunk = {
        'content': 'Gene: Tdpoz2 | LOD Score: 608.58 | Location: Chr 3, 94.34 Mb',
        'metadata': {
            'lod_range': [500, 608.58],
            'chromosomes': ['3'],
            'cis_count': 1,
            'trans_count': 0
        }
    }
    
    enhanced = kb.enhance_qtl_chunk_with_knowledge(example_chunk)
    print("Enhanced chunk content:")
    print(enhanced['content'][:400] + "...")
    
    print(f"\n✅ Knowledge base ready! You can now:")
    print("   - Add biological context to QTL interpretations")
    print("   - Explain statistical significance")
    print("   - Provide chromosome-specific insights") 
    print("   - Connect findings to liver biology") 