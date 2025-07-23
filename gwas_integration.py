#!/usr/bin/env python3
"""
GWAS Integration Module

Provides access to human GWAS data from NHGRI-EBI GWAS Catalog
for trait-gene associations, specifically targeting trait classes:
- Glycemic (diabetes, glucose, insulin resistance)
- Lipid (cholesterol, triglycerides, lipoproteins)
- Hepatic (liver function, fatty liver, hepatic enzymes)
"""

import pandas as pd
import requests
import logging
from typing import Set
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GWASCatalog:
    """
    Provides access to NHGRI-EBI GWAS Catalog data by downloading the
    curated, significant associations file and querying it locally with pandas.
    This is the most reliable method.
    """
    
    def __init__(self, **kwargs):
        self.file_url = "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative"
        self.local_path = Path("./gwas_catalog_associations.tsv")
        self.data = None
        
        self.trait_classes = {
            'glycemic': [
                'diabetes', 'glucose', 'insulin', 'glycemic', 'diabetic', 
                'hyperglycemia', 'hypoglycemia', 'hba1c', 'fasting glucose',
                'insulin resistance', 'glucose tolerance', 'beta cell function'
            ],
            'lipid': [
                'cholesterol', 'triglyceride', 'lipid', 'lipoprotein', 'ldl',
                'hdl', 'vldl', 'fatty acid', 'lipemia', 'dyslipidemia',
                'hyperlipidemia', 'hypercholesterolemia'
            ],
            'hepatic': [
                'liver', 'hepatic', 'alanine aminotransferase', 'alt', 'ast',
                'aspartate aminotransferase', 'alkaline phosphatase', 'alp',
                'bilirubin', 'fatty liver', 'steatosis', 'hepatitis',
                'liver enzyme', 'liver function'
            ]
        }

    def _download_file(self, force=False):
        """Downloads the curated associations file."""
        if self.local_path.exists() and not force:
            logger.info("GWAS curated associations file already exists. Skipping download.")
            return

        logger.info(f"Downloading curated GWAS associations from {self.file_url}...")
        try:
            with requests.get(self.file_url, stream=True) as r:
                r.raise_for_status()
                with open(self.local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("✅ Successfully downloaded curated GWAS associations.")
        except Exception as e:
            logger.error(f"❌ Failed to download GWAS file: {e}")
            raise

    def load_data(self):
        """Loads the associations file into a pandas DataFrame."""
        if self.data is not None:
            return
            
        self._download_file()
        
        logger.info(f"Loading GWAS associations from {self.local_path} into memory...")
        try:
            self.data = pd.read_csv(self.local_path, sep='\t', low_memory=False)
            # Ensure p-value column is numeric, coercing errors to NaN
            self.data['P-VALUE'] = pd.to_numeric(self.data['P-VALUE'], errors='coerce')
            logger.info(f"✅ Loaded {len(self.data)} curated associations.")
        except Exception as e:
            logger.error(f"❌ Failed to load GWAS file into pandas: {e}")
            raise

    def get_genes_for_trait_class(self, trait_class: str, p_value_threshold: float = 5e-8) -> Set[str]:
        """
        Get all genes associated with a specific trait class at genome-wide significance.
        """
        if self.data is None:
            self.load_data()

        if trait_class not in self.trait_classes:
            raise ValueError(f"Trait class must be one of: {list(self.trait_classes.keys())}")

        keywords = self.trait_classes[trait_class]
        # Create a regex pattern to search for any of the keywords (case-insensitive)
        pattern = '|'.join(keywords)
        
        # Filter the DataFrame
        df_trait = self.data[self.data['DISEASE/TRAIT'].str.contains(pattern, case=False, na=False)]
        df_significant = df_trait[df_trait['P-VALUE'] < p_value_threshold]

        if df_significant.empty:
            logger.warning(f"No significant genes found for trait class '{trait_class}'")
            return set()

        all_genes = set()
        # Process 'REPORTED GENE(S)' and 'MAPPED_GENE'
        for col in ['REPORTED GENE(S)', 'MAPPED_GENE']:
            for gene_string in df_significant[col].dropna():
                genes = re.split(r'[,;-]\s*|\s+-\s+', str(gene_string))
                for gene in genes:
                    if gene.strip() and gene.strip() not in ['Intergenic', 'NR']:
                        all_genes.add(gene.strip())

        logger.info(f"Found {len(all_genes)} unique significant genes for {trait_class}")
        return all_genes

def main():
    """Example usage for the new GWASCatalog client."""
    print("="*60)
    print("GWAS Gene Discovery (Reliable Local File Version)")
    print("="*60)
    
    gwas_client = GWASCatalog()
    trait_classes = ['glycemic', 'lipid', 'hepatic']
    
    for trait_class in trait_classes:
        print(f"\n🔍 Searching for {trait_class.upper()} trait genes...")
        genes = gwas_client.get_genes_for_trait_class(trait_class)
        print(f"✅ Found {len(genes)} genes for {trait_class} traits")
        if genes:
            print(f"   Example genes: {', '.join(list(genes)[:10])}")

if __name__ == "__main__":
    main() 