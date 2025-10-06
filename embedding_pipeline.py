#!/usr/bin/env python3
"""
NIH Grants Embedding and Visualization Pipeline

Processes NIH grant data to generate embeddings, perform dimension reduction,
and create visualization-ready data.
"""

import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import math


class GrantsEmbeddingPipeline:
    def __init__(
        self,
        input_tsv: str = "./nih-reporter-grants.tsv",
        embedding_model: str = "google/embeddinggemma-300m",
        embedding_output: str = "./nih-reporter-grants.embedding.npy",
        embd_output: str = "./nih-reporter-grants.embd.npy",
        final_output: str = "./grants.tsv",
        batch_size: int = 32,
        reduce_method: str = "opentsne",
    ):
        self.input_tsv = Path(input_tsv)
        self.embedding_model = embedding_model
        self.embedding_output = Path(embedding_output)
        self.embd_output = Path(embd_output)
        self.final_output = Path(final_output)
        self.batch_size = batch_size
        self.reduce_method = reduce_method.lower()

        # Color mapping for spending categories (20 colors from tab20)
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        self.color_hex = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
                        for r, g, b, _ in self.colors]

    def create_text_representation(self, row: Dict) -> str:
        """Create text representation for embedding."""
        fiscal_year = str(row.get('fiscal_year', ''))
        agency_ic_admin = str(row.get('agency_ic_admin', ''))
        activity_code = str(row.get('activity_code', ''))
        project_title = str(row.get('project_title', ''))
        # convert project_title to camel case
        project_title = ' '.join([word.capitalize() for word in project_title.split()])

        abstract_text = str(row.get('abstract_text', ''))
        spending_categories_desc = str(row.get('spending_categories_desc', ''))

        project_start_date = str(row.get('project_start_date', ''))
        project_end_date = str(row.get('project_end_date', ''))

        pref_terms = str(row.get('pref_terms', ''))

        # overall, adding abstract_text maybe not good for embedding
        # return f"{fiscal_year} | {agency_ic_admin} | {activity_code} | {project_title} | {abstract_text}"

        # category + title is better than abstract + title
        # return f"{fiscal_year} | {agency_ic_admin} | {activity_code} | {spending_categories_desc} | {project_title}"

        # bge may not work well with long text
        # return f"{project_title} | {agency_ic_admin} | {activity_code} | {fiscal_year}" # this not well for embedding in baai/bge-small-v1.5

        # return f"{fiscal_year} | {agency_ic_admin} | {project_title}"
        # return f"{agency_ic_admin} | {project_title} | {fiscal_year}"
        # return f"{project_title} | {fiscal_year}"

        # return f"{project_title} | {agency_ic_admin}"
        # return f"{agency_ic_admin} | {project_title}"
        # return f"{project_title}. Project starts on {project_start_date}, ends on {project_end_date}. {agency_ic_admin}."

        # return f"{project_title} | {pref_terms} {spending_categories_desc}"
    
        # For NIA only
        # return f"{project_title} | {pref_terms} {spending_categories_desc}. Project starts on {project_start_date}, ends on {project_end_date}. {agency_ic_admin}."
        # return f"{project_title} | Project starts on {project_start_date}, ends on {project_end_date}."
    
        return f"{project_title} | {abstract_text} | Project starts on {project_start_date}, ends on {project_end_date}"
    
    

    def generate_embeddings(self):
        """Generate text embeddings for all grants."""
        print("🔤 Generating text embeddings...")

        # Check if embeddings already exist
        if self.embedding_output.exists():
            print(f"⚠️  Embeddings file already exists: {self.embedding_output}")
            response = input("Do you want to overwrite it? (y/N): ")
            if response.lower() != 'y':
                print("Skipping embedding generation.")
                return

        # Import sentence transformers (lazy import)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        # Load the model
        print(f"📥 Loading model: {self.embedding_model}")
        model = SentenceTransformer(self.embedding_model)

        # Read TSV file
        print(f"📖 Reading TSV file: {self.input_tsv}")
        df = pd.read_csv(self.input_tsv, sep='\t')
        print(f"📊 Found {len(df)} grants")

        # Create text representations
        print("🔤 Creating text representations...")
        texts = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing text"):
            text = self.create_text_representation(row)
            texts.append(text)

        # Generate embeddings in batches
        print(f"🧠 Generating embeddings with batch size {self.batch_size}...")
        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"✅ Generated embeddings shape: {embeddings.shape}")

        # Save embeddings
        print(f"💾 Saving embeddings to: {self.embedding_output}")
        np.save(self.embedding_output, embeddings)
        print(f"✅ Embeddings saved successfully!")

    def reduce_dimensions(self, method: Optional[str] = None):
        """Reduce embeddings to 2D using the selected method (openTSNE or UMAP)."""
        method = (method or self.reduce_method or "opentsne").lower()
        if method not in {"opentsne", "umap"}:
            raise ValueError(f"Unknown reduction method: {method}. Choose 'opentsne' or 'umap'.")

        print(f"📉 Reducing dimensions with {method.upper()}...")

        # Check if 2D output already exists
        if self.embd_output.exists():
            print(f"⚠️  2D embeddings file already exists: {self.embd_output}")
            response = input("Do you want to overwrite it? (y/N): ")
            if response.lower() != 'y':
                print("Skipping dimension reduction.")
                return

        # Check if embeddings exist
        if not self.embedding_output.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embedding_output}")

        # Lazy import for the chosen method
        if method == "umap":
            try:
                import umap  # type: ignore
            except ImportError:
                raise ImportError("Please install umap-learn: pip install umap-learn")
        else:  # opentsne
            try:
                from openTSNE import TSNE  # type: ignore
            except ImportError:
                raise ImportError("Please install openTSNE: pip install openTSNE")

        # Load embeddings
        print(f"📥 Loading embeddings from: {self.embedding_output}")
        embeddings = np.load(self.embedding_output)
        print(f"📊 Embeddings shape: {embeddings.shape}")

        # Initialize and run the selected reducer
        print("🔧 Initializing reducer...")
        if method == "umap":
            reducer = umap.UMAP(
                n_components=2,
                verbose=True  # Show UMAP progress
            )
            print("🔄 Performing dimension reduction (UMAP)...")
            embeddings_2d = reducer.fit_transform(embeddings)

            # the range of UMAP results is around -20 to 20
            # we want to scale it to around -110 to 110
            embeddings_2d = embeddings_2d * 5.5

        else:
            # openTSNE
            tsne = TSNE(
                n_components=2,
                n_jobs=-1,
                verbose=True,
                n_iter=3000,
            )
            print("🔄 Performing dimension reduction (openTSNE)...")
            embeddings_2d = tsne.fit(embeddings)
        print(f"✅ 2D embeddings shape: {embeddings_2d.shape}")

        # Save 2D embeddings
        print(f"💾 Saving 2D embeddings to: {self.embd_output}")
        np.save(self.embd_output, embeddings_2d)
        print(f"✅ 2D embeddings saved successfully!")

    def get_spending_category_color(self, spending_categories_desc: str) -> str:
        """Get color based on first spending category."""
        if not spending_categories_desc:
            return self.color_hex[0]  # Default color

        # Split by semicolon and get first category
        categories = spending_categories_desc.split(';')
        if not categories:
            return self.color_hex[0]

        first_category = categories[0].strip()

        # Create a simple hash-based color mapping
        # This ensures consistent colors for the same category
        color_index = hash(first_category) % len(self.color_hex)
        return self.color_hex[color_index]

    def merge_with_embeddings(self):
        """Merge TSV data with 2D embeddings to create final visualization file."""
        print("🔗 Merging TSV data with 2D embeddings...")

        # Check if final output already exists
        if self.final_output.exists():
            print(f"⚠️  Final output file already exists: {self.final_output}")
            response = input("Do you want to overwrite it? (y/N): ")
            if response.lower() != 'y':
                print("Skipping merge.")
                return

        # Check if required files exist
        if not self.input_tsv.exists():
            raise FileNotFoundError(f"Input TSV file not found: {self.input_tsv}")
        if not self.embd_output.exists():
            raise FileNotFoundError(f"2D embeddings file not found: {self.embd_output}")

        # Load data
        print(f"📥 Loading TSV data from: {self.input_tsv}")
        df = pd.read_csv(self.input_tsv, sep='\t')
        print(f"📊 TSV data shape: {df.shape}")

        print(f"📥 Loading 2D embeddings from: {self.embd_output}")
        embeddings_2d = np.load(self.embd_output)
        print(f"📊 2D embeddings shape: {embeddings_2d.shape}")

        # Verify dimensions match
        if len(df) != len(embeddings_2d):
            raise ValueError(f"Dimension mismatch: TSV has {len(df)} rows, embeddings have {len(embeddings_2d)} rows")

        # Create new dataframe with required columns
        print("🔄 Creating merged dataframe...")
        merged_data = []

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Merging data")):
            # Extract required fields
            pid = row.get('core_project_num', '')

            # Date handling
            project_start_date = row.get('project_start_date', '')
            if project_start_date and project_start_date != '' and str(project_start_date) != 'nan':
                date = project_start_date
            else:
                fiscal_year = row.get('fiscal_year', '')
                date = f"{fiscal_year}-01-01" if fiscal_year else ""

            journal = row.get('agency_ic_admin', '')
            title = row.get('project_title', '')

            # Mesh terms combination
            pref_terms = str(row.get('pref_terms', ''))
            spending_categories_desc = str(row.get('spending_categories_desc', ''))
            org_name = str(row.get('org_name', ''))

            mesh_parts = []
            if pref_terms and pref_terms != 'nan':
                mesh_parts.append(pref_terms)
            if spending_categories_desc and spending_categories_desc != 'nan':
                mesh_parts.append(spending_categories_desc)
            if org_name and org_name != 'nan':
                mesh_parts.append(org_name)

            mesh_terms = '; '.join(mesh_parts)

            # Coordinates
            x, y = embeddings_2d[i]

            # Size calculation
            award_amount = row.get('award_amount', 0)
            try:
                award_amount = float(award_amount) if award_amount and str(award_amount) != 'nan' else 0
                size = math.sqrt(award_amount) / 100 if award_amount > 0 else 1
            except (ValueError, TypeError):
                size = 1

            # Citation count (same as award amount)
            citation_count = int(award_amount) if award_amount > 0 else 0

            # Color based on spending category
            color = self.get_spending_category_color(spending_categories_desc)

            merged_data.append({
                'pid': pid,
                'date': date,
                'journal': journal,
                'title': title,
                'mesh_terms': mesh_terms,
                'x': x,
                'y': y,
                'size': size,
                'citation_count': citation_count,
                'color': color
            })

        # Create final dataframe
        final_df = pd.DataFrame(merged_data)
        print(f"✅ Final dataframe shape: {final_df.shape}")

        # Save to TSV
        print(f"💾 Saving merged data to: {self.final_output}")
        final_df.to_csv(self.final_output, sep='\t', index=False)
        print(f"✅ Merged data saved successfully!")

        # Show some statistics
        print(f"\n📊 Final dataset statistics:")
        print(f"   - Total grants: {len(final_df):,}")
        print(f"   - Unique agencies: {final_df['journal'].nunique()}")
        print(f"   - Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        print(f"   - Award amount range: ${final_df['citation_count'].min():,.0f} to ${final_df['citation_count'].max():,.0f}")
        print(f"   - X coordinate range: {final_df['x'].min():.2f} to {final_df['x'].max():.2f}")
        print(f"   - Y coordinate range: {final_df['y'].min():.2f} to {final_df['y'].max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIH Grants Embedding and Visualization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Main action
    parser.add_argument("action", type=str,
                       help="Action to perform: embed (generate embeddings), reduce (openTSNE/UMAP), merge (create final TSV), or all")

    # File paths
    parser.add_argument("--input", type=str, default="./nih-reporter-grants.tsv",
                       help="Input TSV file path")
    parser.add_argument("--embedding_output", type=str, default="./nih-reporter-grants.embedding.npy",
                       help="Output path for embeddings")
    parser.add_argument("--embd_output", type=str, default="./nih-reporter-grants.embd.npy",
                       help="Output path for 2D embeddings")
    parser.add_argument("--final_output", type=str, default="./grants.tsv",
                       help="Final merged TSV output")

    # Model and processing parameters
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m",
                       help="SentenceTransformer model to use")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for embedding generation")
    parser.add_argument("--reduce_method", type=str, choices=["opentsne", "umap"], default="opentsne",
                       help="Dimension reduction method to use")

    args = parser.parse_args()

    pipeline = GrantsEmbeddingPipeline(
        input_tsv=args.input,
        embedding_model=args.model,
        embedding_output=args.embedding_output,
        embd_output=args.embd_output,
        final_output=args.final_output,
        batch_size=args.batch_size,
        reduce_method=args.reduce_method,
    )

    print(f"🚀 NIH Grants Embedding Pipeline")
    print(f"📁 Input: {pipeline.input_tsv}")
    print(f"🤖 Model: {pipeline.embedding_model}")
    print(f"📊 Action: {args.action}")
    print("-" * 50)

    try:
        if "embed" in args.action:
            pipeline.generate_embeddings()

        if "reduce" in args.action:
            pipeline.reduce_dimensions()

        if "merge" in args.action:
            pipeline.merge_with_embeddings()

        if args.action == "all":
            print("🔄 Running full pipeline...")
            pipeline.generate_embeddings()
            pipeline.reduce_dimensions()
            pipeline.merge_with_embeddings()

        print(f"\n🎉 Pipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
