#!/usr/bin/env python3
"""
NIH RePORTER Grant-Publication-Author Network Builder

This script reads a list of grants, downloads grant metadata and publications
from NIH RePORTER API, fetches publication metadata from PubMed, and builds
a network of grants, publications, and authors.

Usage Examples:
    # Basic usage with default settings
    python grant_publication_network.py

    # Use existing grants.tsv file
    python grant_publication_network.py --grant-file grants.tsv

    # Generate full pipeline with embeddings and output files
    python grant_publication_network.py --grant-file grants.tsv --generate-outputs

    # Only generate output files (assumes data already downloaded)
    python grant_publication_network.py --grant-file grants.tsv --only-outputs

    # Use existing embeddings and only create TSV/JSON outputs
    python grant_publication_network.py --grant-file grants.tsv --only-outputs --skip-embeddings --skip-reduction

    # Specify custom paths and model
    python grant_publication_network.py --grant-file my_grants.tsv --embedding-model all-MiniLM-L6-v2 --points-output my_points.tsv

    # Use custom database paths
    python grant_publication_network.py --db-path ./my_grants.db --icite-db ~/my_icite.db

Features:
- Automatically handles API rate limiting
- Uses SQLite database for caching to avoid re-downloading
- Processes data in batches for efficiency
- Shows progress bars for all long-running operations
- Creates network with grants, papers, and authors as nodes
- Generates edges for grant-paper and author-paper relationships
- Text embedding generation using SentenceTransformer models
- t-SNE dimension reduction for 2D visualization using openTSNE
- Integration with iCite database for citation metrics
- Generates gpa.points.tsv and gpa.edges.json for network visualization
- Supports incremental processing (skip embedding/reduction steps)
"""

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import dateparser
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
from openTSNE import TSNE
from sentence_transformers import SentenceTransformer
from datetime import datetime
import math


class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, db_path: str = "./grants.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database and create tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create grants table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS grants (
                    core_project_num TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL
                )
            ''')

            # Create grant_publications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS grant_publications (
                    core_project_num TEXT NOT NULL,
                    pmid INTEGER NOT NULL,
                    PRIMARY KEY (core_project_num, pmid)
                )
            ''')

            # Create papers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    pmid INTEGER PRIMARY KEY,
                    metadata TEXT NOT NULL
                )
            ''')

            # Create indices for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_grant_publications_pmid
                ON grant_publications(pmid)
            ''')

            conn.commit()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def grant_exists(self, core_project_num: str) -> bool:
        """Check if grant already exists in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM grants WHERE core_project_num = ?",
                (core_project_num,)
            )
            return cursor.fetchone() is not None

    def insert_grant(self, core_project_num: str, metadata: str):
        """Insert grant metadata into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO grants (core_project_num, metadata) VALUES (?, ?)",
                (core_project_num, metadata)
            )
            conn.commit()

    def paper_exists(self, pmid: int) -> bool:
        """Check if paper already exists in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM papers WHERE pmid = ?", (pmid,))
            return cursor.fetchone() is not None

    def insert_paper(self, pmid: int, metadata: str):
        """Insert paper metadata into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO papers (pmid, metadata) VALUES (?, ?)",
                (pmid, metadata)
            )
            conn.commit()

    def insert_grant_publication(self, core_project_num: str, pmid: int):
        """Insert grant-publication relationship"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO grant_publications (core_project_num, pmid) VALUES (?, ?)",
                (core_project_num, pmid)
            )
            conn.commit()

    def get_missing_grants(self, core_project_nums: List[str]) -> List[str]:
        """Get list of grants not yet in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(core_project_nums))
            cursor.execute(
                f"SELECT core_project_num FROM grants WHERE core_project_num IN ({placeholders})",
                core_project_nums
            )
            existing = {row[0] for row in cursor.fetchall()}
            return [num for num in core_project_nums if num not in existing]

    def get_missing_papers(self, pmids: List[int]) -> List[int]:
        """Get list of papers not yet in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(pmids))
            cursor.execute(
                f"SELECT pmid FROM papers WHERE pmid IN ({placeholders})",
                pmids
            )
            existing = {row[0] for row in cursor.fetchall()}
            return [pmid for pmid in pmids if pmid not in existing]


class NIHReporterAPI:
    """Handles NIH RePORTER API interactions"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.grants_url = "https://api.reporter.nih.gov/v2/projects/search"
        self.publications_url = "https://api.reporter.nih.gov/v2/publications/search"
        self.batch_size = 500
        self.request_delay = 1  # 100ms delay between requests

    def fetch_grants_metadata(self, core_project_nums: List[str]) -> Dict:
        """Fetch grants metadata from NIH RePORTER API"""
        # Filter out grants that already exist in database
        missing_grants = self.db_manager.get_missing_grants(core_project_nums)
        if not missing_grants:
            print("All grants already in database, skipping download.")
            return {}

        print(f"Fetching metadata for {len(missing_grants)} grants...")

        # Process in batches
        total_fetched = 0
        with tqdm(total=len(missing_grants), desc="Fetching grants") as pbar:
            for i in range(0, len(missing_grants), self.batch_size):
                batch = missing_grants[i:i + self.batch_size]

                request_data = {
                    "criteria": {
                        "use_relevance": True,
                        "project_nums": batch
                    },
                    "offset": 0,
                    "limit": len(batch)
                }

                try:
                    response = requests.post(
                        self.grants_url,
                        json=request_data,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()

                    data = response.json()
                    results = data.get('results', [])

                    # Save each grant to database
                    for grant in results:
                        core_project_num = grant.get('core_project_num')
                        if core_project_num:
                            self.db_manager.insert_grant(
                                core_project_num,
                                json.dumps(grant)
                            )
                            total_fetched += 1

                    pbar.update(len(batch))
                    time.sleep(self.request_delay)

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching grant batch: {e}")
                    continue

        print(f"Successfully fetched {total_fetched} grants")
        return {"fetched": total_fetched}

    def fetch_publications(self, core_project_nums: List[str]) -> List[Dict]:
        """Fetch publications for grants from NIH RePORTER API"""
        print(f"Fetching publications for {len(core_project_nums)} grants...")

        all_publications = []
        total_publications = 0

        with tqdm(total=len(core_project_nums), desc="Fetching publications") as pbar:
            for i in range(0, len(core_project_nums), self.batch_size):
                batch = core_project_nums[i:i + self.batch_size]

                offset = 0
                batch_limit = 500  # API limit per request

                while True:
                    request_data = {
                        "criteria": {
                            "core_project_nums": batch
                        },
                        "offset": offset,
                        "limit": batch_limit
                    }

                    try:
                        response = requests.post(
                            self.publications_url,
                            json=request_data,
                            headers={"Content-Type": "application/json"}
                        )
                        response.raise_for_status()

                        data = response.json()
                        results = data.get('results', [])
                        total = data.get('meta', {}).get('total', 0)

                        if not results:
                            break

                        # Save grant-publication relationships
                        for pub in results:
                            core_project_num = pub.get('coreproject')
                            pmid = pub.get('pmid')

                            if core_project_num and pmid:
                                self.db_manager.insert_grant_publication(
                                    core_project_num, pmid
                                )
                                all_publications.append(pub)
                                total_publications += 1

                        offset += len(results)

                        # If we've fetched all results for this batch, break
                        if offset >= total:
                            break

                        time.sleep(self.request_delay)

                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching publications for batch: {e}")
                        break

                pbar.update(len(batch))

        print(f"Successfully fetched {total_publications} publications")
        return all_publications


class PubMedAPI:
    """Handles PubMed API interactions"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.batch_size = 200  # PubMed recommends max 200 IDs per request
        self.request_delay = 0.34  # NCBI rate limit: 3 requests per second

    def fetch_papers_metadata(self, pmids: List[int]) -> Dict:
        """Fetch paper metadata from PubMed API"""
        # Filter out papers that already exist in database
        missing_pmids = self.db_manager.get_missing_papers(pmids)
        if not missing_pmids:
            print("All papers already in database, skipping download.")
            return {}

        print(f"Fetching metadata for {len(missing_pmids)} papers from PubMed...")

        total_fetched = 0
        with tqdm(total=len(missing_pmids), desc="Fetching papers") as pbar:
            for i in range(0, len(missing_pmids), self.batch_size):
                batch = missing_pmids[i:i + self.batch_size]
                pmid_str = ','.join(map(str, batch))

                params = {
                    'db': 'pubmed',
                    'retmode': 'json',
                    'id': pmid_str
                }

                try:
                    response = requests.get(self.esummary_url, params=params)
                    response.raise_for_status()

                    data = response.json()
                    result = data.get('result', {})

                    # Save each paper to database
                    for pmid_str, paper_data in result.items():
                        if pmid_str.isdigit():  # Skip non-PMID entries like 'uids'
                            pmid = int(pmid_str)
                            self.db_manager.insert_paper(
                                pmid,
                                json.dumps(paper_data)
                            )
                            total_fetched += 1

                    pbar.update(len(batch))
                    time.sleep(self.request_delay)

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching paper batch: {e}")
                    continue

        print(f"Successfully fetched {total_fetched} papers")
        return {"fetched": total_fetched}

    def get_all_pmids_from_database(self) -> List[int]:
        """Get all PMIDs from grant_publications table"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT pmid FROM grant_publications")
            return [row[0] for row in cursor.fetchall()]


class ICiteDatabase:
    """Handles iCite database operations for citation data"""

    def __init__(self, icite_db_path: str = "~/data/icite/latest/icite.db"):
        self.icite_db_path = Path(icite_db_path).expanduser()

    def get_citation_data(self, pmids: List[int]) -> Dict[int, Dict]:
        """Get citation data from iCite database"""
        citation_data = {}

        if not self.icite_db_path.exists():
            print(f"Warning: iCite database not found at {self.icite_db_path}")
            print("Using default values for citation data.")
            for pmid in pmids:
                citation_data[pmid] = {
                    'citation_count': 0,
                    'relative_citation_ratio': 1.0
                }
            return citation_data

        try:
            with sqlite3.connect(self.icite_db_path) as conn:
                cursor = conn.cursor()

                # Get citation data in batches
                batch_size = 1000
                for i in tqdm(range(0, len(pmids), batch_size), desc="Loading citation data"):
                    batch = pmids[i:i + batch_size]
                    placeholders = ','.join('?' * len(batch))

                    cursor.execute(f"""
                        SELECT pmid, citation_count, relative_citation_ratio
                        FROM papers
                        WHERE pmid IN ({placeholders})
                    """, batch)

                    for row in cursor.fetchall():
                        pmid, citation_count, rcr = row
                        citation_data[pmid] = {
                            'citation_count': citation_count or 0,
                            'relative_citation_ratio': rcr or 1.0
                        }

                # Fill missing PMIDs with default values
                for pmid in pmids:
                    if pmid not in citation_data:
                        citation_data[pmid] = {
                            'citation_count': 0,
                            'relative_citation_ratio': 1.0
                        }

        except Exception as e:
            print(f"Error accessing iCite database: {e}")
            print("Using default values for citation data.")
            for pmid in pmids:
                citation_data[pmid] = {
                    'citation_count': 0,
                    'relative_citation_ratio': 1.0
                }

        return citation_data


class EmbeddingGenerator:
    """Handles text embedding generation and dimension reduction"""

    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model_name = model_name
        self.model = None
        self.batch_size = 32

    def load_model(self):
        """Load the SentenceTransformer model"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str], output_path: str = "./gpa.embedding.npy") -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self.load_model()

        print(f"Generating embeddings for {len(texts)} texts...")

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)

        all_embeddings = np.vstack(embeddings)

        print(f"Saving embeddings to {output_path}")
        np.save(output_path, all_embeddings)

        return all_embeddings

    def reduce_dimensions(self, embeddings: np.ndarray, output_path: str = "./gpa.embd.npy") -> np.ndarray:
        """Reduce embeddings to 2D using t-SNE"""
        print("Reducing dimensions using t-SNE...")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=42,
            verbose=True,
            n_jobs=1
        )

        reduced_embeddings = tsne.fit(embeddings)

        print(f"Saving 2D embeddings to {output_path}")
        np.save(output_path, reduced_embeddings)

        return reduced_embeddings


class OutputGenerator:
    """Generates final output files: gpa.points.tsv and gpa.edges.json"""

    def __init__(self, db_manager: DatabaseManager, icite_db: ICiteDatabase):
        self.db_manager = db_manager
        self.icite_db = icite_db

    def parse_date(self, date_str: str, fiscal_year: int = None) -> str:
        """Parse date string to YYYY-MM-DD format using dateparser for robust parsing"""
        if not date_str or pd.isna(date_str):
            if fiscal_year:
                return f"{fiscal_year}-01-01"
            return "1900-01-01"

        try:
            # Use dateparser for robust date parsing (handles formats like "2021 Sep", etc.)
            parsed_date = dateparser.parse(str(date_str))
            if parsed_date:
                return parsed_date.strftime('%Y-%m-%d')
            
            # If dateparser fails, try manual parsing as fallback
            for fmt in ['%Y-%m-%d', '%Y-%m', '%Y', '%m/%d/%Y', '%m-%d-%Y']:
                try:
                    parsed_date = datetime.strptime(str(date_str), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # If all parsing fails, use fiscal year
            if fiscal_year:
                return f"{fiscal_year}-01-01"
            return "2020-01-01"

        except Exception:
            if fiscal_year:
                return f"{fiscal_year}-01-01"
            return "2020-01-01"

    def generate_author_id(self, author_name: str) -> str:
        """Generate a hash ID for an author"""
        return hashlib.md5(author_name.encode()).hexdigest()[:12]

    def create_text_representations(self, grant_df: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
        """Create text representations for grants, papers, and authors"""
        items = []
        texts = []

        # Get all data from database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get grants data
            grant_pids = grant_df['pid'].tolist()
            placeholders = ','.join('?' * len(grant_pids))
            cursor.execute(f"""
                SELECT core_project_num, metadata
                FROM grants
                WHERE core_project_num IN ({placeholders})
            """, grant_pids)

            grants_data = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

            # Process grants
            print("Processing grants...")
            for _, row in tqdm(grant_df.iterrows(), total=len(grant_df), desc="Processing grants"):
                pid = row['pid']
                grant_data = grants_data.get(pid, {})

                # Create text representation
                fiscal_year = grant_data.get('fiscal_year', '')
                agency = self._extract_agency_abbreviation(grant_data.get('agency_ic_admin', ''))
                title = grant_data.get('project_title', '')
                text = f"{title}"

                items.append({
                    'type': 'grant',
                    'pid': pid,
                    'data': grant_data,
                    'row_data': row
                })
                texts.append(text)

            # Get papers data
            cursor.execute("SELECT pmid, metadata FROM papers")
            papers_data = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

            # Process papers
            print("Processing papers...")
            processed_papers = set()
            for pmid, paper_data in tqdm(papers_data.items(), desc="Processing papers"):
                if pmid not in processed_papers:
                    source = paper_data.get('source', '')
                    title = paper_data.get('title', '')
                    text = f"{source} | {title}"

                    items.append({
                        'type': 'paper',
                        'pid': pmid,
                        'data': paper_data
                    })
                    texts.append(text)
                    processed_papers.add(pmid)

            # Process authors
            print("Processing authors...")
            processed_authors = {}
            for pmid, paper_data in tqdm(papers_data.items(), desc="Processing authors"):
                authors = paper_data.get('authors', [])
                for author in authors:
                    if isinstance(author, dict):
                        author_name = author.get('name', '')
                        if author_name and author_name not in processed_authors:
                            source = paper_data.get('source', '')
                            title = paper_data.get('title', '')
                            text = f"{source} | {author_name} | {title}"

                            items.append({
                                'type': 'author',
                                'pid': self.generate_author_id(author_name),
                                'data': {'name': author_name, 'sample_paper': paper_data}
                            })
                            texts.append(text)
                            processed_authors[author_name] = True

        return items, texts

    def create_gpa_points_tsv(self, items: List[Dict], embeddings_2d: np.ndarray,
                             output_path: str = "gpa.points.tsv") -> pd.DataFrame:
        """Create gpa.points.tsv with all required columns"""
        print(f"Creating {output_path}...")

        # Get citation data for all papers
        paper_pmids = [item['pid'] for item in items if item['type'] == 'paper']
        citation_data = self.icite_db.get_citation_data(paper_pmids)

        rows = []
        for i, item in tqdm(enumerate(items), total=len(items), desc="Creating points TSV"):
            item_type = item['type']
            pid = item['pid']
            data = item['data']

            # Basic row data
            row = {
                'pid': pid,
                'x': embeddings_2d[i, 0],
                'y': embeddings_2d[i, 1]
            }

            if item_type == 'grant':
                # Grant data
                row_data = item.get('row_data', {})
                row.update({
                    'date': self.parse_date(
                        data.get('project_start_date'),
                        data.get('fiscal_year')
                    ),
                    'journal': self._extract_agency_abbreviation(data.get('agency_ic_admin', '')),
                    'title': data.get('project_title', ''),
                    'mesh_terms': self._create_grant_mesh_terms(data, row_data),
                    'citation_count': data.get('award_amount', 0),
                    'size': math.sqrt(data.get('award_amount', 0)) / 100 if data.get('award_amount', 0) > 0 else 1,
                    'color': '#ffe100'
                })

            elif item_type == 'paper':
                # Paper data
                cit_data = citation_data.get(pid, {'citation_count': 0, 'relative_citation_ratio': 1.0})
                row.update({
                    'date': self.parse_date(data.get('pubdate')),
                    'journal': data.get('source', ''),
                    'title': data.get('title', ''),
                    'mesh_terms': '',
                    'citation_count': cit_data['citation_count'],
                    'size': cit_data['relative_citation_ratio'],
                    'color': '#f84848'
                })

            elif item_type == 'author':
                # Author data
                sample_paper = data.get('sample_paper', {})
                row.update({
                    'date': self.parse_date(sample_paper.get('pubdate')),
                    'journal': 'Author',
                    'title': data.get('name', ''),
                    'mesh_terms': '',
                    'citation_count': 1,
                    'size': 5,
                    'color': '#1196fc'
                })

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved {len(df)} points to {output_path}")

        return df

    def _extract_agency_abbreviation(self, agency_data) -> str:
        """Extract agency abbreviation from agency_ic_admin data"""
        if isinstance(agency_data, dict):
            return agency_data.get('abbreviation', '')
        elif isinstance(agency_data, str):
            return agency_data
        else:
            return ''

    def _create_grant_mesh_terms(self, grant_data: Dict, row_data: Dict) -> str:
        """Create mesh_terms string for grants"""
        terms = []

        # Add pref_terms
        pref_terms = row_data.get('pref_terms', '') if hasattr(row_data, 'get') else ''
        if pref_terms:
            terms.append(str(pref_terms))

        # Add spending_categories_desc
        spending_desc = row_data.get('spending_categories_desc', '') if hasattr(row_data, 'get') else ''
        if spending_desc:
            terms.append(str(spending_desc))

        # Add org_name
        org_name = grant_data.get('organization', {}).get('org_name', '')
        if org_name:
            terms.append(org_name)

        return ';'.join(terms)

    def create_gpa_edges_json(self, output_path: str = "gpa.edges.json") -> Dict:
        """Create gpa.edges.json with network relationships"""
        print(f"Creating {output_path}...")

        edges = []

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get grant-paper edges
            cursor.execute("SELECT core_project_num, pmid FROM grant_publications")
            grant_paper_edges = cursor.fetchall()

            print("Processing grant-paper edges...")
            for core_project_num, pmid in tqdm(grant_paper_edges, desc="Grant-paper edges"):
                edges.append({
                    'source': core_project_num,
                    'target': pmid,
                    'type': 'grant-paper'
                })

            # Get author-paper edges
            cursor.execute("SELECT pmid, metadata FROM papers")
            print("Processing author-paper edges...")
            for pmid, metadata_str in tqdm(cursor.fetchall(), desc="Author-paper edges"):
                metadata = json.loads(metadata_str)
                authors = metadata.get('authors', [])

                for author in authors:
                    if isinstance(author, dict):
                        author_name = author.get('name', '')
                        if author_name:
                            author_id = self.generate_author_id(author_name)
                            edges.append({
                                'source': author_id,
                                'target': pmid,
                                'type': 'author-paper'
                            })

        edges_data = {
            'edges': edges,
            'stats': {
                'total_edges': len(edges),
                'grant_paper_edges': len([e for e in edges if e['type'] == 'grant-paper']),
                'author_paper_edges': len([e for e in edges if e['type'] == 'author-paper'])
            }
        }

        with open(output_path, 'w') as f:
            json.dump(edges_data, f, indent=2)

        print(f"Saved {len(edges)} edges to {output_path}")
        return edges_data

    def create_metadata_tsv(self, items: List[Dict], texts: List[str],
                           output_path: str = "gpa.metadata.tsv") -> pd.DataFrame:
        """Create gpa.metadata.tsv with item information and text representations"""
        print(f"Creating {output_path}...")

        rows = []
        for item, text in tqdm(zip(items, texts), total=len(items), desc="Creating metadata TSV"):
            rows.append({
                'pid': item['pid'],
                'type': item['type'],
                'text_representation': text
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved metadata for {len(df)} items to {output_path}")

        return df


class NetworkBuilder:
    """Builds network data structure from grants, publications, and authors"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def build_network(self) -> Dict:
        """Build network of grants, publications, and authors"""
        print("Building network data structure...")

        nodes = {
            'grants': {},
            'papers': {},
            'authors': {}
        }
        edges = []

        # Get all data from database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Load grants
            cursor.execute("SELECT core_project_num, metadata FROM grants")
            for core_project_num, metadata_str in tqdm(cursor.fetchall(), desc="Processing grants"):
                metadata = json.loads(metadata_str)
                nodes['grants'][core_project_num] = {
                    'id': core_project_num,
                    'type': 'grant',
                    'title': metadata.get('project_title', ''),
                    'metadata': metadata
                }

            # Load papers
            cursor.execute("SELECT pmid, metadata FROM papers")
            for pmid, metadata_str in tqdm(cursor.fetchall(), desc="Processing papers"):
                metadata = json.loads(metadata_str)
                authors = metadata.get('authors', [])

                nodes['papers'][pmid] = {
                    'id': pmid,
                    'type': 'paper',
                    'title': metadata.get('title', ''),
                    'journal': metadata.get('source', ''),
                    'pubdate': metadata.get('pubdate', ''),
                    'authors': [author.get('name', '') for author in authors if isinstance(author, dict)],
                    'metadata': metadata
                }

                # Extract authors and create author nodes
                for author in authors:
                    if isinstance(author, dict):
                        author_name = author.get('name', '')
                        if author_name:
                            if author_name not in nodes['authors']:
                                nodes['authors'][author_name] = {
                                    'id': author_name,
                                    'type': 'author',
                                    'name': author_name,
                                    'papers': []
                                }
                            nodes['authors'][author_name]['papers'].append(pmid)

                            # Create author-paper edge
                            edges.append({
                                'source': author_name,
                                'target': pmid,
                                'type': 'author-paper'
                            })

            # Load grant-publication relationships
            cursor.execute("SELECT core_project_num, pmid FROM grant_publications")
            for core_project_num, pmid in tqdm(cursor.fetchall(), desc="Processing grant-paper relationships"):
                # Create grant-paper edge
                edges.append({
                    'source': core_project_num,
                    'target': pmid,
                    'type': 'grant-paper'
                })

        # Create summary statistics
        stats = {
            'total_grants': len(nodes['grants']),
            'total_papers': len(nodes['papers']),
            'total_authors': len(nodes['authors']),
            'total_edges': len(edges),
            'grant_paper_edges': len([e for e in edges if e['type'] == 'grant-paper']),
            'author_paper_edges': len([e for e in edges if e['type'] == 'author-paper'])
        }

        print(f"Network built successfully:")
        print(f"  Grants: {stats['total_grants']}")
        print(f"  Papers: {stats['total_papers']}")
        print(f"  Authors: {stats['total_authors']}")
        print(f"  Total edges: {stats['total_edges']}")

        network = {
            'nodes': nodes,
            'edges': edges,
            'stats': stats
        }

        return network

    def save_network_to_json(self, network: Dict, output_path: str = "network.json"):
        """Save network to JSON file"""
        print(f"Saving network to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(network, f, indent=2)
        print(f"Network saved to {output_path}")


class GrantPublicationNetworkBuilder:
    """Main class that orchestrates the entire pipeline"""

    def __init__(self, db_path: str = "./grants.db", icite_db_path: str = "~/data/icite/latest/icite.db",
                 embedding_model: str = "google/embeddinggemma-300m"):
        self.db_manager = DatabaseManager(db_path)
        self.nih_api = NIHReporterAPI(self.db_manager)
        self.pubmed_api = PubMedAPI(self.db_manager)
        self.network_builder = NetworkBuilder(self.db_manager)
        self.icite_db = ICiteDatabase(icite_db_path)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.output_generator = OutputGenerator(self.db_manager, self.icite_db)

    def read_grant_list(self, file_path: str, pid_column: str = "pid") -> List[str]:
        """Read grant list from TSV file"""
        try:
            df = pd.read_csv(file_path, sep='\t')
            if pid_column not in df.columns:
                raise ValueError(f"Column '{pid_column}' not found in {file_path}. Available columns: {list(df.columns)}")

            grant_list = df[pid_column].dropna().unique().tolist()
            print(f"Read {len(grant_list)} unique grants from {file_path}")
            return grant_list

        except Exception as e:
            print(f"Error reading grant list from {file_path}: {e}")
            raise

    def run(self, grant_file: str, pid_column: str = "pid", output_path: str = "network.json"):
        """Run the complete pipeline"""
        print("Starting Grant-Publication-Author Network Builder...")

        # Step 1: Read grant list
        print(f"\nStep 1: Reading grant list from {grant_file}...")
        grant_list = self.read_grant_list(grant_file, pid_column)

        # Step 2: Fetch grant metadata
        print(f"\nStep 2: Fetching grant metadata from NIH RePORTER...")
        self.nih_api.fetch_grants_metadata(grant_list)

        # Step 3: Fetch publications
        print(f"\nStep 3: Fetching publications from NIH RePORTER...")
        publications = self.nih_api.fetch_publications(grant_list)

        # Step 4: Fetch paper metadata
        print(f"\nStep 4: Fetching paper metadata from PubMed...")
        all_pmids = self.pubmed_api.get_all_pmids_from_database()
        if all_pmids:
            self.pubmed_api.fetch_papers_metadata(all_pmids)
        else:
            print("No publications found to fetch metadata for.")

        # Step 5: Build network
        print(f"\nStep 5: Building network data structure...")
        network = self.network_builder.build_network()

        # Step 6: Save network
        print(f"\nStep 6: Saving network...")
        self.network_builder.save_network_to_json(network, output_path)

        print("\n✅ Pipeline completed successfully!")
        print("\nOutput files:")
        print(f"  - Database: {self.db_manager.db_path}")
        print(f"  - Network: {output_path}")

    def generate_embeddings_and_output(self, grant_file: str, pid_column: str = "pid",
                                     embedding_path: str = "./gpa.embedding.npy",
                                     reduced_embedding_path: str = "./gpa.embd.npy",
                                     points_output: str = "gpa.points.tsv",
                                     edges_output: str = "gpa.edges.json",
                                     metadata_output: str = "gpa.metadata.tsv",
                                     skip_embeddings: bool = False,
                                     skip_reduction: bool = False):
        """Generate embeddings and create final output files"""
        print("Starting embedding generation and output creation pipeline...")

        # Step 1: Read grant list and create text representations
        print(f"\nStep 1: Reading grant list and creating text representations...")
        grant_df = pd.read_csv(grant_file, sep='\t')
        items, texts = self.output_generator.create_text_representations(grant_df)

        # Step 2: Create metadata TSV
        print(f"\nStep 2: Creating metadata file...")
        self.output_generator.create_metadata_tsv(items, texts, metadata_output)

        # Step 3: Generate embeddings (if not skipping)
        embeddings = None
        if not skip_embeddings and not Path(embedding_path).exists():
            print(f"\nStep 3: Generating text embeddings...")
            embeddings = self.embedding_generator.generate_embeddings(texts, embedding_path)
        else:
            if skip_embeddings:
                print(f"\nStep 3: Skipping embedding generation as requested")
            else:
                print(f"\nStep 3: Loading existing embeddings from {embedding_path}")
            embeddings = np.load(embedding_path)

        # Step 4: Reduce dimensions (if not skipping)
        embeddings_2d = None
        if not skip_reduction and not Path(reduced_embedding_path).exists():
            print(f"\nStep 4: Reducing dimensions with t-SNE...")
            embeddings_2d = self.embedding_generator.reduce_dimensions(embeddings, reduced_embedding_path)
        else:
            if skip_reduction:
                print(f"\nStep 4: Skipping dimension reduction as requested")
            else:
                print(f"\nStep 4: Loading existing 2D embeddings from {reduced_embedding_path}")
            embeddings_2d = np.load(reduced_embedding_path)

        # Step 5: Create gpa.points.tsv
        print(f"\nStep 5: Creating points TSV file...")
        points_df = self.output_generator.create_gpa_points_tsv(items, embeddings_2d, points_output)

        # Step 6: Create gpa.edges.json
        print(f"\nStep 6: Creating edges JSON file...")
        edges_data = self.output_generator.create_gpa_edges_json(edges_output)

        print("\n✅ Embedding and output generation pipeline completed successfully!")
        print("\nOutput files:")
        print(f"  - Embeddings: {embedding_path}")
        print(f"  - 2D Embeddings: {reduced_embedding_path}")
        print(f"  - Points: {points_output}")
        print(f"  - Edges: {edges_output}")
        print(f"  - Metadata: {metadata_output}")

        return {
            'items': items,
            'texts': texts,
            'embeddings': embeddings,
            'embeddings_2d': embeddings_2d,
            'points_df': points_df,
            'edges_data': edges_data
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build Grant-Publication-Author Network from NIH RePORTER data"
    )
    parser.add_argument(
        "--grant-file",
        default="grant-list.tsv",
        help="Path to TSV file containing grant list (default: grant-list.tsv)"
    )
    parser.add_argument(
        "--pid-column",
        default="pid",
        help="Column name containing core_project_num (default: pid)"
    )
    parser.add_argument(
        "--db-path",
        default="./grants.db",
        help="Path to SQLite database file (default: ./grants.db)"
    )
    parser.add_argument(
        "--output",
        default="network.json",
        help="Output path for network JSON file (default: network.json)"
    )
    parser.add_argument(
        "--icite-db",
        default="~/data/icite/latest/icite.db",
        help="Path to iCite database file for citation data (default: ~/data/icite/latest/icite.db)"
    )
    parser.add_argument(
        "--embedding-model",
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model for embeddings (default: google/embeddinggemma-300m)"
    )
    parser.add_argument(
        "--generate-outputs",
        action="store_true",
        help="Generate gpa.points.tsv and gpa.edges.json files with embeddings"
    )
    parser.add_argument(
        "--embedding-path",
        default="./gpa.embedding.npy",
        help="Path to save/load high-dimensional embeddings (default: ./gpa.embedding.npy)"
    )
    parser.add_argument(
        "--reduced-embedding-path",
        default="./gpa.embd.npy",
        help="Path to save/load 2D embeddings (default: ./gpa.embd.npy)"
    )
    parser.add_argument(
        "--points-output",
        default="gpa.points.tsv",
        help="Output path for points TSV file (default: gpa.points.tsv)"
    )
    parser.add_argument(
        "--edges-output",
        default="gpa.edges.json",
        help="Output path for edges JSON file (default: gpa.edges.json)"
    )
    parser.add_argument(
        "--metadata-output",
        default="gpa.metadata.tsv",
        help="Output path for metadata TSV file (default: gpa.metadata.tsv)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation and use existing embeddings"
    )
    parser.add_argument(
        "--skip-reduction",
        action="store_true",
        help="Skip dimension reduction and use existing 2D embeddings"
    )
    parser.add_argument(
        "--only-outputs",
        action="store_true",
        help="Only generate output files, skip data fetching pipeline"
    )

    args = parser.parse_args()

    try:
        builder = GrantPublicationNetworkBuilder(
            args.db_path,
            args.icite_db,
            args.embedding_model
        )

        if args.only_outputs:
            # Only generate output files
            builder.generate_embeddings_and_output(
                args.grant_file,
                args.pid_column,
                args.embedding_path,
                args.reduced_embedding_path,
                args.points_output,
                args.edges_output,
                args.metadata_output,
                args.skip_embeddings,
                args.skip_reduction
            )
        elif args.generate_outputs:
            # Run full pipeline including output generation
            builder.run(args.grant_file, args.pid_column, args.output)
            builder.generate_embeddings_and_output(
                args.grant_file,
                args.pid_column,
                args.embedding_path,
                args.reduced_embedding_path,
                args.points_output,
                args.edges_output,
                args.metadata_output,
                args.skip_embeddings,
                args.skip_reduction
            )
        else:
            # Run basic pipeline only
            builder.run(args.grant_file, args.pid_column, args.output)

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()