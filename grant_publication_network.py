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

    # Specify custom column name and output
    python grant_publication_network.py --grant-file my_grants.tsv --pid-column project_id --output my_network.json

    # Use custom database path
    python grant_publication_network.py --db-path ./my_grants.db

Features:
- Automatically handles API rate limiting
- Uses SQLite database for caching to avoid re-downloading
- Processes data in batches for efficiency
- Shows progress bars for all long-running operations
- Creates network with grants, papers, and authors as nodes
- Generates edges for grant-paper and author-paper relationships
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import requests
import pandas as pd
from tqdm import tqdm


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

    def __init__(self, db_path: str = "./grants.db"):
        self.db_manager = DatabaseManager(db_path)
        self.nih_api = NIHReporterAPI(self.db_manager)
        self.pubmed_api = PubMedAPI(self.db_manager)
        self.network_builder = NetworkBuilder(self.db_manager)

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

    args = parser.parse_args()

    try:
        builder = GrantPublicationNetworkBuilder(args.db_path)
        builder.run(args.grant_file, args.pid_column, args.output)
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()