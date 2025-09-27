#!/usr/bin/env python3
"""
NIH RePORTER Projects Downloader

Downloads all projects from NIH RePORTER API with caching and rate limiting.
Splits requests by Fiscal Year + Org State to manage rate limits.
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


class NIHProjectsDownloader:
    def __init__(self, 
        cache_dir: str = "./cache",
        year_start: int = 1985,
        year_end: int = 2025,
    ):
        self.api_url = "https://api.reporter.nih.gov/v2/projects/search"
        self.cache_dir = Path(cache_dir)
        self.batch_size = 500
        self.rate_limit_delay = 2.0  # 2 seconds delay between requests

        # US states and territories for org_states
        self.states = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "DC", "PR", "VI", "GU", "AS", "MP"
        ]

        # Fiscal years range (NIH RePORTER typically has data from 1985 onwards)
        self.fiscal_years = list(range(year_start, year_end + 1))  # Adjust end year as needed

    def ensure_cache_dir(self, fiscal_year: int, org_state: str) -> Path:
        """Ensure cache directory exists for given fiscal year and org state."""
        cache_path = self.cache_dir / str(fiscal_year) / org_state
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_cache_file_path(self, fiscal_year: int, org_state: str, batch_number: int) -> Path:
        """Get cache file path for given parameters."""
        cache_path = self.ensure_cache_dir(fiscal_year, org_state)
        return cache_path / f"{batch_number}.json"

    def is_cached(self, fiscal_year: int, org_state: str, batch_number: int) -> bool:
        """Check if data is already cached."""
        cache_file = self.get_cache_file_path(fiscal_year, org_state, batch_number)
        return cache_file.exists()

    def load_from_cache(self, fiscal_year: int, org_state: str, batch_number: int) -> Dict:
        """Load data from cache file."""
        cache_file = self.get_cache_file_path(fiscal_year, org_state, batch_number)
        with open(cache_file, 'r') as f:
            return json.load(f)

    def save_to_cache(self, fiscal_year: int, org_state: str, batch_number: int, data: Dict):
        """Save data to cache file."""
        cache_file = self.get_cache_file_path(fiscal_year, org_state, batch_number)
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def make_api_request(self, fiscal_year: int, org_state: str, offset: int = 0, limit: int = 500) -> Dict:
        """Make API request to NIH RePORTER."""
        request_body = {
            "criteria": {
                "use_relevance": True,
                "fiscal_years": [fiscal_year],
                "org_states": [org_state]
            },
            "offset": offset,
            "limit": limit
        }

        try:
            response = requests.post(self.api_url, json=request_body, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request failed for FY {fiscal_year}, {org_state}, offset {offset}: {e}")
            raise

    def get_total_count(self, fiscal_year: int, org_state: str) -> int:
        """Get total count for fiscal year + org state combination."""
        batch_number = 0

        if self.is_cached(fiscal_year, org_state, batch_number):
            data = self.load_from_cache(fiscal_year, org_state, batch_number)
            return data.get('meta', {}).get('total', 0)

        # Make first request to get total count
        data = self.make_api_request(fiscal_year, org_state, offset=0, limit=self.batch_size)
        self.save_to_cache(fiscal_year, org_state, batch_number, data)

        # Sleep for rate limiting
        time.sleep(self.rate_limit_delay)

        return data.get('meta', {}).get('total', 0)

    def download_for_combination(self, fiscal_year: int, org_state: str) -> int:
        """Download all projects for a fiscal year + org state combination."""
        total_count = self.get_total_count(fiscal_year, org_state)

        if total_count == 0:
            return 0

        # Calculate number of batches needed
        num_batches = (total_count + self.batch_size - 1) // self.batch_size

        downloaded_count = 0

        # Progress bar for this combination
        desc = f"FY {fiscal_year} - {org_state}"
        with tqdm(total=total_count, desc=desc, unit="projects") as pbar:
            for batch_num in range(num_batches):
                if self.is_cached(fiscal_year, org_state, batch_num):
                    # Load from cache and count projects
                    cached_data = self.load_from_cache(fiscal_year, org_state, batch_num)
                    batch_count = len(cached_data.get('results', []))
                    downloaded_count += batch_count
                    pbar.update(batch_count)
                    continue

                # Calculate offset for this batch
                offset = batch_num * self.batch_size

                try:
                    # Make API request
                    data = self.make_api_request(fiscal_year, org_state, offset, self.batch_size)

                    # Save to cache
                    self.save_to_cache(fiscal_year, org_state, batch_num, data)

                    # Update counters
                    batch_count = len(data.get('results', []))
                    downloaded_count += batch_count
                    pbar.update(batch_count)

                    # Rate limiting delay (only for non-cached requests)
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    print(f"\nError downloading batch {batch_num} for FY {fiscal_year}, {org_state}: {e}")
                    break

        return downloaded_count

    def download_all_projects(self):
        """Download all projects from NIH RePORTER."""
        print("Starting NIH RePORTER Projects Download")
        print(f"Expected total projects: ~2.9 million")
        print(f"Fiscal years: {self.fiscal_years[0]} - {self.fiscal_years[-1]}")
        print(f"States/territories: {len(self.states)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Rate limit delay: {self.rate_limit_delay}s")
        print("-" * 50)

        total_downloaded = 0
        total_combinations = len(self.fiscal_years) * len(self.states)

        # Overall progress tracking
        print(f"Processing {total_combinations} combinations (Fiscal Year × State)")

        for fiscal_year in self.fiscal_years:
            for org_state in self.states:
                try:
                    print(f"\n📅 Processing FY {fiscal_year} - {org_state}")

                    # Download for this combination
                    downloaded = self.download_for_combination(fiscal_year, org_state)
                    total_downloaded += downloaded

                    if downloaded > 0:
                        print(f"✅ Downloaded {downloaded:,} projects for FY {fiscal_year} - {org_state}")
                    else:
                        print(f"ℹ️  No projects found for FY {fiscal_year} - {org_state}")

                except KeyboardInterrupt:
                    print(f"\n⚠️  Download interrupted by user")
                    print(f"Total downloaded so far: {total_downloaded:,} projects")
                    return total_downloaded

                except Exception as e:
                    print(f"❌ Error processing FY {fiscal_year} - {org_state}: {e}")
                    continue

        print(f"\n🎉 Download completed!")
        print(f"Total projects downloaded: {total_downloaded:,}")
        return total_downloaded

    def get_download_statistics(self):
        """Get statistics about downloaded data."""
        total_projects = 0
        total_combinations = 0
        completed_combinations = 0

        for fiscal_year in tqdm(self.fiscal_years, desc="Processing fiscal years"):
            for org_state in self.states:
                total_combinations += 1

                # Check if any data exists for this combination
                if self.is_cached(fiscal_year, org_state, 0):
                    completed_combinations += 1
                    total_count = self.get_total_count(fiscal_year, org_state)
                    total_projects += total_count

        print(f"🔍 Download Statistics:")
        print(f"- Total combinations: {total_combinations}")
        print(f"- Completed combinations: {completed_combinations}")
        print(f"- Total projects found: {total_projects:,}")
        print(f"- Progress: {completed_combinations/total_combinations*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download NIH RePORTER projects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # action: stats, download
    parser.add_argument("action", type=str, default="stat", choices=["stat", "download"])
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--year_start", type=int, default=1985)
    parser.add_argument("--year_end", type=int, default=2025)
    args = parser.parse_args()
    
    downloader = NIHProjectsDownloader(
        cache_dir=args.cache_dir,
        year_start=args.year_start,
        year_end=args.year_end,
    )

    # Show current statistics if any data exists
    try:
        if args.action == "stat":
            downloader.get_download_statistics()
            
        elif args.action == "download":
            # Start download
            downloader.download_all_projects()
            
        else:
            print(f"Invalid action: {args.action}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise