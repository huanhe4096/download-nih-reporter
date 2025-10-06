#!/usr/bin/env python3
"""
NIH RePORTER Data Converter

Converts cached NIH RePORTER JSON data to TSV format.
Reads from ./cache/<fiscal_year>/<org_state>/<batch_number>.json
Outputs to TSV file with selected fields.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm


class NIHDataConverter:
    def __init__(self, cache_dir: str = "./cache", output_file: str = "./nih-reporter-grants.tsv"):
        self.cache_dir = Path(cache_dir)
        self.output_file = Path(output_file)
        self.seen_project_nums: Set[str] = set()
        self.seen_core_project_nums: Set[str] = set()
        self.skipped_project_nums = []
        self.skipped_core_project_nums = []

        # TSV column headers
        self.headers = [
            "project_num",
            "core_project_num",
            "fiscal_year",
            "org_name",
            "award_amount",
            "principal_investigators",
            "agency_ic_admin",
            "pref_terms",
            "abstract_text",
            "project_title",
            "spending_categories_desc",
            "date_added",
            "project_start_date",
            "project_end_date",
            "award_type",
            "activity_code"
        ]


    def discover_json_files(self) -> List[Path]:
        """Discover all JSON files in the cache directory."""
        json_files = []

        if not self.cache_dir.exists():
            print(f"❌ Cache directory does not exist: {self.cache_dir}")
            return json_files

        # Pattern: ./cache/<fiscal_year>/<org_state>/<batch_number>.json
        for fiscal_year_dir in self.cache_dir.iterdir():
            if not fiscal_year_dir.is_dir():
                continue

            for org_state_dir in fiscal_year_dir.iterdir():
                if not org_state_dir.is_dir():
                    continue

                for json_file in org_state_dir.glob("*.json"):
                    json_files.append(json_file)

        return sorted(json_files)

    def count_total_projects(self, json_files: List[Path]) -> int:
        """Count total number of projects across all JSON files."""
        total_count = 0

        print("🔍 Counting total projects...")
        for json_file in tqdm(json_files, desc="Scanning files"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results = data.get('results', [])
                    total_count += len(results)
            except Exception as e:
                print(f"⚠️  Error reading {json_file}: {e}")
                continue

        return total_count

    def clean_text(self, text: str) -> str:
        """Clean text by replacing tabs and newlines with spaces."""
        if not text:
            return ""
        # Replace tabs and newlines with spaces, then clean up multiple spaces
        cleaned = re.sub(r'[\t\n\r]+', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def extract_date(self, date_str: str) -> str:
        """Extract YYYY-MM-DD from date string."""
        if not date_str:
            return ""
        # Take first 10 characters (YYYY-MM-DD)
        return str(date_str)[:10]

    def extract_project_data(self, project: Dict) -> Optional[Dict]:
        """Extract required fields from a project record."""
        try:
            # Extract project number and core project number
            project_num = project.get('project_num', '')
            core_project_num = project.get('core_project_num', '')

            # Check for duplicate project_num
            if project_num in self.seen_project_nums:
                self.skipped_project_nums.append(project_num)
                return None

            # Check for duplicate core_project_num (use project_num if core is empty)
            check_core = core_project_num if core_project_num else project_num
            if check_core in self.seen_core_project_nums:
                self.skipped_core_project_nums.append([project_num, core_project_num])
                return None

            # Add to seen sets
            self.seen_project_nums.add(project_num)
            self.seen_core_project_nums.add(check_core)

            # Extract organization name
            org_name = ""
            organization = project.get('organization', {})
            if organization:
                org_name = organization.get('org_name', '')

            # Extract fiscal year
            fiscal_year = project.get('fiscal_year', '')

            # Extract award amount
            award_amount = project.get('award_amount', '')

            # Extract principal investigators
            pis = project.get('principal_investigators', [])
            pi_names = []
            if pis:
                for pi in pis:
                    full_name = pi.get('full_name', '')
                    if full_name:
                        pi_names.append(full_name)
            principal_investigators = ', '.join(pi_names)

            # Extract agency IC admin abbreviation
            agency_ic_admin = ""
            agency_ic = project.get('agency_ic_admin', {})
            if agency_ic:
                agency_ic_admin = agency_ic.get('abbreviation', '')

            # Extract and clean text fields
            pref_terms = self.clean_text(project.get('pref_terms', ''))
            abstract_text = self.clean_text(project.get('abstract_text', ''))
            project_title = self.clean_text(project.get('project_title', ''))
            spending_categories_desc = self.clean_text(project.get('spending_categories_desc', ''))

            # Extract dates
            date_added = self.extract_date(project.get('date_added', ''))
            project_start_date = self.extract_date(project.get('project_start_date', ''))
            project_end_date = self.extract_date(project.get('project_end_date', ''))

            # Extract award type and activity code
            award_type = project.get('award_type', '')
            activity_code = project.get('activity_code', '')

            return {
                'project_num': project_num,
                'core_project_num': core_project_num,
                'fiscal_year': fiscal_year,
                'org_name': org_name,
                'award_amount': award_amount,
                'principal_investigators': principal_investigators,
                'agency_ic_admin': agency_ic_admin,
                'pref_terms': pref_terms,
                'abstract_text': abstract_text,
                'project_title': project_title,
                'spending_categories_desc': spending_categories_desc,
                'date_added': date_added,
                'project_start_date': project_start_date,
                'project_end_date': project_end_date,
                'award_type': award_type,
                'activity_code': activity_code
            }

        except Exception as e:
            print(f"⚠️  Error extracting project data: {e}")
            return None

    def convert_to_tsv(self):
        """Convert all cached JSON data to TSV format with streaming output."""
        print("🚀 Starting conversion to TSV...")

        # Discover JSON files
        json_files = self.discover_json_files()
        if not json_files:
            print("❌ No JSON files found in cache directory")
            return

        print(f"📁 Found {len(json_files)} JSON files")

        # Count total projects for progress tracking
        total_projects = self.count_total_projects(json_files)
        print(f"📊 Total projects to process: {total_projects:,}")

        # Open output file for streaming
        processed_count = 0
        written_count = 0

        print(f"✍️  Writing to: {self.output_file}")

        with open(self.output_file, 'w', newline='', encoding='utf-8') as tsv_file:
            # Create TSV writer
            writer = csv.DictWriter(tsv_file, fieldnames=self.headers, delimiter='\t')

            # Write header
            writer.writeheader()

            # Process files with progress bar
            with tqdm(total=total_projects, desc="Converting projects", unit="projects") as pbar:
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        results = data.get('results', [])

                        for project in results:
                            processed_count += 1
                            pbar.update(1)

                            # Extract project data
                            project_data = self.extract_project_data(project)

                            if project_data:
                                # Write to TSV
                                writer.writerow(project_data)
                                written_count += 1

                    except Exception as e:
                        print(f"\n⚠️  Error processing {json_file}: {e}")
                        continue

        # Final statistics
        print(f"\n🎉 Conversion completed!")
        print(f"📊 Statistics:")
        print(f"   - Total projects processed: {processed_count:,}")
        print(f"   - Projects written to TSV: {written_count:,}")
        print(f"   - Duplicates skipped (project_num): {len(self.skipped_project_nums):,}")
        print(f"   - Duplicates skipped (core_project_num): {len(self.skipped_core_project_nums):,}")
        print(f"   - Total duplicates skipped: {len(self.skipped_project_nums) + len(self.skipped_core_project_nums):,}")
        print(f"   - Output file: {self.output_file}")
        print(f"   - File size: {self.output_file.stat().st_size / (1024*1024):.1f} MB")
        
        # save skipped project numbers to file
        with open('skipped_project_nums.txt', 'w') as f:
            for project_num in self.skipped_project_nums:
                f.write(f"{project_num}\n")
        print(f"* saved skipped project numbers to file")
        
        # save skipped core project numbers to file
        with open('skipped_core_project_nums.txt', 'w') as f:
            for project_num, core_project_num in self.skipped_core_project_nums:
                f.write(f"{project_num}\t{core_project_num}\n")
        print(f"* saved skipped core project numbers to file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NIH RePORTER cached JSON data to TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Directory containing cached JSON files")
    parser.add_argument("--output", type=str, default="./nih-reporter-grants.tsv",
                       help="Output TSV file path")

    args = parser.parse_args()

    converter = NIHDataConverter(cache_dir=args.cache_dir, output_file=args.output)
    print(f"📄 NIH RePORTER Data Converter")
    print(f"📁 Cache directory: {converter.cache_dir}")
    print(f"📊 Output file: {converter.output_file}")
    print("-" * 50)

    try:
        converter.convert_to_tsv()
    except KeyboardInterrupt:
        print("\n⚠️  Conversion interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise