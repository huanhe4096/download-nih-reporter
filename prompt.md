# Prompt for generating the downloader

I want to implement a python script that can help download all the NIH RePORTER grant and publication data. First of all, I want to download all the projects from NIH RePORTER. Please follow these requirements:

1. Due to the rate limit, please split the request into Fiscal Year + Org State, and batch into 500 per request. 
2. Each request must be saved locally to avoid resend. Create a local folder to save the cache data `./cache/<fiscal_year>/<org_state>/<batch_number>.json`. So before sending a request, check the cache folder first. For example, if `./cache/1988/CA/0.json` is already there, just read that file and get the total number and then send split requests.
3. For the first request to a Fiscal Year + Org State, check the total number of records and then plan the offset and limit.
4. The request endpoint for project is `https://api.reporter.nih.gov/v2/projects/search`
5. The request body is a JSON object, looks like:

```json
{
  "criteria": {
    "use_relevance": true,
    "fiscal_years": [1985],
    "org_states": [
      "CA"
    ]
  },
  "offset": 0,
  "limit": 500
}
```

6. Due to the rate limit, please sleep 1 second after a request if the data is not cached.
7. Display for each year+state combination, use tqdm to show an estimation of how long it takes, and also show how many has been downloaded, as we know there are about 2.9 million projects, it will take a long time to download and process.

# Prompt for parsing the downloaded data file to tabular format

I want to create a python script that read the NIH RePORTER grant data from a given path and convert to a .tsv file. Please follow these requirements:

1. All raw data are saved in a local folder in the following format `./cache/<fiscal_year>/<org_state>/<batch_number>.json`.
2. Each json contains a part of the grant data, for example:
```json
{
  "meta": {
    "search_id": "ER-AuCBZAk-N8muffzyktA",
    "total": 155,
    "offset": 0,
    "limit": 1,
    "sort_field": null,
    "sort_order": "ASC",
    "sorted_by_relevance": false,
    "properties": {
      "URL": "https:/reporter.nih.gov/search/ER-AuCBZAk-N8muffzyktA/projects"
    }
  },
  "results": []
}
```
The `results` is a list of grant data, and it has many attributes. For example:

```json
{
      "appl_id": 10497514,
      "subproject_id": null,
      "fiscal_year": 2023,
      "project_num": "1R15DE032063-01A1",
      "project_serial_num": "DE032063",
      "organization": {
        "org_name": "MARSHALL UNIVERSITY",
        "city": null,
        "country": null,
        "org_city": "Huntington",
        "org_country": "UNITED STATES",
        "org_state": "WV",
        "org_state_name": null,
        "dept_type": "PHARMACOLOGY",
        "fips_country_code": null,
        "org_duns": [
          "036156615"
        ],
        "org_ueis": [
          "HH1NQ1B5MPV3"
        ],
        "primary_duns": "036156615",
        "primary_uei": "HH1NQ1B5MPV3",
        "org_fips": "US",
        "org_ipf_code": "4842001",
        "org_zipcode": "257550002",
        "external_org_id": 4842001
      },
      "award_type": "1",
      "activity_code": "R15",
      "award_amount": 444000,
      "is_active": true,
      "project_num_split": {
        "appl_type_code": "1",
        "activity_code": "R15",
        "ic_code": "DE",
        "serial_num": "032063",
        "support_year": "01",
        "full_support_year": "01A1",
        "suffix_code": "A1"
      },
      "principal_investigators": [
        {
          "profile_id": 10583324,
          "first_name": "A.R.M.",
          "middle_name": "Ruhul",
          "last_name": "Amin",
          "is_contact_pi": true,
          "full_name": "A.R.M. Ruhul Amin",
          "title": "ASSOCIATE PROFESSOR (TENURED)"
        }
      ],
      "contact_pi_name": "AMIN, A.R.M. RUHUL",
      "program_officers": [
        {
          "first_name": "ZHONG",
          "middle_name": "",
          "last_name": "CHEN",
          "full_name": "ZHONG  CHEN"
        }
      ],
      "agency_ic_admin": {
        "code": "DE",
        "abbreviation": "NIDCR",
        "name": "National Institute of Dental and Craniofacial Research"
      },
      "agency_ic_fundings": [
        {
          "fy": 2023,
          "code": "DE",
          "name": "National Institute of Dental and Craniofacial Research",
          "abbreviation": "NIDCR",
          "total_cost": 94000,
          "direct_cost_ic": 63514,
          "indirect_cost_ic": 30486
        },
        {
          "fy": 2023,
          "code": "GM",
          "name": "National Institute of General Medical Sciences",
          "abbreviation": "NIGMS",
          "total_cost": 350000,
          "direct_cost_ic": 236486,
          "indirect_cost_ic": 113514
        }
      ],
      "cong_dist": "WV-01",
      "spending_categories": [
        108,
        132,
        216,
        232,
        276,
        556,
        701,
        729
      ],
      "project_start_date": "2023-08-01T00:00:00",
      "project_end_date": "2026-07-31T00:00:00",
      "organization_type": {
        "name": "SCHOOLS OF PHARMACY",
        "code": "10",
        "is_other": false
      },
      "geo_lat_lon": {
        "lon": -82.44721,
        "lat": 38.416466
      },
      "opportunity_number": "PAR-19-134",
      "full_study_section": {
        "srg_code": "ZRG1",
        "srg_flex": null,
        "sra_designator_code": "OTC",
        "sra_flex_code": "A",
        "group_code": "80",
        "name": "Special Emphasis Panel[ZRG1 OTC-A (80)]"
      },
      "award_notice_date": "2023-07-25T00:00:00",
      "is_new": false,
      "mechanism_code_dc": "RP",
      "core_project_num": "R15DE032063",
      "terms": "",
      "pref_terms": "AKT1 gene;Affect;Animal Model",
      "abstract_text": "",
      "project_title": "Targeting oncogenic pathways for chemoprevention of head and neck cancer by FLLL12",
      "phr_text": "",
      "spending_categories_desc": "Biotechnology; Cancer; Dental/Oral and Craniofacial Disease; Digestive Diseases; Genetics; Orphan Drug; Prevention; Rare Diseases",
      "agency_code": "NIH",
      "covid_response": null,
      "arra_funded": "N",
      "budget_start": "2023-08-01T00:00:00",
      "budget_end": "2026-07-31T00:00:00",
      "cfda_code": "93.121",
      "funding_mechanism": "Non-SBIR/STTR",
      "direct_cost_amt": 300000,
      "indirect_cost_amt": 144000,
      "project_detail_url": "https://reporter.nih.gov/project-details/10497514",
      "date_added": "2023-08-05T16:15:57"
    }
```

3. In this python script, I only need the following attributes

- project_num: the grant number. most of time this should be unique, but if it's duplicated in whole, skip the new one. count how many are skipped.
- core_project_num: the core number of this grant. Use this core_project_num to distinguish projects, if a core_project_num exists, we can skip the new one. count how many are skipped.
- fiscal_year: the fiscal_year number.
- organization.org_name: organization is a json object, I only need the `org_name`.
- award_amount: the amount of this award.
- principal_investigators: this is a list items, I only need the `full_name` for each PI, you can concat the full names using comma.
- agency_ic_admin.abbreviation: just need the abbreviation.
- pref_terms: it's the keywords of this grant, replace any tab symbol with a space.
- abstract_text: the abstract, also replace any tab (\t) or new line (\n) symbol with a space to avoid bug.
- project_title: the title of this grant
- spending_categories_desc: keep this attribute
- date_added: I only need the date in YYYY-MM-DD format, usually just the first 10 characters.
- project_start_date: only need the date in YYYY-MM-DD format.
- project_end_date: only need the date in YYYY-MM-DD format.
- award_type: the type of this award.
- activity_code: the activity code, e.g., R01, R15, K99, etc.

1. Due to the large size, show a progress bar.
2. Save the output .tsv file to `./nih-reporter-grants.tsv` by default if user does not provide path to output
3. Due to the large size of data files, stream output the result to the output file, avoid saving all files into memory before dumping into files.


# Prompt for generate semantic embeddings and 2d

I want to implement a Python script that read the tsv file and generate text embeddings and umap dimension reductions for visualization. Please follow these requirements:

1. Read the input data `./nih-reporter-grants.tsv` or user given input path to a tsv file.
2. For each grant in the tsv file, create a string using the following format:
   "{fiscal_year} | {agency_ic_admin} | {activity_code} | {project_title} | {abstract_text}"
3. Use SentenceTransformer to embed the string to text embedding, use the model `google/embeddinggemma-300m` or any user provided model slug.
4. Due the large size of dataset, please batch the embedding request to SentenceTransformer.
5. Save all the embedding to `./nih-reporter-grants.embedding.npy` file, which is a NumPy file. User can also specify where to save.
6. Once the high dimensional embedding is ready, user can reduce the embedding to 2d using umap (python umap-learn package). 
7. The 2d embedding can be saved into a `./nih-reporter-grants.embd.npy` file. User can also specify where to save.
8. Once the 2d embedding file is ready, user can merge the tsv file with the 2d embedding to a new file, e.g., `./grants.tsv`, or user specified file name.
9. In the merged tsv file, only need the following columns:
    - pid: core_project_num from the input tsv
    - date: project_start_date. if project_start_date is not available, use fiscal_year but convert to <fiscal_year>-01-01 for same format.
    - journal: agency_ic_admin
    - title: project_title from the input tsv
    - mesh_terms: pref_terms and spending_categories_desc from the input tsv. and add org_name at last. using ; to concat them.
    - x: the first dimension from 2d embedding
    - y: the second dimension from 2d embedding
    - size: sqrt(award_amount) / 100
    - citation_count: award_amount
    - color: based on the first item in the spending_categories_desc (split by ; and trim), use a color dictionary to generate color. In total I want about 20 colors (e.g, tab20 schema in matplotlib)
10. Each step, e.g., generate text embedding, dimension reduction, and merge, can run seperately. So we don't need to re-run embedding or dimension reduction again.
11. For each step, please show a progress bar, especiall generate text embedding and merge. for umap, turn on verbose.


# Prompt for downloading the relevant publications

I want to implemenet a Python script that reads a list of grants and download the related publications for creating network of grant-publication-authors. Please follow these requirements:

- For each step, try to add a tqdm to show a progress
- Read the input grant list `grant-list.tsv`, or user given input path to a tsv file. The `pid` column contains the `core_project_num` for a grant, or user given input column name.
- Get the grant metadata from NIH RePORTER API and save the tmp information into a local sqlite database, for example, `./grants.db`. Create a table, `grants`, which contains 2 columns: `core_project_num` and `metadata`.
- The request endpoint for project is `https://api.reporter.nih.gov/v2/projects/search`
- Due to the rate limit, please sleep 1 second after a request if the data is not cached.
- The request body is a JSON object, looks like:
```json
{
  "criteria": {
    "use_relevance": true,
    "project_nums": [
      "U18TR003807"
    ]
  },
  "offset": 0,
  "limit": 1
}
```
The response json contains a part of the grant data, for example:
```json
{
  "meta": {
    "search_id": "ER-AuCBZAk-N8muffzyktA",
    "total": 155,
    "offset": 0,
    "limit": 1,
    "sort_field": null,
    "sort_order": "ASC",
    "sorted_by_relevance": false,
    "properties": {
      "URL": "https:/reporter.nih.gov/search/ER-AuCBZAk-N8muffzyktA/projects"
    }
  },
  "results": []
}
```
The `results` contains a list of grant data. Each object in `results` is a grant. Just this JSON object into the `grants` table. Also, you can use this table as a local cache to check if a project or grant has been downloaded or not. You can send multiple core_project_nums as a batch to reduce the number of total requests. For each grant, there maybe already a a record in the database, due to each grant may have multiple duplicates. We only want to keep one single grant in our database.
- For each grant in the tsv file, the `pid` is the grant `core_project_num`. Please use this core_project_num to get publications from NIH RePORTER API.
- The NIH RePORTER API is `https://api.reporter.nih.gov/v2/publications/search`. Use a POST request to send a json request like the following:
```json 
{
  "criteria": {
    "core_project_nums": [
      "U18TR003807"
    ]
  },
  
  "offset": 0,
  "limit": 1
}
```
You can send multiple core_project_nums in one request. The response would look like the following:

```json
{
  "meta": {
    "search_id": null,
    "total": 6,
    "offset": 0,
    "limit": 1,
    "sort_field": "coreproject",
    "sort_order": "desc",
    "sorted_by_relevance": false,
    "properties": {}
  },
  "results": [
    {
      "coreproject": "U18TR003807",
      "pmid": 37908159,
      "applid": 10320988
    }
  ],
  "facet_results": []
}
```
Due to the rate limit, please split the request into batch. Each batch is 500 per request. also please sleep 1 second after a request if the data is not cached.
- Save the requests into a local tmp sqlite database. Create a table `grant_publications`, which has two columns: `core_project_num` and `pmid`.
- The local sqlite database can also be used a cache to avoid re-download the publications.
- Once downloaded all the publications for a given list, next step is to fetch the metadata of all the publications from PubMed API.
- PubMed provides an esummary api: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=37908159,35579189. You can use comma to concat multiple pmids together.
- The returned JSON looks like 
```json
{
  "header": {
    "type": "esummary",
    "version": "0.3"
  },
  "result": {
    "35579189": {
      "uid": "35579189",
      "pubdate": "2022 Jun 28",
      "epubdate": "2022 Jun 28",
      "source": "Lab Chip",
      "authors": [{
        "name": "Rima XY",
        "authtype": "Author",
        "clusterid": ""
      }, ...]
      ...
    }
  }
}
```
For each pmid, you will find the returned data in the `result.<pmid>`. Please save the paper's metadata in sqlite database. Create a table called `papers`, which has a two columns `pmid` and `metadata`. Just save the result json of that paper to the `metadata`.
- Once you get all the papers, the next step is to build a network of grants, publications, and authors. The basic idea is:
- Each grant, paper, or author is a node, the linkage between grant, paper, and author form edges.
- For example, if a paper is mentioned in two grants, then draw two lines between this paper and two grants.
- Same for authors, if an author is included in multiple papers, then draw multiple lines between this author and those papers.

Once all grants, publications, and authors information are ready. Let's create the final file for output.
There are two files for the output: `gpa.points.tsv` and `gpa.edges.json`.

First, for the `gpa.points.tsv`, follow these requirements:

- This `gpa.points.tsv` contains the following columns. And for each type of data, please generate differently. Follow these requirements:
    - pid: for grants, use core_project_num from the input tsv; for papers, use pmid; for authors, generate a hash string.
    - date: for grants, project_start_date, if project_start_date is not available, use fiscal_year but convert to <fiscal_year>-01-01 for same format; for papers, convert the pubdate to YYYY-MM-DD format; for authors, use the same date of paper.
    - journal: for grant, use agency_ic_admin; for paper, use source or journal; for authors, just use "Author".
    - title: for grant, use project_title from the input tsv; for paper, use title; for author, use their name.
    - mesh_terms: for grant, use pref_terms and spending_categories_desc from the input tsv. and add org_name at last. using ; to concat them; for paper, leave empty; for author, also leave empty
    - x: the first dimension from 2d embedding, later need to create from the text embedding
    - y: the second dimension from 2d embedding
    - citation_count: for grant, use award_amount; for papers, find the `citation_count` in another sqlite database `~/data/icite/latest/icite.db`'s `papers` table by `pmid` column; for authors, leave this as 1.
    - size: for grant, use the sqrt(award_amount) / 100; for paper, find the `relative_citation_ratio` in the icite.db as well; for authors, just leave as 5.
    - color: for grant, use #ffe100; for paper, use #f84848; for authors, use #1196fc.
- Read the input grant list `grant-list.tsv`, or user given input path to a tsv file.
- For each grant in the tsv file, create a string using the following format:
   "{fiscal_year} | {agency_ic_admin} | {project_title}"

   Get all papers supported by those grants from the local sqlite database. 
   For each paper supported by a grant, create a string using the following format:
   "{source} | {title}"

   For each author, also create a string using the following format. Just use one title from a author's publication.
   "Author name: {name} published {title}"

   As you know, paper and author may have duplicates in grants, so make sure de-duplicate. 
   Make sure you create a `gpa.metadata.tsv` and during this process, create a `gpa.edges.json`, that contains
   Next, let's create embeddings from all grants, papers, and author names.
- Use SentenceTransformer to embed the string to text embedding, use the model `google/embeddinggemma-300m` or any user provided model slug.
- Due the large size of dataset, please batch the embedding request to SentenceTransformer.
- Save all the embedding to `./gpa.embedding.npy` file, which is a NumPy file. User can also specify where to save.
- Once the high dimensional embedding is ready, user can reduce the embedding to 2d using tsne (python opentsne package). 
- The 2d embedding can be saved into a `./gpa.embd.npy` file. User can also specify where to save.
- Once the 2d embedding file is ready, user can merge the tsv file with the 2d embedding 
-  Each step, e.g., generate text embedding, dimension reduction, and merge, can run seperately. So we don't need to re-run embedding or dimension reduction again.
- For each step, please show a progress bar, especiall generate text embedding and merge. for umap, turn on verbose.