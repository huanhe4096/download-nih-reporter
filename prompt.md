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

- project_num: the grant number, but I only the core project number. for example, for "1R15DE032063-01A1", I only need the "R15DE032063", it's the second character to the character before a dash. Since there could be duplicated project number, please check whether a project number has been processed, if duplicated in whole or the core project number, you can skip it. Just count how many you have skipped.
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

1. Due to the large size, show a progress bar.
2. Save the output .tsv file to `./nih-reporter-grants.tsv` by default if user does not provide path to output
3. Due to the large size of data files, stream output the result to the output file, avoid saving all files into memory before dumping into files.


# Prompt for generate semantic embeddings and 2d