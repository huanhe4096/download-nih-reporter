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


# Prompt for generate semantic embeddings and 2d