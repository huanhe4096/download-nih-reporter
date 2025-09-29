# v4 test

```python
return f"{fiscal_year} | {agency_ic_admin} | {project_title}"
```

tSNE

Looks nice.

# v5 test

Just change the order of fiscal_year

```python
return f"{agency_ic_admin} | {project_title} | {fiscal_year}"
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 64 \
  --embedding_output nih-reporter-grants.embedding.v5.npy \
  --embd_output=nih-reporter-grants.embd.v5.npy \
  --final_outpu grants.v5.tsv
```

Ok, I think the `google/embeddinggemma-300m` model is:

1. case sensitive
2. sequence sensitive


# v6 test

Make the title not that case

```python
project_title = ' '.join([word.capitalize() for word in project_title.split()])
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 64 \
  --embedding_output nih-reporter-grants.embedding.v6.npy \
  --embd_output=nih-reporter-grants.embd.v6.npy \
  --final_outpu grants.v6.tsv
```


# v7 test

Remove funding agency, just keep title and fiscal year.

```python
return f"{project_title} | {fiscal_year}"
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 64 \
  --embedding_output nih-reporter-grants.embedding.v7.npy \
  --embd_output=nih-reporter-grants.embd.v7.npy \
  --final_outpu grants.v7.tsv
```


# v8 test

Just title.

```python
return project_title
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 64 \
  --embedding_output nih-reporter-grants.embedding.v8.npy \
  --embd_output=nih-reporter-grants.embd.v8.npy \
  --final_outpu grants.v8.tsv
```


# v9 test

Just title + agency.

```python
return f"{project_title} | {agency_ic_admin}"
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 64 \
  --embedding_output nih-reporter-grants.embedding.v9.npy \
  --embd_output=nih-reporter-grants.embd.v9.npy \
  --final_outpu grants.v9.tsv
```


# v10 test

Reverse the order of title and agency.

Basically, not much difference.

```python
return f"{agency_ic_admin} | {project_title}"
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 96 \
  --embedding_output nih-reporter-grants.embedding.v10.npy \
  --embd_output=nih-reporter-grants.embd.v10.npy \
  --final_outpu grants.v10.tsv
```


# v11 test

I want to use a more natural way to describe 

```python
return f"{project_title}. Project starts on {project_start_date}, ends on {project_end_date}. {agency_ic_admin}."
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 96 \
  --embedding_output nih-reporter-grants.embedding.v11.npy \
  --embd_output=nih-reporter-grants.embd.v11.npy \
  --final_outpu grants.v11.tsv
```



# v12 test

I want to try text only, but not too long.

The result is interesting, projects are well splited with year. Why?

```python
return f"{project_title} | {pref_terms} {spending_categories_desc}"
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 32 \
  --embedding_output nih-reporter-grants.embedding.v12.npy \
  --embd_output=nih-reporter-grants.embd.v12.npy \
  --final_outpu grants.v12.tsv
```




# v13 test

I want to try same v12 text, but using BAAI/bge-small-en-v1.5

```python
return f"{project_title} | {pref_terms} {spending_categories_desc}"
```

Run:

```bash
python embedding_pipeline.py all \
  --batch_size 32 \
  --model BAAI/bge-small-en-v1.5\
  --embedding_output nih-reporter-grants.embedding.v13.npy \
  --embd_output=nih-reporter-grants.embd.v13.npy \
  --final_outpu grants.v13.tsv
```
