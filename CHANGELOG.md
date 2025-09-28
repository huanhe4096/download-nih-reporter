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

Just title?

Remove funding agency, just keep title and fiscal year.

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