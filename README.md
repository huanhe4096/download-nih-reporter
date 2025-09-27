# Download NIH RePORTER data

First, install deps

```bash
pip install -r requirements.txt
```

Download all projects:

```bash
python download_nih_projects.py download
```

If everything works, the downloaded JSON files should be saved at `./cache` folder. 
Running `python download_nih_project.py stat` will show something similar to this:

```
🔍 Download Statistics:
- Total combinations: 2296
- Completed combinations: 2295
- Total projects found: 2,719,706
- Progress: 100.0%
```