# Jupyter Notebook Quick Reference

## Starting Jupyter

### Start Jupyter Notebook (Classic Interface)
```bash
jupyter notebook
```

### Start JupyterLab (Modern Interface)
```bash
jupyter lab
```

### Start Jupyter on a specific port
```bash
jupyter notebook --port 8889
```

### Start Jupyter without opening browser
```bash
jupyter notebook --no-browser
```

### Open a specific notebook
```bash
jupyter notebook getting_started.ipynb
```

## Common Jupyter Commands

### List running notebook servers
```bash
jupyter notebook list
```

### Stop a notebook server
Press `Ctrl+C` in the terminal where Jupyter is running, then confirm with `y`

### Convert notebook to other formats
```bash
# Convert to HTML
jupyter nbconvert --to html notebook.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf notebook.ipynb

# Convert to Python script
jupyter nbconvert --to script notebook.ipynb
```

### Execute a notebook from command line
```bash
jupyter nbconvert --to notebook --execute notebook.ipynb
```

## Keyboard Shortcuts (in Notebook)

### Command Mode (press Esc to enter)
- `A` - Insert cell above
- `B` - Insert cell below
- `D, D` - Delete selected cell
- `M` - Change cell to Markdown
- `Y` - Change cell to Code
- `Shift + Enter` - Run cell and select below
- `Ctrl + Enter` - Run cell
- `Alt + Enter` - Run cell and insert below

### Edit Mode (press Enter to enter)
- `Tab` - Code completion or indent
- `Shift + Tab` - Tooltip
- `Ctrl + ]` - Indent
- `Ctrl + [` - Dedent
- `Ctrl + A` - Select all
- `Ctrl + Z` - Undo
- `Ctrl + Shift + Z` - Redo

## Jupyter Magic Commands

### Useful Magic Commands in Notebooks
```python
# Time execution of a single line
%time <code>

# Time execution of entire cell
%%time

# Display matplotlib plots inline
%matplotlib inline

# Load external Python file
%load filename.py

# Run external Python file
%run filename.py

# List all variables
%who

# System command
!pip install package_name
```

## Troubleshooting

### Jupyter not found
Make sure you've activated your virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Port already in use
```bash
jupyter notebook --port 8889
```

### Clear output of all cells
In Jupyter: Cell → All Output → Clear

### Restart kernel
In Jupyter: Kernel → Restart

### Update Jupyter
```bash
pip install --upgrade jupyter jupyterlab notebook
```
