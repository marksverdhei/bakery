# python-template

This project uses [`uv`](https://github.com/astral-sh/uv), which can be installed in one line: 
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To get started with the project:

```
git clone https://github.com/marksverdhei/python-template.git
cd python-template
```

Then install project with
```
uv sync --dev
source .venv/bin/activate
```

For development, make sure to install the project with the `-e` flag 
so the source code isn't distributed to `.venv`:
```
uv pip install -e .
```

Then you can run the default cli, which can be customized in `cli.py` and `pyproject.toml`  
```
$ python-template
Hello from python-template
```

How to install:
```bash
pip install git+https://github.com/marksverdhei/python-template.git
```
