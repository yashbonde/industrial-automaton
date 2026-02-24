# Industrial Automaton

Read about this research at [yashbonde.com](https://yashbonde.com/blogs/automata/0-prologue-start)

```sh
git clone https://github.com/yashbonde/industrial-automaton.git
cd industrial-automaton
uv venv
uv sync
uv pip install -e .

# print data to feed into AI
uv run inmaton-models
uv run inmaton-tasks

# train a new model
uv run inmaton --help
```
