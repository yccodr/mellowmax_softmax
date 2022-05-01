# mellowmax_softmax

## Project Setup

1. install [poetry](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)
2. download dependency

    ```bash
    cd <project_dir>
    poetry install
    ```

3. you are good to go

## Carry out an experiment

```bash
poetry run start -e <experiment_name>
```

### Experiment Name

- `custom_mdp`
- `random_mdp`
- `taxi`
- `lunar_lander`

## Dos & Don'ts

- Make sure you are developing on different branch instead of 'main' branch.
- After your work is done, push your branch and create a pull request to merge your code into 'main' branch.
- Do not merge your code into 'main' branch on local and push it. May cause conflict with others' code.

## Formatters

### Using vscode

add these lines below in .vscode/settings.json

```json
{
    "python.formatting.provider": "yapf"
}
```

You can also enable "format on save" in the setting.

1. Right click on the file and click "Sort Imports"
2. Right click on the file and click "Format Document"

## Coding Convention

Optional. It would be nice if you follow.

[Google Style Guide](https://google.github.io/styleguide/pyguide.html)

[Naming Convention](https://google.github.io/styleguide/pyguide.html#s3.16-naming)
