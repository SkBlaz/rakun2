
## emacs noise
find . -name '*~' -type f -delete

## noise
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

## import cleanup
find . -name '*.py' | xargs autoflake --in-place --remove-unused-variables --expand-star-imports

## formatting
find . -name '*.py' -print0 | xargs -0 yapf -i

flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
