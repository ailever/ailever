python setup.py bdist_wheel
python -m twine upload dist/ailever-0.1.$1-py3-none-any.whl
git add .
git commit -m "update"
git push
pip install -U ailever
