sed '1d' docs/requirements.txt > docs/_requirements.txt
pip freeze | grep ailever > docs/requirements.txt
cat docs/_requirements.txt >> docs/requirements.txt
rm docs/_requirements.txt

python setup.py bdist_wheel
python -m twine upload dist/ailever-0.2.$1-py3-none-any.whl

git add .
git commit -m "update"
git push
pip install -U ailever
