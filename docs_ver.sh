pip install -U ailever

sed '1d' docs/requirements.txt > docs/_requirements.txt
pip freeze | grep ailever > docs/requirements.txt
cat docs/_requirements.txt >> docs/requirements.txt
rm docs/_requirements.txt

git add .
git commit -m "docs"
git push
