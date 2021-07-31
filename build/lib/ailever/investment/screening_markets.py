import os
import numpy as np
from numpy import linalg
import pandas as pd

def reits_screening(period, path):
    print(f'[AILEVER] Recommandations based on latest {period}')
    idx2csv = dict()

    for idx, csv in enumerate(os.listdir(path)):
        idx2csv[idx] = csv
        if idx == 0 :from github import Github
g = Github("username", "password")

repo = g.get_user().get_repo(GITHUB_REPO)
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))

with open('/tmp/file.txt', 'r') as file:
    content = file.read()

# Upload to github
git_prefix = 'folder1/'
git_file = git_prefix + 'file.txt'
if git_file in all_files:
    contents = repo.get_contents(git_file)
    repo.update_file(contents.path, "committing files", content, contents.sha, branch="master")
    print(git_file + ' UPDATED')
else:
    repo.create_file(git_file, "committing files", content, branch="master")
    print(git_file + ' CREATED')

            base = pd.read_csv(path+csv)['close'][-period:].fillna(method='bfill').fillna(method='ffill').values[:,np.newaxis]
        else:
            appending = pd.read_csv(path+csv)['close'][-period:].fillna(method='bfill').fillna(method='ffill').values[:,np.newaxis]
            base = np.c_[base, appending]

    x, y = np.arange(base.shape[0]), base
    bias = np.ones_like(x)
    X = np.c_[bias, x]

    b = linalg.inv(X.T@X) @ X.T @ y
    yhat = X@b
    recommand = yhat[-1] - yhat[-2]
    return list(map(lambda x: idx2csv[x][:-4], np.argsort(recommand)[::-1]))

