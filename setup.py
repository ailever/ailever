from setuptools import setup, find_packages

setup(name                = 'ailever',
<<<<<<< HEAD
      version             = '0.2.381',
=======
      version             = '0.2.383',
>>>>>>> a45846f53a01ea4fdbcc1562f87d9330349ccfa2
      description         = 'Clever Artificial Intelligence',
      author              = 'ailever',
      author_email        = 'ailever.group@gmail.com',
      url                 = 'https://3044d22ab9179f52ecb34567f62c8b9819f0333d@github.com/ailever/ailever',
      install_requires    = ['dash', 'dash_bootstrap_components', 'plotly', 'beautifulsoup4', 'yahooquery', 'finance-datareader', 'statsmodels', 'matplotlib', 'pandas', 'numpy', 'requests'],
      packages            = find_packages(exclude = []),
      keywords            = ['ailever', 'clever', 'artificial intelligence'],
      python_requires     = '>=3',
      package_data        = {},
      zip_safe            = False,
      classifiers         = ['Programming Language :: Python :: 3.6',
                             'Programming Language :: Python :: 3.7',
                             'Programming Language :: Python :: 3.8',
                             'Programming Language :: Python :: 3.9',
                             'Programming Language :: Python :: 3.10'])
