from setuptools import setup, find_packages

setup(name                = 'ailever',
<<<<<<< HEAD
      version             = '0.2.457',
=======
      version             = '0.2.456',
>>>>>>> e3eb344b02fc4ee3047f901f34d63eb197ddec59
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
