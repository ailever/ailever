from setuptools import setup, find_packages
 
setup(name                = 'ailever',
      version             = '0.0.1',
      description         = 'Clever Artificial Intelligence',
      author              = 'ailever',
      author_email        = 'ailever.group@gmail.com',
      url                 = 'https://github.com/ailever/ailever',
      install_requires    =  [],
      packages            = find_packages(exclude = []),
      keywords            = ['ailever', 'clever', 'artificial intelligence'],
      python_requires     = '>=3',
      package_data        = {},
      zip_safe            = False,
      classifiers         = ['Programming Language :: Python :: 3',
                             'Programming Language :: Python :: 3.2',
                             'Programming Language :: Python :: 3.3',
                             'Programming Language :: Python :: 3.4',
                             'Programming Language :: Python :: 3.5',
                             'Programming Language :: Python :: 3.6',
                             'Programming Language :: Python :: 3.7'])
