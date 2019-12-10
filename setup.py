from distutils.core import setup
from os import path
curr_dir = path.abspath(path.dirname(__file__))
with open(path.join(curr_dir, 'README.md'), encoding='utf-8') as f:
  long_desc = f.read()

  
setup(
  name = 'ADKit',
  packages = ['ADKit'],
  version = '0.11',
  license='MIT',
  description = 'A lightweight Python library supporting forward and reverse mode automatic differentiation variables and computation.',
  long_description = long_desc,
  long_description_content_type = 'text/markdown',
  author = 'The Differentiators (Michael Scott, Dimitris Vamvourellis, Yiwen Wang, Royce Yap)',
  author_email = 'mscott935.ms@gmail.com',
  url = 'https://github.com/the-differentiators/cs207-FinalProject',
  download_url = 'https://github.com/the-differentiators/cs207-FinalProject/archive/0.11.tar.gz',
  keywords = ['Automatic Differentiation', 'Differentiation', 'Derivatives', 'Math', 'Forward Mode', 'Reverse Mode'],
  install_requires=[
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)