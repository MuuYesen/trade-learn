import setuptools

with open('README.md', 'r') as fh:
    README = fh.read()

VERSION = '0.1.1.8'

setuptools.setup(
    name='trade-learn',
    version=VERSION,
    author='',
    description='trade-learn Python Package',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'scikit-learn',
        'seaborn',
        'bokeh',
        'yfinance',
        'mootdx',
        'pydot',
        'quantstats',
        'htmlmin'
    ],
    url='https://github.com/MuuYesen/trade-learn',
    packages=setuptools.find_packages(include=['tradelearn', 'tradelearn.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
