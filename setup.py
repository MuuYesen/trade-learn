import setuptools

from os import path as os_path
this_directory = os_path.abspath(os_path.dirname(__file__))
def read_file(filename):
    desc = ''
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        desc = f.read()
    return desc
def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if (not line.startswith('#')) and ('==' in line)]


with open('README.md', 'r') as fh:
    README = fh.read()

VERSION = '0.1.2.0'

setuptools.setup(
    name='trade-learn',
    version=VERSION,
    author='',
    description='trade-learn Python Package',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=read_requirements('requirements.txt'),
    url='https://github.com/MuuYesen/trade-learn',
    packages=setuptools.find_namespace_packages(),
    package_data={'': ['*.json', '*.html', '*.js', '*.css', '*.j2']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
