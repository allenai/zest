"""setup.py file for packaging zest."""

from setuptools import setup


with open('readme.md', 'r') as readme_file:
    readme = readme_file.read()


setup(
    name='zest',
    version='0.1.0',
    description='Learn NLP tasks from instructions.',
    long_description=readme,
    url='https://github.com/allenai/zest',
    author='Nicholas Lourie',
    author_email='nicholasl@allenai.org',
    keywords='artificial intelligence ai machine learning ml',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=['zest'],
    package_dir={'': 'src'},
    scripts=[],
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>= 3.6',
    zip_safe=False)
