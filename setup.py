from setuptools import setup, find_packages

def read_requirements(file_path):
    """Read the requirements from a requirements.txt file."""
    with open(file_path) as f:
        # Strip any whitespace and skip empty lines
        return [line.strip() for line in f if line and not line.startswith("#")]

# Read the requirements
requirements = read_requirements('requirements.txt')

setup(
    name='decima2',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    # Other metadata options
    author='Torty Sivill',
    author_email='tortysivill@decima2.co.uk',
    description='Evaluation Toolkit for Machine Learning Models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Decima2/Decima2Toolkit',
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",  # Corrected classifier
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6, <=3.12',
)
