from setuptools import setup, find_packages


setup(
    name='birdnet',
    version='2.4.0',
    url='https://github.com/kahst/BirdNET-Analyzer',
    author='Stefan Kahl',
    author_email='sk2487@cornell.edu',
    description='BirdNET analyzer for scientific audio data processing.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'bottle',
        'resampy',
        'librosa',
        'requests',
        'gradio',
    ],
    tests_requries=[
        'pytest',
        'pytest-cov',
    ],
)
