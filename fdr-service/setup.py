from setuptools import setup, find_packages

setup(
    name="fdr-service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'lightgbm>=4.0.0',
        'fastapi>=0.100.0',
        'uvicorn>=0.23.0',
        'pydantic>=2.0.0',
        'shap>=0.42.0',
        'pytest>=7.4.0',
    ],
    python_requires='>=3.8',
)
