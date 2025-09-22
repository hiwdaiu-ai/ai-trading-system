#!/usr/bin/env python3
"""
AI Trading System Setup Configuration

This file contains the setup configuration for the AI Trading System,
defining package metadata, dependencies, and installation requirements
for time series forecasting, LSTM neural networks, and sentiment analysis.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-trading-system",
    version="0.1.0",
    author="AI Trading Team",
    author_email="team@aitradingsystem.com",
    description="AI Trading System integrating time series forecasting, LSTM, and sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hiwdaiu-ai/ai-trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.900",
            "pre-commit>=2.20.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-trading=main:main",
        ],
    },
    package_data={
        "ai_trading_system": ["data/*.csv", "models/*.pkl"],
    },
    include_package_data=True,
    zip_safe=False,
)
