from setuptools import setup, find_packages
import os

def read_version():
    with open("src/version.py") as f:
        exec(f.read())
        return locals()["__version__"]

def read_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

setup(
    name="think-ar",
    version=read_version(),
    description="Real-time AR tattoo visualization system with AI backend",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="thInk Team",
    author_email="contact@think-ar.dev",
    url="https://github.com/think-ar/think",
    packages=find_packages(include=['src*', 'server*', 'microservices*', 'tools*']),
    include_package_data=True,
    install_requires=[
        # Core dependencies
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "torch>=2.1.0",
        "pillow>=10.0.0",
        "mediapipe>=0.10.0",
        "diffusers>=0.24.0",
        "transformers>=4.36.0",
        
        # Server dependencies
        "fastapi>=0.109.0",
        "uvicorn>=0.25.0",
        "python-jose[cryptography]>=3.4.0",
        "bcrypt>=4.1.0",
        "SQLAlchemy>=2.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "stripe>=8.11.0",
        "psutil>=5.9.0",
        
        # Added missing dependencies
        "jinja2>=3.1.0",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "prometheus-client>=0.19.0",
        "alembic>=1.13.0",  # For database migrations
        "redis>=5.0.0"      # For caching
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-asyncio>=0.23.0',
            'black>=23.12.0',
            'mypy>=1.8.0',
            'flake8>=7.0.0',
            'docker>=7.0.0',  # Added for container management
            'pre-commit>=3.5.0'  # Added for git hooks
        ],
        'gpu': [
            'torch>=2.1.0+cu118'
        ],
        'monitoring': [  # Added monitoring tools
            'prometheus-client>=0.19.0',
            'grafana-api>=1.0.3'
        ]
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "think=src.main:main",
            "think-server=server.app:main",
            "think-worker=microservices.text_to_tattoo.main:main"  # Added worker entry point
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={
        'think': [
            'assets/*.ico',
            'models/*.pth',
            'config/*.json',
            'config/environments/*.json',
            'templates/*.html',    # Added templates
            'static/**/*',        # Added static files
            'migrations/**/*'     # Added migrations
        ]
    },
    data_files=[
        ('config', ['config/development.json', 'config/production.json']),
        ('config/secrets', ['config/secrets.yaml.template'])  # Added secrets template
    ]
)