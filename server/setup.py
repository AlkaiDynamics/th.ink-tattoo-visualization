from setuptools import setup, find_packages

setup(
    name='think_backend',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fastapi==0.95.1',
        'uvicorn==0.22.0',
        'SQLAlchemy==1.4.46',
        'pydantic==1.10.7',
        'passlib[bcrypt]==1.7.4',
        'python-jose[cryptography]==3.4.0',
        'stripe==4.34.0',
        'python-dotenv==1.0.0',
        'requests==2.31.0',
        'python-multipart==0.0.5',
        'typing-extensions==4.5.0',
    ],
    entry_points={
        'console_scripts': [
            'start-server=server.app:app',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='Th.ink AI-powered Tattoo Visualization System Backend',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/think-backend',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Framework :: FastAPI',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)