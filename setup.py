from setuptools import setup, find_packages

setup(
    name="agrosarthi_rl_env",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0",
        "openai>=1.0",
    ],
)
