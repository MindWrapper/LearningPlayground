from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fast_ai_chapter2",
    version="0.0.1",
    packages=["."],
    install_requires=requirements,
)