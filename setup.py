import os

# using pip install -e . installs this package but cannot be found by import
import setuptools

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(f"{dir_path}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


print(setuptools.find_packages())

setuptools.setup(
    name="jiayi_sudoku_solver",
    version="0.0.1",
    author="Jiayi Cox",
    author_email="jiayisixer@gmail.com",
    description="Solve sudoku with simple computer science methods, machine learning, and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # packages = ['code']
    packages=setuptools.find_packages(),
)
