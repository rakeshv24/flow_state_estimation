from setuptools import setup, find_packages

setup(
    name="active-perception",
    version="0.1",
    description="Active Perception Framework for Flow State Estimation",
    author="Rakesh Vivekanandan",
    author_email="vivekanr@oregonstate.edu",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "scipy", "pyyaml"],
)
