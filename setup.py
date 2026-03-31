from setuptools import setup, find_packages

setup(
    name="Hydromatic_Simulator",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Pillow",         
        "tensorflow",  
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'Hydromatic_Simulator = Hydromatic_Simulator.gui:run_gui',
        ],
    },
    author="YS",
    description="Hydromatic Simulator",
    python_requires=">=3.7",
)
