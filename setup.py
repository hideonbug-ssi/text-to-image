from setuptools import setup, find_namespace_packages

setup(
    name='handwritten-text-recognition',
    description='Handwritten text recognition',
    packages=find_namespace_packages(include=['htr_pipeline', 'htr_pipeline.*']),
    install_requires=['numpy',
                      'onnxruntime',
                      'opencv-python',
                      'scikit-learn',
                      'editdistance',
                      'path'],
    python_requires='>=3.8',
    package_data={'htr_pipeline.models': ['*']}
)
