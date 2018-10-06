from setuptools import setup, find_packages

setup(name='darts',
      version='0.1',
      description='DARTS',
      url='',
      author='',
      author_email='',
      license='MIT',
      packages=['darts'],
      install_requires=[
						'flask',
						'numpy',
						'opencv-python',
						'pillow',
						'python-gflags',
						'requests',
						'markdown',
      ],
      zip_safe=False)
