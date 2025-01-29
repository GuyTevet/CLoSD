import subprocess
import sys

dependencies = ['joblib', 'torch', 'numpy', 'scipy', 'trimesh', 'xml-python',]
subprocess.run([sys.executable, "-m", "ensurepip"])
subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies)