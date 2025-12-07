import os

os.system("py -m pip install -r requirements.txt")
os.system("py Data_Generator.py")
os.system("py Model.py")