import os
root = "U:\debarras\ia-export"
for path, subdirs, files in os.walk(root):
    for name in files:
        print(os.path.join(path, name))
