Hirshfeld Analysis module for PySCF
===================================

2022-02-16

* Version 0.2

Install
-------

There are two different options for installation:

* Install to python site-packages folder
```
pip install git+https://github.com/pyscf/hirshfeld
```

* Install in a custom folder for development
```
git clone https://github.com/pyscf/hirshfeld.git /home/abc/local/path

# Set pyscf extended module path:
echo 'export PYSCF_EXT_PATH=/home/abc/local/path:$PYSCF_EXT_PATH' >> ~/.bashrc
```

You can find more details of extended modules in the document
[extension modules](http://pyscf.org/pyscf/install.html#extension-modules)
