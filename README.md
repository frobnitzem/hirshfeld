Hirshfeld Analysis module for PySCF
===================================

2022-02-17

* Version 0.3
   - fixed `make_symmetric` function to keep total spin constant

   - added comparison test to original Hirshfeld paper

* Version 0.2
   - implemented `_dm_of` function to replace `make_rdm1()`,
     but handle fractional occupancies correctly

   - changed `_calc` to use UKS by default, since hydrogen had
     a strange `mo_occ` shape with ROKS.

   - fixed charge() and integrate() functions to integrate
     charge densities, rather than electron densities

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
