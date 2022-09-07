"""A collection of miscellaneous utility functions.

TODO: Anything that used to be in nems.utils.py should go in this directory,
      with functions of similar purpose grouped in modules. Even if there's just
      one function that belongs in a solo group, it should *not* just be dumped
      in 'misc' or 'etc' or a similarly named file with other one-offs.

      Alternatively, if all utilites can be reasonably assigned to specific
      libraries, we can just get rid of this directory altogether.


Contents
--------
    `json.py` : Ensure proper json-ification of Models, Layers, etc.
                NOTE: tools from `json.py` should *not* be imported here
                (for the `nems.tools` API), as this can easily cause circular
                imports.

"""
