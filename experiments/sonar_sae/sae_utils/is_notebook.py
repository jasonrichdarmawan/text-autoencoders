def is_notebook():
  try:
    __IPYTHON__ # type: ignore
    return True
  except NameError:
    return False