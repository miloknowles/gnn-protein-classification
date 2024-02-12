import os


def top_folder(rel='') -> str:
  """Returns the path to the top of the repository."""
  return os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), "../../")), rel)


def data_folder(rel='') -> str:
  """Returns a path relative to the `data` folder."""
  return os.path.join(top_folder('data'), rel)


def models_folder(rel='') -> str:
  """Returns a path relative to the `models` folder."""
  return os.path.join(top_folder('models'), rel)