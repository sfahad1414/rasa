import importlib.metadata
import sys


def test_tensorflow_text_install():
    installed_packages_list = [
        dist.metadata["Name"].lower().replace("_", "-")
        for dist in importlib.metadata.distributions()
        if "Name" in dist.metadata
    ]
    tf_text_installed = "tensorflow-text" in installed_packages_list

    if sys.platform == "win32":
        assert not tf_text_installed
    else:
        assert tf_text_installed
