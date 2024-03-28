import sys
import json

def setup_cladius_package():
    with open("auto/base.json", "r") as json_file:
        package_data = json.load(json_file)
    sys.path.append(package_data["cladius_package_path"])

def get_base_config():
    with open("auto/base_config.json", "r") as json_file:
        package_data = json.load(json_file)
    return package_data