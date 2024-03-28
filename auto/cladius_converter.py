from utils.cladius_sys_import import setup_cladius_package
from externals.colors import Colors, print_color, format_color

import argparse

import json
import os
import inspect
import importlib.util

def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="Cladius Converter")

    parser.add_argument("--loader", type=str, help="Name of the loader. Must be in auto/loader and implement ILoader interface")
    parser.add_argument("--save", type=str, help="Name of the file to save the data")

    return parser.parse_args()

def check_if_loader_exists(loader_name: str):
    loader_directory = "auto/loader"
    loader_files = os.listdir(loader_directory)
    loader_files = [file for file in loader_files if file.endswith(".py")]

    for loader_file in loader_files:
        file_path = os.path.join(loader_directory, loader_file)
        module_name = os.path.splitext(loader_file)[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader_module)

        for name, obj in inspect.getmembers(loader_module):
            if inspect.isclass(obj):
                if loader_name == name:
                    print_color(f"Loader {loader_name} exists.", Colors.GREEN)
                    return module_name

    print_color(f"Loader {loader_name} does not exist.", Colors.RED)
    exit(1)

def load_loader(module_name: str = "cladius_loader_unit_test", loader_name: str = "FashionMNISTLoader"):
    loader_module = __import__(f"auto.loader.{module_name}", fromlist=[loader_name])
    loader_class = getattr(loader_module, loader_name)
    loader_instance = loader_class()
    return loader_instance

def save_loader(loader, save_file: str):
    loader_data = loader.data
    save_path = f"auto/loader/saved/{save_file}"
    with open(save_path, 'w') as json_file:
        json.dump(loader_data, json_file)
    print_color(f"Data saved successfully to {save_path}", Colors.GREEN)

def execute_command(args: argparse.Namespace):
    if args.loader is None:
        print_color("Loader name not provided", Colors.RED)
        exit(0)
    else:
        print_color(f"Loader Name: {args.loader}", Colors.DARK_YELLOW)
    
    if args.save is None:
        print_color("Save file not provided", Colors.RED)
        exit(0)
    else:
        print_color(f"Save File Path: {args.save}", Colors.DARK_YELLOW)
    module_name = check_if_loader_exists(args.loader)
    loader = load_loader(module_name, args.loader)
    save_loader(loader, args.save)

if __name__ == "__main__":
    setup_cladius_package()
    print_color("Cladius Converter", Colors.YELLOW)
    args = parse_args()
    execute_command(args)
