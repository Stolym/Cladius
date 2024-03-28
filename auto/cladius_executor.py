from utils.cladius_sys_import import setup_cladius_package, get_base_config
from utils.cladius_serializer import base64_to_ndarray, ndarray_to_base64
from externals.colors import Colors, print_color, format_color
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import argparse

import os
import inspect
import importlib.util

import json

import seaborn as sns

import torch
import tensorflow as tf

import numpy as np

import time
from datetime import datetime, UTC


class ConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not getattr(namespace, self.dest):
            setattr(namespace, self.dest, {})
        key, type, value = values
        try:
            if type == "int":
                value = int(value)
            elif type == "float":
                value = float(value)
            elif type == "str":
                value = str(value)
            elif type == "bool":
                value = bool(value)
        except ValueError:
            pass
        getattr(namespace, self.dest)[key] = value

# --model: Model Name to use must be in ['bert', 'gpt2', 'roberta', 'distilbert', 'albert', 'xlnet', 'xlm', 'ctrl', 'electra', 'reformer', 'funnel', 'bart', 't5', 'pegasus', 'blenderbot', 'dialogpt', 'gpt_neo', 'prophetnet', 'mbart', 'marian', 'turing', 'fsmt']
def parse_args() -> argparse.Namespace:
    """
    All commands for Cladius Executor:

    1. cladius_executor.py [command] [args]

    --model: Model Name to use must be in auto/model and have IModel interface
    --test: boolean, whether to test the model
    --train: boolean, whether to train the model
    --predict: boolean, whether to predict using the model
    --data: Path to data file (json) with data_x and data_y
    --save: Path to save the model, auto/model/saved/[model_name]
    --load: Path to load the model weights, auto/model/saved/[model_name]
    --help: Display help message

    2. cladius_executor.py --model [model_name] --train --data [data_file] --save [model_weights]
    3. cladius_executor.py --model [model_name] --test --data [data_file] --load [model_weights]
    4. cladius_executor.py --model [model_name] --predict --data [data_file] --load [model_weights]
    """
    parser = argparse.ArgumentParser(description="Cladius Executor")
    
    parser.add_argument("--model", type=str, help="Model name to use, must be in auto/model and implement IModel interface")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--predict", action="store_true", help="Whether to predict using the model")
    parser.add_argument("--data", type=str, help="Path to data file (JSON) with data_x and data_y")
    parser.add_argument("--save", type=str, help="Path to save the model, defaults to auto/model/saved/[model_name]")
    parser.add_argument("--load", type=str, help="Path to load the model weights, defaults to auto/model/saved/[model_name]")

    parser.add_argument("--config", nargs=3, action=ConfigAction, help="Set the update configuration for the model. Use as --config key type(int, float, str) value")
    parser.add_argument("--cconfig", nargs=3, action=ConfigAction, help="/!\ Depracated /!\ Set the update configuration for the cladius. Use as --cconfig key type(int, float, str) value")

    return parser.parse_args()


def check_if_model_exist(model_name: str) -> str:
    loader_directory = "auto/model"
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
                if model_name == name:
                    print_color(f"Model {model_name} exists.", Colors.GREEN)
                    return module_name
    print_color(f"Model {model_name} does not exist.", Colors.RED)
    exit(1)

def load_model_instance(module_name: str, model_name: str, shared_config: dict = {}):
    model_module = __import__(f"auto.model.{module_name}", fromlist=[model_name])
    model_class = getattr(model_module, model_name)
    model_instance = model_class(shared_config=shared_config)
    return model_instance


def load_cladius_config(args: argparse.Namespace):
    base_config = get_base_config()
    base_config.update(args.cconfig)
    return base_config


def str_to_numpy_dtype(dtype: str):
    return {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64
    }[dtype]
    
def str_to_torch_dtype(dtype: str):
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64
    }[dtype]
    
def load_data_loader(data_path: str, eval_split: float = 0.2):
    with open(data_path, "r") as json_file:
        data = json.load(json_file)

    print_color(f"Data X Shape: {data['data_x_shape']}", Colors.DARK_YELLOW)
    print_color(f"Data Y Shape: {data['data_y_shape']}", Colors.DARK_YELLOW)

    print_color(f"Data X Type: {data['data_x_dtype']}", Colors.DARK_YELLOW)
    print_color(f"Data Y Type: {data['data_y_dtype']}", Colors.DARK_YELLOW)

    data["data_x"] = base64_to_ndarray(data["data_x"], dtype=str_to_numpy_dtype(data['data_x_dtype']), shape=data["data_x_shape"])
    data["data_y"] = base64_to_ndarray(data["data_y"], dtype=str_to_numpy_dtype(data['data_y_dtype']), shape=data["data_y_shape"])

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data["data_x"], data["data_y"], test_size=eval_split)
    return data_x_train, data_x_test, data_y_train, data_y_test, data['data_x_dtype'], data['data_y_dtype']


def save_torch_model(model, save_file: str):
    save_path = f"auto/model/saved/{save_file}"
    torch.save(model.model.state_dict(), save_path)
    print_color(f"Model saved successfully to {save_path}", Colors.GREEN)

def train_torch_model(model, args: argparse.Namespace):
    config = model.config
    data_path = f"auto/loader/saved/{args.data}"
    data_x_train, data_x_test, data_y_train, data_y_test, x_dtype, y_dtype = load_data_loader(data_path, config["eval_split"])
    
    data_x_train = torch.tensor(data_x_train, dtype=str_to_torch_dtype(x_dtype))
    data_x_test = torch.tensor(data_x_test, dtype=str_to_torch_dtype(x_dtype))
    data_y_train = torch.tensor(data_y_train, dtype=str_to_torch_dtype(y_dtype))
    data_y_test = torch.tensor(data_y_test, dtype=str_to_torch_dtype(y_dtype))

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    plots = config["plots"]

    losses = {
        "train": [],
        "test": []
    }

    accuracies = {
        "train": [],
        "test": []
    }
    print_color(f"Training Model: {model}", Colors.DARK_YELLOW)

    for epoch in range(epochs):
        for i in range(0, len(data_x_train), batch_size):
            start = time.time()
            model.model.train()
            model.model.zero_grad()

            data_x = data_x_train[i:i+batch_size]
            data_y = data_y_train[i:i+batch_size]

            y_pred = model.model(data_x)
            loss = model.loss(y_pred, data_y)
            loss.backward()
            model.optimizer.step()

            losses["train"].append(loss.item())
            accuracies["train"].append((torch.argmax(y_pred, dim=1) == data_y).float().mean())
            delta = time.time() - start
            print_color(f"Epoch: {epoch+1:>5d}/{epochs} | Batch: {i+1:>5d}/{len(data_x_train)} | Loss: {np.round(loss.item(), 4):>5f} | Acc: {np.round(accuracies['train'][-1].item(), 4):>5f}", Colors.DARK_YELLOW)
            print_color(f"Time Train: {datetime.fromtimestamp(delta, UTC).strftime('%H:%M:%S.%f')}", Colors.DARK_YELLOW)
            continue

            start = time.time()
            model.model.eval()
            with torch.no_grad():
                t_loss = 0
                t_acc = 0
                for i in range(0, len(data_x_test), batch_size):
                    data_x = data_x_test[i:i+batch_size]
                    data_y = data_y_test[i:i+batch_size]
                    y_pred = model.model(data_x)
                    loss = model.loss(y_pred, data_y)
                    t_loss += loss.item()
                    t_acc += (torch.argmax(y_pred, dim=1) == data_y).float().mean()

                loss = t_loss / (len(data_x_test) / batch_size)
                acc = t_acc / (len(data_x_test) / batch_size)

                losses["test"].append(loss)
                accuracies["test"].append(acc)
            delta = time.time() - start
            print_color(f"Time Eval: {datetime.fromtimestamp(delta, UTC).strftime('%H:%M:%S.%f')}", Colors.DARK_YELLOW)
    if plots:
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(losses["train"], label="Train Loss")
        ax[0].plot(losses["test"], label="Test Loss")
        ax[0].set_title("Loss")
        ax[0].legend()

        ax[1].plot(accuracies["train"], label="Train Acc")
        ax[1].plot(accuracies["test"], label="Test Acc")
        ax[1].set_title("Accuracy")
        ax[1].legend()

        plt.show()

    save_torch_model(model, args.save)
        


def train_model(model, args: argparse.Namespace):
    if model.type == "pytorch":
        train_torch_model(model, args)
    else:
        print_color("Model train type not supported actually {model.type}", Colors.RED)
        exit(0)

def test_model(model, args: argparse.Namespace):
    pass

def predict_model(model, args: argparse.Namespace):
    pass

def execute_command(args: argparse.Namespace):
    if args.model is None:
        print_color("Model name not provided", Colors.RED)
        exit(0)
    else:
        print_color(f"Model Name: {args.model}", Colors.DARK_YELLOW)

    if not(args.train) and not(args.test) and not(args.predict):
        print_color("No command provided", Colors.RED)
        exit(0)
    else:
        print_color(f"Mode: (Train: {args.train}, Test: {args.test}, Predict: {args.predict})", Colors.DARK_YELLOW)
    
    if args.train:
        if args.data is None:
            print_color("Data file not provided", Colors.RED)
            exit(0)
        else:
            print_color(f"Data File Path: {args.data}", Colors.DARK_YELLOW)
    elif args.test or args.predict:
        if args.data is None:
            print_color("Data file not provided", Colors.RED)
            exit(0)
        else:
            print_color(f"Data File Path: {args.data}", Colors.DARK_YELLOW)
        
        if args.load is None:
            print_color("Model weights not provided", Colors.RED)
            exit(0)
        else:
            print_color(f"Model Weights Path: {args.load}", Colors.DARK_YELLOW)
    model_module_name = check_if_model_exist(args.model)
    model = load_model_instance(model_module_name, args.model, args.config)
    print_color(f"Model Loaded: {model}", Colors.DARK_YELLOW)
    
    if args.train:
        train_model(model, args)
    elif args.test:
        test_model(model, args)
    elif args.predict:
        predict_model(model, args)

if __name__ == "__main__":
    setup_cladius_package()
    print_color("Cladius Executor", Colors.YELLOW)
    args = parse_args()
    print_color(args, Colors.PURPLE)

    execute_command(args)