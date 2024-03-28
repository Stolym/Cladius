from auto.externals.colors import Colors, print_color, format_color
import json

class IModel():
    
    def __init__(self, type: str = None, shared_config: dict = {}) -> None:
        self.type : str = type
        self.model = None

        assert self.type == "tensorflow" or self.type == "pytorch", "Type must be either 'tensorflow' or 'pytorch'"
        assert self.load_model != None, "load_model method must be implemented"
        assert self.load_optimizer != None, "load_optimizer method must be implemented"
        assert self.load_loss != None, "load_loss method must be implemented"
        assert self.compile_model != None, "compile_model method must be implemented"

        self.config.update(shared_config)
        print_color(f"Actual Model Config: {json.dumps(self.config, indent=4)}", Colors.ORANGE)

        self.model = self.load_model()
        self.optimizer = self.load_optimizer(self.model)
        self.loss = self.load_loss()
        self.model = self.compile_model()

        