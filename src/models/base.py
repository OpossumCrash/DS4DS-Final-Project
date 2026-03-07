import abc
import torch 

class BaseModel(torch.nn.Module, abc.ABC):
    """
    Base class for the spatio-temporal dataset models.
    It should accept in the forward pass:
    - x: the input data, of shape (B, T, C, H, W)
    Where B is the batch size, T is the number of time steps, C is the number 
    of channels, H and W are the height and width of the input data.
    
    The models should be able to handle T=1,...,N time steps as input and 
    depending on the settings output T_out=1,...,M time steps as output.
    
    """
    def __init__(self, t_in: int, t_out: int, c_out: int):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.c_out = c_out

    @abc.abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: input data of shape (B, T, C, H, W)
        Returns:
            output data of shape (B, T_out, C_out, H, W)
        """
        pass