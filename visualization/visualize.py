from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.getcwd() + os.sep + "..")
os.environ["PATH"] += os.pathsep + r"D:\Tools\windows_10_cmake_Release_Graphviz-12.2.1-win64\Graphviz-12.2.1-win64\bin"
from module.densenet import Densenet
import torch
from torchviz import make_dot

saved_folder = "densenet_traced"

model = Densenet()
model.eval()
model.cpu() 
dummy_input = torch.randn(1, 3, 256, 256)

def graphvize_save(model, dummy_input):
    output = model(dummy_input)
    make_dot(output, params=dict(model.named_parameters())).render("model", format="png")

def tensorboard_save(model, dummy_input, saved_folder):
    traced = torch.jit.trace(model, dummy_input)
    writer = SummaryWriter(f'runs{os.sep}{saved_folder}')
    writer.add_graph(traced, dummy_input)
    writer.close()


if __name__ == "__main__":
    tensorboard_save(model, dummy_input, saved_folder)


