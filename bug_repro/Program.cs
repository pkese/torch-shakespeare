using System;
using TorchSharp;

var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

Console.WriteLine($"device={device}");

const int vocabSize = 65;
const int blockSize = 32;

var model = torch.jit.load<torch.Tensor, torch.Tensor>("../shakespeare.pt.zip").to(device);

Console.WriteLine("Model parameters");
foreach (var (name, param) in model.named_parameters())
{
    Console.WriteLine($"{name.PadRight(36)}: {param}");
}

var xs = torch.tensor(new int[blockSize].Select(_ => new Random().Next(vocabSize)).ToArray(), device: device, dtype: torch.int64).unsqueeze(0);

var ys = model.forward(xs);

Console.WriteLine($"shapes: xs={xs.shape} ys={ys.shape}");