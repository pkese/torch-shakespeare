#!/usr/bin/env -S dotnet fsi

#I "cuda-hack/bin/Release/net7.0/publish/"
#r "cuda-hack.dll"
#r "TorchSharp.dll"

open System
open TorchSharp

let device =
    match torch.cuda.is_available() with
    | true -> torch.CUDA
    | false -> torch.CPU

printfn "device=%A" device

//#load "llm.fsx"

let cfg = {|
    vocabSize = 65
    blockSize = 32
    batchSize = 4L
    encodingSize = 80L
    nHeads = 4L
    dropout = 0.1
    bias = false
|}        

let model =
    //new Llm.LanguageModel(2, cfg.nHeads, cfg.encodingSize, cfg.vocabSize, cfg.blockSize, cfg.bias, cfg.dropout)
    torch.jit.load<torch.Tensor, torch.Tensor>("shakespeare.pt.zip")
    |> fun model -> model.``to``(device)

printfn "Model parameters"
for struct(name,param) in model.named_parameters() do
    printfn $"%s{name.PadRight(36)}: %s{param.ToString()}"

let xs =
    torch
        .tensor([| for i in 1..cfg.blockSize -> Random.Shared.Next() % cfg.vocabSize |], device=device, dtype=torch.int64)
        .unsqueeze(0)

let ys = model.forward(xs)

printfn $"shapes: xs=%A{xs.shape} ys=%A{ys.shape}"


