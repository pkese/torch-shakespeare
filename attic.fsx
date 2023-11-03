#I "cuda/bin/Release/net7.0/publish/"
//#r "cuda.dll"
#r "TorchSharp.dll"

open System
open TorchSharp

let inline ( ** ) (x:torch.Tensor) y = torch.matmul(x, y)
let inline ( .* ) (x:torch.Tensor) y = torch.dot(x, y)

type Head(headSize:int64) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("Head")
    let blockSize, encodingSize = cfg.blockSize, cfg.encodingSize
    let key = torch.nn.Linear(encodingSize, headSize, hasBias=false)
    let query = torch.nn.Linear(encodingSize, headSize, hasBias=false)
    let value = torch.nn.Linear(encodingSize, headSize, hasBias=false)
    let scale = torch.Tensor.op_Implicit(MathF.Pow(float32 headSize, -0.5f))
    let tril = torch.tril(torch.ones(blockSize, blockSize, dtype=torch.bool)).log()
    do
        this.RegisterComponents()
        //this.register_buffer("tril", tril)
        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore    

    override this.forward(xs) =
        match xs.shape with
        | [| B; T; C |] ->
            //printfn "xs.shape: %A" xs.shape
            use k = key.forward(xs)
            //printfn "here"
            use q = query.forward(xs)
            use v = value.forward(xs)
            use k = k.transpose(-2, -1)
            use wei = (q ** k) * scale
            //printfn "wei1: %A" (wei.select(0L,0L).print())
            let wei = wei + tril
            //printfn "wei2: %A" (wei.select(0L,0L).print())
            use wei = torch.softmax(wei, dim=1)
            //printfn "wei3: %A" (wei.select(0L,0L).print())
            let out = torch.matmul(wei, v)
            out
        | _ -> failwith "Expected input shape [B, T, C]"

type MultiHeadAttention(nHeads, headSize) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("MultiHeadAttention")
    let heads = torch.nn.ModuleList([| for i in 1..nHeads -> new Head(headSize) |])
    let proj = torch.nn.Linear(cfg.encodingSize, cfg.encodingSize, hasBias=false)
    do
        this.RegisterComponents()
        //this.register_buffer("tril", tril)
        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore    

    override this.forward(xs) =
        torch.cat([| for head in heads -> head.forward(xs) |], dim=2)
        |> proj.forward

//let ha = new MultiHeadAttention(4, 25)
//let t = torch.rand([|1L;8L;int64 encodingSize|])
//ha.forward(t)
