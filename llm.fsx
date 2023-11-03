// Note this hack:
//
// Polyglot notebooks extension won't load CUDA libraries if installed as below:
//   #r "nuget: libtorch-cuda-12.1-linux-x64, 2.1.0.1"
//   #r "nuget: TorchSharp, 0.101.1"
//
// instead you need to `dotnet publish -c Release ./cuda-hack`
// so we can load the .dll manually

#I "cuda-hack/bin/Release/net7.0/publish/"
#r "cuda-hack.dll"
#r "TorchSharp.dll"


open System
open TorchSharp

let forceDevice =
    //Some torch.CPU
    None

let device = 
    match forceDevice with
    | Some device -> device
    | None ->
        if torch.cuda.is_available() then
            printfn "Using CUDA / GPU"
            torch.CUDA
        else
            printfn "Using CPU"
            torch.CPU



type CausalSelfAttention(nHeads:int64, encodingSize:int64, hasBias:bool, dropout:float) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("CausalSelfAttention")
    let c_attn = torch.nn.Linear(encodingSize, encodingSize * 3L, hasBias=hasBias, dtype=torch.float32)
    let resid_dropout = torch.nn.Dropout(dropout)
    do
        this.RegisterComponents()
        //if device.``type`` = DeviceType.CUDA then this.``to``(device) |> ignore    

    override this.forward(xs) =
        let xs_shape = xs.shape
        match xs_shape with
        | [|B;T;C|] ->
            let qkv = c_attn.forward(xs).view(B,T,nHeads*3L,C/nHeads).transpose(1,2).split(nHeads, dim=1)
            use y = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_casual=true, p=dropout)
            for t in qkv do t.Dispose()
            use y = y.transpose(1,2).contiguous().view(xs_shape)
            resid_dropout.forward(y)
        | _ -> failwith "Expected input shape [B, T, C]"

    (* // base forward without opt
    member this.forward(xs) =
        let qkv = c_attn.forward(xs).split(encodingSize, dim=2)
        printfn "shape1 = %A" (qkv.[0].shape)
        let xs_shape = xs.shape
        match xs_shape, qkv with
        | [|B;T;C|], [|q;k;v|]->
            let headSize = [|B;T;nHeads;C/nHeads|]
            use k = k.view(headSize).transpose(1,2) // (B, nh, T, hs)
            use q = q.view(headSize).transpose(1,2)
            use v = v.view(headSize).transpose(1,2)
            use y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_casual=true, p=dropout)
            use y = y.transpose(1,2).contiguous().view(xs_shape)
            resid_dropout.forward(y)
        | _ -> failwith "Expected input shape [B, T, C]"
    *)        

type MLP(encodingSize:int64, hasBias:bool, dropout:float) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("MLP")
    let net = torch.nn.Sequential([|
        "c_fc", torch.nn.Linear(encodingSize, encodingSize * 4L, hasBias=hasBias, dtype=torch.float32) :> torch.nn.Module<_,_>
        "gelu", torch.nn.GELU() :> _
        "c_proj", torch.nn.Linear(encodingSize * 4L, encodingSize, hasBias=hasBias, dtype=torch.float32) :> _
        "dropout", torch.nn.Dropout(dropout) :> _
    |])
    do
        this.RegisterComponents()
        //if device.``type`` = DeviceType.CUDA then this.``to``(device) |> ignore    

    override this.forward(xs) = net.forward(xs)

type Block(nHeads, encodingSize:int64, hasBias:bool, dropout) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("Block")
    let ln1 = torch.nn.LayerNorm(encodingSize, dtype=torch.float32)
    let attn = new CausalSelfAttention(nHeads, encodingSize, hasBias, dropout)
    let ln2 = torch.nn.LayerNorm(encodingSize, dtype=torch.float32)
    let mlp = new MLP(encodingSize, hasBias, dropout)
    do
        this.RegisterComponents()
        //if device.``type`` = DeviceType.CUDA then this.``to``(device) |> ignore
    override this.forward(xs) =
        use ln1 = ln1.forward xs
        use attn = attn.forward ln1
        use xs = xs + attn
        use ln2 = ln2.forward xs
        use mlp = mlp.forward ln2
        xs + mlp

type EmbeddingModel(vocabSize, blockSize, encodingSize) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("EmbeddingModel")
    let tok_emb = torch.nn.Embedding(vocabSize, encodingSize, dtype=torch.float32)
    let pos_emb = torch.nn.Embedding(blockSize, encodingSize, dtype=torch.float32)
    let positions = torch.arange(0L, blockSize, dtype=torch.int64, requires_grad=false)
    do
        this.RegisterComponents()
        //if device.``type`` = DeviceType.CUDA then this.``to``(device) |> ignore    

    override _.forward (input: torch.Tensor) =
        use tok_emb = tok_emb.forward(input)
        use pos_emb = pos_emb.forward(positions)
        tok_emb + pos_emb

type LanguageModel(nLayers, nHeads, nEmbed, vocabSize:int, blockSize:int, hasBias, dropout) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("LanguageModel")
    let ``1`` = Scalar.op_Implicit(1.0f)
    let layers = torch.nn.Sequential([|
        yield "embed", new EmbeddingModel(vocabSize, blockSize, nEmbed) :> torch.nn.Module<_,_>
        for layer in 1..nLayers do
            yield $"block{layer}", new Block(nHeads, nEmbed, hasBias, dropout) :> _
        yield "de_embed", torch.nn.Linear(nEmbed, vocabSize, dtype=torch.float32)
    |])

    do
        this.RegisterComponents()
        //if device.``type`` = DeviceType.CUDA then this.``to``(device) |> ignore    

    override _.forward (input: torch.Tensor) =
        layers.forward input

