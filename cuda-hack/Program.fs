// See https://aka.ms/new-console-template for more information

open TorchSharp

[<EntryPoint>]
let main argv =
    printfn "Hello, World!"
    printfn $"torch cuda: {torch.cuda.is_available()}"
    0