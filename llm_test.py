from gpt4all import GPT4All

def main():
    model_name='starcoder-newbpe-q4_0.gguf'
    # model_name='orca-mini-3b-gguf2-q4_0.gguf'
    model = GPT4All(model_name, device='gpu')
    output = model.generate("develop a python function that recursively computes the sum of the first 30 terms of the Fibonacci sequence'", max_tokens=4000)
    print(output)

    
if __name__ == '__main__':
    
    main()