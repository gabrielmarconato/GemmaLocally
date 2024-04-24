import onnxruntime_genai as og
import argparse
import time
from collections import Counter

def main(args):
    print("Loading model...")

    # Carregue a pasta do modelo aqui
    model = og.Model(f'model/cpu_and_mobile/cpu-int4-rtn-block-32')

    print("--- Model loaded ---")

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    print("--- Tokenizer created ---")
    print()
    
    
    search_options = {name:getattr(args, name) for name in ['do_sample', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}

    # Textos de Entrada do Modelo ---------------------------------------
    System_Prompt = "You are a data engineer specialist"
    User_Prompt = "teach me how to create a datawarehouse"
    # --------------------------------------------------------------------

    # Tamanho Máximo e Mínimo das Respostas do Modelo --------------------
    Response_Max_Size = 512
    Response_Min_Size = 32
    # --------------------------------------------------------------------


    words = (System_Prompt + User_Prompt).split()
    wordCount = Counter(words)

    search_options['max_length'] = wordCount.total() + Response_Max_Size
    search_options['min_length'] = wordCount.total() + Response_Min_Size

    input_tokens = tokenizer.encode(System_Prompt + User_Prompt)

    params = og.GeneratorParams(model)
    params.try_use_cuda_graph_with_max_batch_size(0)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)
    print("--- Generator created ---")

    print("Running generation loop ...")
    print()

    with open('resposta.txt', 'w') as file:
        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                
                new_token = generator.get_next_tokens()[0]
                file.write(tokenizer_stream.decode(new_token))

        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        except Exception as e:
            print("An error occurred:", str(e))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_random_sampling', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-s', '--system_prompt', type=str, default='', help='Prepend a system prompt to the user input prompt. Defaults to empty')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
    args = parser.parse_args()
    main(args)