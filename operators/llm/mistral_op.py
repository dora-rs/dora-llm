from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from dora import DoraStatus

import pyarrow as pa

model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"


class Operator:
    def __init__(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
        # Load model
        self.model = AutoAWQForCausalLM.from_quantized(
            model_name_or_path,
            fuse_layers=True,
            trust_remote_code=False,
            safetensors=True,
        )

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            values = dora_event["value"].to_pylist()
            if dora_event["id"] == "prompt":
                output = self.prompt(
                    "you're a code expert. Respond with code only.",
                    values[0],
                )
                send_output("reply_prompt", pa.array([output]))
        return DoraStatus.CONTINUE

    def prompt(self, system_message, prompt):
        prompt_template = f"""<|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
        """
        token_input = self.tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
        # Generate output
        generation_output = self.model.generate(
            token_input,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=512,
        )
        # Get the tokens from the output, decode them, print them
        outputs = ""
        for token_output in generation_output:
            outputs += self.tokenizer.decode(token_output)

        # Get text between im_start and im_end

        text_output = outputs.split("<|im_start|> assistant\n")[1].split("<|im_end|>")[0]
        return text_output
