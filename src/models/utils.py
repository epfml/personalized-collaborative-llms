from .lora import GPTLoRA


def get_model(args):
    """ Return the right model """
    if args.model == 'lora':
        if args.use_pretrained != 'none':  # Not use to resume training from checkpoint but to finetune use lora
            model = GPTLoRA.from_pretrained(args.use_pretrained, args)
        else:
            model = GPTLoRA(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
