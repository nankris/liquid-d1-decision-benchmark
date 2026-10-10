"""USD per million tokens. Set explicit prices before running hosted chat."""
D1_INPUT_PER_MILLION = 0.04
# TODO: Add verified model prices, e.g. 'openrouter:provider/model': {'input': ..., 'output': ...}.
CHAT_PRICES = {}

def rates(model):
    if model in ('d1-local', 'd1-hosted', 'openrouter-d1'):
        return {'input': D1_INPUT_PER_MILLION, 'output': 0.0}
    return CHAT_PRICES.get(model)

def cost(model, input_tokens, output_tokens):
    price = rates(model)
    if price is None:
        return None
    return (input_tokens * price['input'] + output_tokens * price['output']) / 1_000_000
