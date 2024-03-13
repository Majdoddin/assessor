import torch
import torch.nn as nn
import torch.nn.functional as F
import boolean

#nanoGPT of Andrej Karpathy
#export PYTHONPATH="${PYTHONPATH}:path/to/nanoGPT"
import model

def is_polish_normal_form(sequence):
    """
    Checks if the given sequence is a Boolean Formula in Polish normal form.

    The Polish notation (prefix notation) for Boolean formulas requires that:
    - Each operator must come before its operands.
    - Binary operators 'And', 'Or' take exactly two operands.
    - Unary operator 'Not' takes exactly one operand.
    - 'X1', 'X2', ..., 'Xn' are considered as variables and operands.

    Args:
    sequence (list): A sequence of strings representing the formula.

    Returns:
    bool: True if the sequence is a Boolean Formula in Polish normal form, False otherwise.
    Example usage:
    sequence = ["And", "Or", "X1", "Not", "X2", "X3"]
    is_valid = is_polish_normal_form(sequence)
    is_valid
    """

    # Stack to hold the count of operands needed for each operator
    operand_stack = []

    # Process the sequence in reverse order
    for token in reversed(sequence):
        if token in {"and", "or"}:
            # Binary operators require two operands
            if len(operand_stack) >= 2:
                # Pop two operands and push one as result of the binary operation
                operand_stack.pop()
                operand_stack.pop()
                operand_stack.append(1)
            else:
                # Not enough operands for a binary operator
                return False
        elif token == "not":
            # Unary operator requires one operand
            if operand_stack:
                # Pop one operand and push one as result of the unary operation
                operand_stack.pop()
                operand_stack.append(1)
            else:
                # Not enough operands for a unary operator
                return False
        else:
            # Variables and literals count as operands
            operand_stack.append(1)

    # Valid formula in Polish notation should leave exactly one operand on the stack
    return len(operand_stack) == 1

def build_tree(tokens):
    if not tokens:
        return None

    token = tokens.pop(0)
    if token in ['and', 'or', 'not']:
        node = [token]
        # For 'not', expect exactly one operand, for 'and'/'or', expect at least two.
        expected_operands = 1 if token == 'not' else 2
        while expected_operands > 0 and tokens:
            operand, tokens = build_tree(tokens)
            node.append(operand)
            expected_operands -= 1
        return node, tokens
    else:
        # If the token is a variable, return it as is.
        return token, tokens

def to_nested_list(token_list):
    tree, remaining_tokens = build_tree(list(token_list))
    if remaining_tokens:
        raise ValueError("Invalid input: Unused tokens remain after parsing.")
    return tree

def to_parenthesized_string(tree):
    if isinstance(tree, str):
        return tree

    operator = tree[0]
    if operator == 'not':
        # For 'not', apply it to the single operand.
        return f"not {to_parenthesized_string(tree[1])}"
    else:
        # Otherwise, join the operands with the operator, applying recursion.
        operands = [to_parenthesized_string(operand) for operand in tree[1:]]
        return '(' + f" {operator} ".join(operands) + ')'

def expression_depth(expr):
    """
    Calculate the depth of a Boolean expression.

    :param expr: An expression from the boolean algebra module.
    :return: The depth of the expression.
    """
    # Base case: if the expression is a symbol, its depth is 0
    if isinstance(expr, boolean.Symbol):
        return 0

    # If the expression is a NOT operation, its depth is 1 + depth of the inner expression
    if isinstance(expr, boolean.NOT):
        return 1 + expression_depth(expr.args[0])

    # If the expression is an AND or OR operation, calculate the depth of each argument
    if isinstance(expr, (boolean.AND, boolean.OR)):
        return 1 + max(expression_depth(arg) for arg in expr.args)

    # For any other type of expression, return 0 as a default (though this should not happen)
    return 0

def sample_formula(model, max_len, itos, start_tkn, var_tkn, temperature=1.0, top_p=None, vocab_size=10, no_single_var = False):
    if top_p is not None and (top_p <= 0 or top_p >= 1):
        raise ValueError(f"top_k must in (0, ]")
    detached_tokens = None
    while True:
        #generate tokens till either formula is complete in PNF. But if  max_len tokens generated, start again.
        if detached_tokens != None: print(f"too long.")
        detached_tokens = torch.full((1, 1), start_tkn, dtype=torch.long, device=next(model.parameters()).device)
        generated_logits = []
        for idx in range(max_len):
            logits, loss = model(detached_tokens, targets = None)  #(batch_num, vocab_size)
            logits = logits[0, -1, :].unsqueeze(0)
            next_token_logits = logits / temperature

            def top_p_f(probs):
                #remove neglible probabilty among ops and var_tkn
                sorted_prob, sorted_idx = torch.sort(probs, descending=True)
                cp = torch.cumsum(sorted_prob, dim=-1)  #cumulative_probabilities
                sorted_prob[1:][cp[:-1] >= top_p] = 0
                for j in sorted_idx:
                    probs[sorted_idx[j]] = sorted_prob[j]

            sm = next_token_logits[0, start_tkn+1:var_tkn+1]
            #no two consecutive nots
            if detached_tokens[0, -1].item() == 3:
                sm[2] = float('-inf')

            sm = F.softmax(next_token_logits[0, start_tkn+1:var_tkn+1], dim=-1)
            #no start_token
            #sm[:, start_tkn] = 0
            #if no_single_var, first generated token should not be a var
            # if (idx == 0 and no_single_var):
            #     sm[:, var_tkn] = 0 #consider this by backward

            if top_p is not None:
                top_p_f(sm)

            next_tkn = (start_tkn+1) + torch.multinomial(sm, num_samples=1)   #index 0 is start_tkn

            if (next_tkn.item() == var_tkn):
                #FIXME if top_p remove vars with negligible prob
                sm = F.softmax(next_token_logits[0, var_tkn+1:], dim=-1)

                if top_p is not None:
                    top_p_f(sm)

                next_tkn = (var_tkn+1)+torch.multinomial(sm, num_samples=1)
                #if it is an unused var, take the next unused var instead.
                #makes more readable, but makes training harder
                # used_tkns  = [v for v in detached_tokens[0] if v > var_tkn]
                # if used_tkns:
                #     if next_tkn not in used_tkns:
                #         next_tkn = 1 + torch.tensor([max(used_tkns)])
                # else:
                #     next_tkn =  1 + torch.tensor([var_tkn])

            detached_tokens = torch.cat((detached_tokens, next_tkn.unsqueeze(dim=0).detach()), dim = 1)
            generated_logits.append(logits[0])

            if is_polish_normal_form([itos[t.item()] for t in detached_tokens[0, 1:]]):
                return detached_tokens[0], torch.stack(generated_logits, dim= 0)




