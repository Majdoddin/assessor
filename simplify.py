import boolean

def deMorgan(formula):
    # Base case: If the formula is a single variable or an empty list, return it.
    if not formula or isinstance(formula, str):
        return formula

    # Simplify all children first.
    simplified_children = [deMorgan(child) for child in formula[1:]]

    # Now apply simplification rules to the current node.
    op = formula[0]
    if op == 'not' and simplified_children and simplified_children[0][0] == 'not':
        # Apply double negation elimination.
        return simplified_children[0][1]  # Return the grandchild directly.
    elif op in ['and', 'or']:
        # Check if De Morgan's law can be applied:
        # If all operands are negations, switch the operator and negate all children.
        if all(child[0] == 'not' for child in simplified_children if child):
            new_op = 'or' if op == 'and' else 'and'
            new_children = [child[1] for child in simplified_children]
            return ['not', [new_op] + new_children]
        else:
            return [op] + simplified_children

    # If the operation is not 'and' or 'or', return as is (e.g., for 'not' with a single operand).
    return [op] + simplified_children

# Example usage
# formulas = [['and', ['not', ['not', 'x1']], ['not', 'x2']],
#             ['and', ['not', 'x1'], ['not', 'x2']],]
# for f in formulas:
#     print(deMorgan(f))  # Output should be ['or', 'x1', ['not', 'x2']]

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

# Example usage:
# formulas = [['and', 'and', 'X1', 'and', 'X1', 'X1', 'X3'],
#     ['and', 'x1', 'not', 'x2'],
#            ['x1'],
#            [],
#            ['x1', 'x2'],
#            ]
# for f in formulas:
#     nested_list = to_nested_list(f)
#     print(nested_list)  # Should output: ['and', 'x1', ['not', 'x2']]

def flatten_tree(tree):
    if isinstance(tree, str):
        return [tree]
    
    # Assume the first item is the operator and the rest are operands.
    flat_list = [tree[0]]  # Start with the operator
    for operand in tree[1:]:
        # Recursively flatten each operand and add it to the flat list.
        flat_list.extend(flatten_tree(operand))
    
    return flat_list

# Example usage:
# nested_list = ['and', 'x1', ['not', 'x2']]
# flat_list = flatten_tree(nested_list)
# print(flat_list)  # Expected output: ['and', 'x1', 'not', 'x2']

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



# Example usage of the function
# algebra = boolean.BooleanAlgebra()
# example_expr = algebra.parse('A and (B or C)')
# depth = expression_depth(example_expr)
# print(depth)

# Example usage:
# nested_list = ['and', ['and', 'x1', ['not', 'x2']], ['or', 'x3', ['not', 'x4']]]
# output_string = to_parenthesized_string(nested_list)
# print(output_string)  # Expected output: "(x1 and not x2) and (x3 or not x4)"

# algebra = boolean.BooleanAlgebra()
# algebra.parse(output_string, simplify=False)




