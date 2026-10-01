import ast
import operator as op

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}
_ALLOWED_UNARY = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        fn = _ALLOWED_BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError("Unsupported operator")
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return fn(left, right)

    if isinstance(node, ast.UnaryOp):
        fn = _ALLOWED_UNARY.get(type(node.op))
        if fn is None:
            raise ValueError("Unsupported unary operator")
        return fn(_eval(node.operand))

    raise ValueError("Unsupported expression")


def evaluate_tokens(tokens):
    usable = list(tokens)
    if usable and usable[-1] == "=":
        usable.pop()

    if not usable:
        raise ValueError("Empty expression")

    expression = "".join(usable)
    expression = (
        expression
        .replace("×", "*")
        .replace("✕", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )

    tree = ast.parse(expression, mode="eval")
    value = _eval(tree)

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return expression, value
