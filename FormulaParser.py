__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com""
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

# Third-Party Imports
import ast, operator

def safe_eval(expr, names):
    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        raise ValueError("Invalid formula syntax")
    OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }
    def _visit(n):
        if isinstance(n, ast.Expression):
            return _visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            l = _visit(n.left)
            r = _visit(n.right)
            if type(n.op) in OPS:
                return OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp):
            v = _visit(n.operand)
            if type(n.op) in OPS:
                return OPS[type(n.op)](v)
        if isinstance(n, ast.Name) and n.id in names:
            return names[n.id]
        raise ValueError(f"Unsupported expression: {expr}")
    return _visit(node)