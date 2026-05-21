from string import ascii_lowercase

from pytensor.graph import node_rewriter
from pytensor.tensor import einsum
from pytensor.tensor.basic import constant
from pytensor.tensor.basic import stack as tstack
from pytensor.tensor.einsum import Einsum
from pytensor.tensor.math import matmul, variadic_mul
from pytensor.tensor.rewriting.ofg import inline_ofg_node
from pytensor.tensor.shape import shape_tuple, specify_shape
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.math import Dot
from pytensor.xtensor.rewriting.utils import register_lower_xtensor


def _combined_size(tensor, axes):
    """Return the product of ``tensor``'s sizes at ``axes`` as a scalar variable.

    Static dims are multiplied together in Python so the result is a single
    constant when possible, instead of a chain of ``Mul`` nodes over individual
    ``ScalarConstant``s.
    """
    sizes = shape_tuple(tensor)
    static_product = 1
    runtime_terms = []
    for axis in axes:
        s = tensor.type.shape[axis]
        if s is None:
            runtime_terms.append(sizes[axis])
        else:
            static_product *= s
    if not runtime_terms:
        return constant(static_product, dtype="int64")
    if static_product == 1:
        return variadic_mul(*runtime_terms)
    return variadic_mul(constant(static_product, dtype="int64"), *runtime_terms)


def _matmul_lower(x, y, x_dims, y_dims, contracted, out_dims):
    """Lower a 2-operand xtensor.dot to a single ``matmul`` when possible.

    Returns ``None`` for pure outer products (no shared contracted dim), in
    which case the caller should fall back to ``einsum``.

    Shared dims become matmul batch axes; multi-dim M / K / N groups are
    collapsed via ``reshape`` and uncollapsed after the matmul.
    """
    x_dims = list(x_dims)
    y_dims = list(y_dims)
    contracted = list(contracted)

    # Dims listed as contracted but present in only one operand are pure
    # reductions (xarray semantics): sum them out before the matmul.
    x_only_contracted = [d for d in contracted if d in x_dims and d not in y_dims]
    y_only_contracted = [d for d in contracted if d in y_dims and d not in x_dims]
    if x_only_contracted:
        x = x.sum([x_dims.index(d) for d in x_only_contracted])
        for d in x_only_contracted:
            x_dims.remove(d)
    if y_only_contracted:
        y = y.sum([y_dims.index(d) for d in y_only_contracted])
        for d in y_only_contracted:
            y_dims.remove(d)

    k_dims = [d for d in contracted if d in x_dims and d in y_dims]
    if not k_dims:
        # No shared contracted dim -> outer product; not matmul-shaped.
        return None

    shared = [d for d in x_dims if d in y_dims and d not in k_dims]
    x_only = [d for d in x_dims if d not in y_dims and d not in k_dims]
    y_only = [d for d in y_dims if d not in x_dims and d not in k_dims]

    # Cache original per-dim sizes (static if possible) so we can uncollapse
    # M / N groups after matmul.
    dim_sizes: dict[str, object] = {}
    x_shape = shape_tuple(x)
    y_shape = shape_tuple(y)
    for axis, d in enumerate(x_dims):
        dim_sizes[d] = x_shape[axis]
    for axis, d in enumerate(y_dims):
        dim_sizes.setdefault(d, y_shape[axis])

    # Permute to (shared..., x_only..., k...) and (shared..., k..., y_only...).
    x_perm = [x_dims.index(d) for d in shared + x_only + k_dims]
    if x_perm != list(range(len(x_dims))):
        x = x.transpose(x_perm)
    y_perm = [y_dims.index(d) for d in shared + k_dims + y_only]
    if y_perm != list(range(len(y_dims))):
        y = y.transpose(y_perm)

    n_shared = len(shared)
    n_k = len(k_dims)

    # Collapse multi-dim outer / contracted groups so the trailing two axes
    # are exactly the (M, K) / (K, N) matmul axes. Empty outer groups get a
    # dummy length-1 axis (squeezed back out below). ``k_dims`` is always
    # non-empty here.
    def _collapse(tensor, outer_axes, k_axes, is_lhs):
        n_outer = len(outer_axes)
        if n_outer == 1 and n_k == 1:
            # Already in the right layout: just a single M (or N) and a single K.
            return tensor

        new_shape = list(shape_tuple(tensor)[:n_shared])
        if n_outer == 0:
            outer_size = constant(1, dtype="int64")
        else:
            outer_size = _combined_size(tensor, outer_axes)
        k_size = _combined_size(tensor, k_axes)

        if is_lhs:
            new_shape += [outer_size, k_size]
        else:
            new_shape += [k_size, outer_size]
        return tensor.reshape(tstack(new_shape))

    # After the transposes above, x's outer axes are [n_shared, n_shared+len(x_only))
    # and its k axes are [n_shared+len(x_only), n_shared+len(x_only)+n_k). For y,
    # k axes are [n_shared, n_shared+n_k) and outer axes are [n_shared+n_k, ...).
    x_outer_axes = range(n_shared, n_shared + len(x_only))
    x_k_axes = range(n_shared + len(x_only), n_shared + len(x_only) + n_k)
    y_k_axes = range(n_shared, n_shared + n_k)
    y_outer_axes = range(n_shared + n_k, n_shared + n_k + len(y_only))

    x = _collapse(x, x_outer_axes, x_k_axes, is_lhs=True)
    y = _collapse(y, y_outer_axes, y_k_axes, is_lhs=False)

    res = matmul(x, y)

    # If x_only / y_only were empty, M / N is a dummy length-1 axis: squeeze it.
    squeeze_axes: list[int] = []
    if not y_only:
        squeeze_axes.append(res.type.ndim - 1)
    if not x_only:
        squeeze_axes.append(res.type.ndim - 2)
    if squeeze_axes:
        res = res.squeeze(tuple(squeeze_axes))

    # Uncollapse multi-dim M / N groups back to their original axes.
    if len(x_only) > 1 or len(y_only) > 1:
        new_shape = list(shape_tuple(res)[:n_shared])
        new_shape.extend(dim_sizes[d] for d in x_only)
        new_shape.extend(dim_sizes[d] for d in y_only)
        res = res.reshape(tstack(new_shape))

    current_dims = shared + x_only + y_only
    if current_dims != list(out_dims):
        res = res.transpose([current_dims.index(d) for d in out_dims])

    return res


@register_lower_xtensor
@node_rewriter(tracks=[Dot])
def lower_dot(fgraph, node):
    """Lower an xtensor ``Dot`` to plain tensor ops.

    Matmul-shaped contractions (anything with at least one shared contracted
    dim) lower to a single ``matmul`` (``Blockwise(Dot)``), with reshape only
    where multiple M / K / N dims must be collapsed and uncollapsed. Pure
    outer products fall back to ``einsum`` and inline the resulting
    ``OpFromGraph`` so ``pytensor.grad`` sees flat ops.
    """
    [x, y] = node.inputs
    [out] = node.outputs

    x_tensor = tensor_from_xtensor(x)
    y_tensor = tensor_from_xtensor(y)

    out_tensor = _matmul_lower(
        x_tensor,
        y_tensor,
        x.type.dims,
        y.type.dims,
        node.op.dims,
        out.type.dims,
    )

    if out_tensor is None:
        # Outer-product fallback: build an einsum string and lower via einsum.
        all_dims = list(dict.fromkeys(x.type.dims + y.type.dims + out.type.dims))
        if len(all_dims) > len(ascii_lowercase):
            raise ValueError("Too many dimensions to map to einsum subscripts")
        dim_to_char = dict(zip(all_dims, ascii_lowercase))
        x_subs = "".join(dim_to_char[d] for d in x.type.dims)
        y_subs = "".join(dim_to_char[d] for d in y.type.dims)
        out_subs = "".join(dim_to_char[d] for d in out.type.dims)
        einsum_str = f"{x_subs},{y_subs}->{out_subs}"
        out_tensor = einsum(einsum_str, x_tensor, y_tensor)
        if out_tensor.owner is not None and isinstance(out_tensor.owner.op, Einsum):
            # Inline so grad / ShapeOpt see flat ops instead of an OFG wrapper.
            [out_tensor] = inline_ofg_node(out_tensor.owner)

    out_tensor = specify_shape(out_tensor, out.type.shape)
    return [xtensor_from_tensor(out_tensor, out.type.dims)]
