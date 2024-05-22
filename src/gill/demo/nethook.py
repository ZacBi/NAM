"""
Utilities for instrumenting a torch model. 用于为 torch 模型进行仪器化的工具。

Trace will hook one layer at a time.      Trace 将一次钩住一层。
TraceDict will hook multiple layers at once.  TraceDict一次钩住多个层。
subsequence slices intervals from Sequential modules.   从 Sequential 模块中切片子序列间隔
get_module, replace_module, get_parameter resolve dotted names.  get_module、replace_module、get_parameter 解析点分隔的名称。
set_requires_grad recursively sets requires_grad in module parameters.  set_requires_grad 递归设置模块参数中的 requires_grad。
"""

import contextlib
import copy
import inspect
from collections import OrderedDict

import torch


class Trace(contextlib.AbstractContextManager):
    """
    为了在计算给定网络时保留指定层的输出
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and 可以直接传递一个层模块而不需要层名称，其输出将会被保留。默认情况下，会返回对输出对象的直接引用，但可以通过选项来控制
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be    clone=True - 保留输出的副本，如果您希望在稍后网络可能就地修改输出之前查看输出，则这可能很有用
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By    detach=True - 保留分离的引用或副本。（默认情况下，该值将保持连接到图形。）
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.                 retain_grad=True - 请求在输出上保留梯度。在调用backward()后，ret.output.grad会被填充。
        retain_output=False - can disable retaining the output.     retain_input=True - 也保留输入
        edit_output=fn - calls the function to modify the output    retain_output=False - 可以禁用保留输出
            of the layer before passing it the rest of the model.   edit_output=fn - 在将输出传递给模型的其余部分之前调用该函数以修改层的输出。fn可以选择接受（原始输出，层名称）参数
            fn can optionally accept (output, layer) arguments      stop=True - 在运行该层后抛出StopForward异常，这允许仅运行模型的一部分   
            for the original output and the layer name.            
        stop=True - throws a StopForward exception after the layer  感觉
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self, 
        
        module,
        
        layer=None,
        
        retain_output=True,
        
        retain_input=False,
        
        clone=False,
        
        detach=False,
        
        retain_grad=False,
        
        edit_output=None,
        
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        一种用闭包替换前向方法的方法，该闭包拦截调用，并跟踪挂钩以便可以恢复
        闭包：闭包（closure）是指一个函数与其相关的引用环境的组合。闭包允许函数访问其自身定义范围外的变量，即使创建闭包的上下文已经不存在
        前向方法： 将操作或请求传递给另一个对象或组件的方法
        闭包拦截调用，并跟踪挂钩以便可以恢复这句话的意思是：
        闭包在实现时会拦截函数的调用，并记录相关的信息，例如执行的时间点、传入的参数等。这些记录被称为“挂钩”（hooks），可以用来在之后恢复函数的执行状态，重新执行相同的操作。
        闭包通过捕获和保存这些信息，可以在需要时重新创建函数的上下文，以便继续或重新执行特定的操作。
        """
        
        
        retainer = self
        
        self.layer = layer
        if layer is not None:
            
            
            module = get_module(module, layer)
        
            
        
        
        
        
        def retain_hook(m, inputs, output):
            
            if retain_input:
                retainer.input = recursive_copy( 
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  
            if edit_output:
                
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output
        
        
        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()
class Trace_1(contextlib.AbstractContextManager):
    def __init__(
        self, 
        
        module=None,
        
        layer=None,
        
        retain_output=True,
        
        retain_input=False,
        
        clone=False,
        
        detach=False,
        
        retain_grad=False,
        
        edit_output=None,
        
        stop=False,
        hand_act = None
    ):
        retainer = self
        
        self.layer = layer
        self.act = []
        if module is not None:
            print("no")
        if layer is not None:
            
            
            module = get_module(module, layer)
        
            
        
        
        
        
        def retain_hook(m, inputs, output):
            
            if retain_input:
                retainer.input = recursive_copy( 
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  
            if edit_output:
                
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            self.act.append(output)
            return output

        
        
        # self.registered_hook = module.register_forward_hook(retain_hook)
        self.hand_act = [module.model.lm.model.decoder.layers[n].activation_fn.register_forward_hook(retain_hook) for n in
                           range(32)]
        #self.hand_act = [module.model.lm.model.decoder.layers[0].activation_fn.register_forward_hook(retain_hook)]
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        # self.registered_hook.remove()
        for h in self.hand_act:
            h.remove()



class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        
        module,
        
        layers=None,
        
        retain_output=True,
        
        retain_input=False,
        
        clone=False,
        
        detach=False,
        
        retain_grad=False,
        
        edit_output=None,
        
        stop=False,
    ):
        self.stop = stop
        
        
        def flag_last_unseen(it):
            try:
                
                it = iter(it)
                
                prev = next(it)
                
                seen = set([prev])
            except StopIteration:
                return
            
            for item in it:
                
                if item not in seen:
                    
                    yield False, prev
                    
                    seen.add(item)
                    
                    prev = item
            yield True, prev
        
        for is_last, layer in flag_last_unseen(layers):
            
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """
    """
    如果在运行网络时唯一需要的输出是保留的子模块，那么通过使用 Trace(submodule, stop=True) 可以立即停止执行，并抛出 StopForward() 异常。当 Trace 作为上下文管理器使用时，它会捕获该异常，并可以按照以下方式使用：
    python
    with Trace(net, layername, stop=True) as tr:
        net(inp)  
        print(tr.output)
    在这段代码中，net 是神经网络模型，inp 是输入数据。通过 Trace 的上下文管理器，可以控制网络执行到指定的 layername 层后立即停止，并且通过 tr.output 可以获取到停止后的输出结果。
    """
    pass



def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    """
    将一个张量或包含张量的对象的引用复制一份，可以选择性地对张量进行分离和克隆操作。如果 `retain_grad` 参数为 true，则会标记原始张量以保留梯度信息。
    输入的x就是上面提到的张量或柏寒张量的对象的引用
    """
    
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        
        if retain_grad:
            if not x.requires_grad:
                
                x.requires_grad = True
            
            
            
            x.retain_grad()
        elif detach:
            
            x = x.detach()
        if clone:
            
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."






def subsequence(
    
    sequential,
    
    first_layer=None,
    
    last_layer=None,
    
    after_layer=None,
    
    upto_layer=None,
    
    single_layer=None,
    
    share_weights=False,
):
    """
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    """
    
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    """
    """
    递归辅助函数用于支持进入带有点分隔层名称的子序列（subsequence()）。在这个辅助函数中，first、last、after 和 upto 是根据点分隔拆分而成的名称数组。只能进入嵌套的 Sequentials。
    """
    
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    
    including_children = (first is None) and (after is None)
    
    included_children = OrderedDict()
    
    
    
    
    
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    
    for name, layer in sequential._modules.items():
        
        if name == F:
            first = None
            including_children = True
        
        if name == A and AN is not None:  # just like F if not a leaf.
            after = None
            including_children = True
        
        if name == U and UN is None:
            upto = None
            including_children = False
        
        if including_children: 
            # AR = full name for recursive descent if name matches.
            
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # just like L if not a leaf.
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)
def get_module(model, name):
    return model

def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    # original_module = getattr(model, attr_name)
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    
    
    argspec = inspect.getfullargspec(fn)
    
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)
