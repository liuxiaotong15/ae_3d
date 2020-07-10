"""
Functions that use multiple times
"""

from torch import nn
import torch, os
import numpy as np

commit_id = str(os.popen('git --no-pager log -1 --oneline').read()).split(' ', 1)[0]

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        # nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.xavier_normal_(layer.weight, gain=1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # loss = lnet.loss_func(
    #     v_wrap(np.vstack(bs)),
    #     v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
    #     v_wrap(np.array(buffer_v_target)[:, None]))
    loss = lnet.loss_func(
        v_wrap(np.array(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name, r_history, global_max_ep_r):
    ret = 0
    with global_ep.get_lock():
        global_ep.value += 1
        ret = global_ep.value
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    with global_max_ep_r.get_lock():
        global_max_ep_r.value = max(global_max_ep_r.value, ep_r)
    res_queue.put(global_ep_r.value)
    if global_ep.value % 1000 == 0:
        print(
        commit_id,
        name,
        "Ep:", global_ep.value,
        "| Ep_r_ma: %.6f" % global_ep_r.value,
        "| Ep_r_cur: %.6f" % ep_r,
        '| Ep_r_max: %.6f' % global_max_ep_r.value
        )
    # for rh in r_history:
    #     print(rh)
    # print('-' * 100)
    return ret

