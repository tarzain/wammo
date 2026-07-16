import torch

from wammo.model.flow import euler_step_toward_data, interpolate, velocity_target


def test_flow_signed_direction_regression():
    x0 = torch.tensor([2.0, -1.0])
    noise = torch.tensor([10.0, 3.0])
    t = torch.tensor(0.25)
    xt = interpolate(x0, noise, t)
    v = velocity_target(x0, noise)
    assert torch.allclose(xt, torch.tensor([4.0, 0.0]))
    restored = euler_step_toward_data(xt, v, dt=0.25)
    torch.testing.assert_close(restored, x0)

